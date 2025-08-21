"""
스도쿠 문제 해결 알고리즘 비교: Pure Brute-force vs Backtracking (Numba JIT 최적화)
=================================================================================

이 스크립트는 스도쿠 퍼즐에 대해 두 가지 해결 전략, 즉 순수 완전 탐색(Pure Brute-force)과
백트래킹(Backtracking) 알고리즘의 성능을 비교합니다.
성능 지표로는 실행 시간과 탐색한 노드 수를 사용하며, Numba를 이용한 JIT 컴파일을 통해
계산 집약적인 부분을 최적화합니다.

주요 기능:
1.  스도쿠 퍼즐 문자열을 NumPy 배열 및 비트마스크 형태로 변환합니다.
2.  Numba JIT으로 컴파일된 백트래킹 및 완전 탐색 함수를 구현합니다.
3.  멀티프로세싱을 사용하여 여러 스도쿠 퍼즐을 병렬로 처리하고 성능 데이터를 수집합니다.
4.  수집된 데이터를 Pandas DataFrame으로 정리하고 CSV 파일로 저장합니다.
5.  Matplotlib을 사용하여 빈칸 수에 따른 실행 시간 및 노드 수 비교 그래프를 생성합니다.

출력:
-   `sudoku_compare_results.csv`: 각 퍼즐별 실행 시간, 노드 수, 비율 등의 결과 데이터.
-   3개의 비교 그래프 (PNG 파일로 저장하거나 화면에 표시).
"""

import os, sys, time, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl
import matplotlib.font_manager as fm

# ─── Numba JIT 설정 ───
try:
    from numba import njit, set_num_threads
    NUMBA = True
    set_num_threads(1) # Numba 내부 스레딩을 1로 고정 (멀티프로세싱과 충돌 방지 및 예측 가능한 성능)
except ImportError:
    NUMBA = False
    def njit(*a, **k):          # type: ignore
        def deco(f): return f
        return deco

# ─── 문자열 → NumPy 보드 & 빈칸 ndarray ───
def to_board_mask(pz: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    스도쿠 퍼즐 문자열을 NumPy 배열 및 제약 조건 검사를 위한 비트마스크로 변환합니다.

    Args:
        pz (str): 81자리 스도쿠 퍼즐 문자열. 빈칸은 '0' 또는 '.'으로 표시.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - board (np.ndarray): 9x9 스도쿠 보드 (int8).
            - row_mask (np.ndarray): 각 행의 숫자 사용 현황 비트마스크 (uint16).
            - col_mask (np.ndarray): 각 열의 숫자 사용 현황 비트마스크 (uint16).
            - box_mask (np.ndarray): 각 3x3 박스의 숫자 사용 현황 비트마스크 (uint16).
            - empty_cells (np.ndarray): 빈칸의 (행, 열) 좌표 배열 (int8).
    """
    board = np.zeros((9, 9), np.int8)
    row_mask = np.zeros(9, np.uint16) # 각 행에 사용된 숫자를 비트마스크로 저장
    col_mask = np.zeros(9, np.uint16) # 각 열에 사용된 숫자를 비트마스크로 저장
    box_mask = np.zeros(9, np.uint16) # 각 3x3 박스에 사용된 숫자를 비트마스크로 저장
    empty: List[Tuple[int, int]] = []
    for i, ch in enumerate(pz):
        r, c = divmod(i, 9)
        if ch in ('0', '.'):
            empty.append((r, c))
        else:
            n = int(ch); board[r, c] = n; bit = 1 << n
            row_mask[r] |= bit; col_mask[c] |= bit; box_mask[(r // 3)*3 + c // 3] |= bit
    return board, row_mask, col_mask, box_mask, np.array(empty, np.int8)

# ─── Backtracking (JIT) ───
@njit(cache=True) if NUMBA else (lambda f: f)
def bt_jit(empty, d, board, row, col, box):
    if d == empty.shape[0]:
        return True
    r, c = empty[d]; b = (r // 3)*3 + c // 3
    for n in range(1, 10):
        bit = 1 << n
        if (row[r]&bit) or (col[c]&bit) or (box[b]&bit): continue
        board[r, c] = n
        row[r] |= bit; col[c] |= bit; box[b] |= bit
        if bt_jit(empty, d+1, board, row, col, box): return True
        board[r, c] = 0
        row[r] ^= bit; col[c] ^= bit; box[b] ^= bit
    return False

# ─── leaf 비트마스크 검증 ───
@njit(cache=True) if NUMBA else (lambda f: f)
def board_valid_fast(flat):
    row = np.zeros(9, np.uint16)
    col = np.zeros(9, np.uint16)
    box = np.zeros(9, np.uint16)
    for idx in range(81):
        n = flat[idx]; bit = 1 << n
        r, c = idx // 9, idx % 9
        b = (r // 3)*3 + c // 3
        if (row[r]&bit) or (col[c]&bit) or (box[b]&bit):
            return False
        row[r] |= bit; col[c] |= bit; box[b] |= bit
    return True

# ─── Pure Brute-force 확장 ───
@njit(cache=True) if NUMBA else (lambda f: f)
def brute_expand(empty, d, board, cnt):   # cnt[0]=nodes, cnt[1]=solutions
    if d == empty.shape[0]:
        cnt[0] += 1
        if board_valid_fast(board.ravel()): cnt[1] += 1
        return
    r, c = empty[d]
    for n in range(1, 10):
        board[r, c] = n
        brute_expand(empty, d+1, board, cnt)
    board[r, c] = 0

def brute_worker(part):
    board, empty_arr = part
    cnt = np.zeros(2, np.int64)
    brute_expand(empty_arr, 0, board, cnt)
    return cnt[0]      # 반환 = 노드 수 (solutions 는 필요시 cnt[1])

# ─── Python Backtracking 노드 카운트 ───
def bt_count(empty_py, d, bd, row, col, box, cnt):
    if d == len(empty_py): return
    r, c = empty_py[d]; b = (r // 3)*3 + c // 3
    for n in range(1, 10):
        bit = 1 << n
        if (row[r]&bit) or (col[c]&bit) or (box[b]&bit): continue
        cnt[0] += 1
        bd[r][c] = n; row[r] |= bit; col[c] |= bit; box[b] |= bit
        bt_count(empty_py, d+1, bd, row, col, box, cnt)
        bd[r][c] = 0
        row[r] ^= bit; col[c] ^= bit; box[b] ^= bit

# ─── 워커: 퍼즐 1개 ───
FIRST_SPLIT = 3            # 9³=729 가지 분할
def run_single(task):
    idx, pz, _ = task
    board,row,col,box,empty = to_board_mask(pz)
    blanks = empty.shape[0]

    # Backtracking
    t0=time.perf_counter()
    bt_jit(empty,0,board.copy(),row.copy(),col.copy(),box.copy())
    bt_time=time.perf_counter()-t0
    bt_nodes=[0]; bt_count(empty.tolist(),0,board.tolist(),
                           [int(x) for x in row],[int(x) for x in col],
                           [int(x) for x in box],bt_nodes)

    # Brute-force 분할
    parts=[]
    def split(d,bd,emp):
        if d==FIRST_SPLIT or d==emp.shape[0]:
            parts.append((bd.copy(),emp[d:].copy()));return
        r,c=emp[d]
        for n in range(1,10):
            bd[r,c]=n; split(d+1,bd,emp)
        bd[r,c]=0
    split(0,board.copy(),empty)

    # Brute 실행
    t0=time.perf_counter()
    bf_nodes=sum(brute_worker(p) for p in parts)
    bf_time=time.perf_counter()-t0

    return {"idx":idx,"blanks":blanks,
            "bt_time":bt_time,"bf_time":bf_time,
            "bt_nodes":bt_nodes[0],"bf_nodes":bf_nodes}

# ─── 실험 ───
def run_experiment(csv):
    df=pd.read_csv(csv)
    clue_col="clue_numbers" if "clue_numbers" in df.columns else "clues"
    tasks=[(i+1,pz,cl) for i,(pz,cl) in enumerate(zip(df["quizzes"],df[clue_col]))]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        res=[f.result() for f in as_completed(ex.submit(run_single,t) for t in tasks)]
    return pd.DataFrame(sorted(res,key=lambda d:d["idx"]))

# ─── 한글 글꼴 ───
def set_korean_font():
    for name in ("NanumGothic","Malgun Gothic","Segoe UI","Arial Unicode MS"):
        if any(name in f.name for f in fm.fontManager.ttflist):
            mpl.rcParams['font.family']=name; break
    mpl.rcParams['axes.unicode_minus']=False

# ─── Main ───
if __name__=="__main__":
    mp.freeze_support()
    CSV="150_samples_2to8_blanks.csv"
    df=run_experiment(CSV)
    df["ratio"]=df["bf_nodes"]/df["bt_nodes"]

    print(df.head()); print("평균 노드 비율:",df["ratio"].mean())

    set_korean_font()

# ----- 그래프 ①: 빈칸 수에 따른 탐색 노드 수 비교 (로그 스케일) -----
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_title('빈칸 수에 따른 탐색 노드 수 비교', fontsize=16)
    ax1.set_xlabel('빈칸 수', fontsize=12)
    ax1.set_ylabel('탐색 노드 수 (로그 스케일)', fontsize=12)
    ax1.set_yscale('log') # Y축을 로그 스케일로 설정

    # Brute-force와 Backtracking의 노드 수 산점도
    ax1.scatter(df['blanks'], df['bf_nodes'], marker='x', alpha=0.7, label='Brute-force 노드')
    ax1.scatter(df['blanks'], df['bt_nodes'], alpha=0.7, label='Backtracking 노드')
    ax1.legend(loc='upper left')
    ax1.grid(True, which="both", ls="--", linewidth=0.5)

    # 두 번째 Y축 (비율) 생성
    ax2 = ax1.twinx()
    ax2.set_ylabel('노드 비율 (Brute / Back) (로그 스케일)', fontsize=12)
    ax2.set_yscale('log')
    # 비율을 점선으로 표시
    ax2.plot(df['blanks'], df['ratio'], 'k--', alpha=0.8, label='노드 비율 (Brute/Back)')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.show()


    # ----- 그래프 ②: 빈칸 수에 따른 실행 시간 비교 (로그 스케일) -----
    plt.figure(figsize=(10, 6))
    plt.title('빈칸 수에 따른 실행 시간 비교', fontsize=16)
    plt.xlabel('빈칸 수', fontsize=12)
    plt.ylabel('실행 시간 (초, 로그 스케일)', fontsize=12)
    plt.yscale('log') # Y축을 로그 스케일로 설정

    # Brute-force와 Backtracking의 실행 시간 산점도
    plt.scatter(df['blanks'], df['bf_time'], marker='x', alpha=0.7, label='Brute-force 시간')
    plt.scatter(df['blanks'], df['bt_time'], alpha=0.7, label='Backtracking 시간')

    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


    # ----- 그래프 ③: 실행 시간 대비 노드 수 산점도 (로그-로그 스케일) -----
    plt.figure(figsize=(10, 6))
    plt.title('실행 시간 대비 탐색 노드 수 관계', fontsize=16)
    plt.xlabel('탐색 노드 수 (로그 스케일)', fontsize=12)
    plt.ylabel('실행 시간 (초, 로그 스케일)', fontsize=12)
    plt.xscale('log') # X축을 로그 스케일로 설정
    plt.yscale('log') # Y축을 로그 스케일로 설정

    # Brute-force와 Backtracking의 시간-노드 관계 산점도
    plt.scatter(df['bf_nodes'], df['bf_time'], marker='x', alpha=0.7, label='Brute-force')
    plt.scatter(df['bt_nodes'], df['bt_time'], alpha=0.7, label='Backtracking')

    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


    df.to_csv("sudoku_compare_results.csv",index=False)
    print("CSV 저장 완료")
