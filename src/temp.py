# move_from_gt_list.py
import os
import shutil
from pathlib import Path

# === 사용자 설정 ===
GT_LIST = "list/gt_hackerton_xd.txt"   # 파일명 목록(한 줄에 하나)
SOURCE_DIRS = [
    "../VAD_dataset/XDClipFeatures/XDTestClipFeatures",
    # 필요하면 소스 디렉터리를 더 추가
]
DEST_DIR = "../hackerton2-1/CLIPFeatures"     # 이동 대상 폴더

# 동작 옵션
DRY_RUN = False          # True면 실제 이동하지 않고 무엇을 할지 출력만 함
KEEP_STRUCTURE = False   # True면 소스의 하위 폴더 구조를 그대로 보존하여 이동
OVERWRITE = False        # 대상에 동일 파일명이 있으면 덮어쓸지 여부


def read_gt_names(gt_path: str):
    names = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            names.append(name)
    return names


def index_sources_by_basename(source_dirs, wanted_names):
    """
    소스 디렉터리들을 순회하며 wanted_names(파일명)과 일치하는 파일을 찾는다.
    동일 basename이 여러 위치에 있으면 가장 먼저 발견한 경로를 기록하고,
    이후 발견은 duplicates에 적어둔다.
    """
    wanted = set(wanted_names)
    found_map = {}     # {basename: Path}
    duplicates = {}    # {basename: [Path, ...]}

    for root in source_dirs:
        root = Path(root)
        if not root.is_dir():
            print(f"[WARN] 소스 디렉터리 미존재: {root}")
            continue
        # 확장자 지정이 필요하면 '*.npy' 등으로 바꿔도 됨
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            bn = p.name
            if bn in wanted:
                if bn not in found_map:
                    found_map[bn] = p
                else:
                    duplicates.setdefault(bn, []).append(p)

    return found_map, duplicates


def safe_move(src: Path, dest_dir: Path, keep_structure=False, overwrite=False, dry_run=False):
    dest_dir = dest_dir if keep_structure else dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    if keep_structure:
        # 소스 루트 구조 보존을 원한다면, 호출부에서 상대경로 계산이 필요함
        # 여기서는 파일명만으로 이동(구조 보존 X). 구조 보존 옵션을 쓰려면
        # 호출부에서 dest_path를 직접 만들어 전달하는 방식을 추천.
        pass

    dest_path = dest_dir / src.name
    if dest_path.exists():
        if overwrite:
            if not dry_run:
                shutil.move(str(src), str(dest_path))
            print(f"[OVERWRITE] {src} -> {dest_path}")
        else:
            print(f"[SKIP: exists] {dest_path}")
    else:
        if not dry_run:
            shutil.move(str(src), str(dest_path))
        print(f"[MOVE] {src} -> {dest_path}")


def main():
    names = read_gt_names(GT_LIST)
    print(f"[INFO] 이동 대상 파일 수(이름 기준): {len(names)}")

    found_map, duplicates = index_sources_by_basename(SOURCE_DIRS, names)
    not_found = [n for n in names if n not in found_map]

    dest_dir = Path(DEST_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 이동
    moved = 0
    for nm in names:
        if nm not in found_map:
            continue
        src = found_map[nm]
        safe_move(src, dest_dir, keep_structure=KEEP_STRUCTURE, overwrite=OVERWRITE, dry_run=DRY_RUN)
        moved += 1

    # 리포트
    print("\n=== 요약 ===")
    print(f"이동 완료: {moved} / {len(names)}")
    if not_found:
        print(f"미발견: {len(not_found)}개 (아래 목록)")
        for x in not_found[:50]:
            print("  -", x)
        if len(not_found) > 50:
            print("  ... (생략)")

    if duplicates:
        print(f"\n중복 발견 파일명: {len(duplicates)}개")
        for bn, paths in list(duplicates.items())[:20]:
            print(f"  - {bn}")
            for p in paths[:3]:
                print(f"      {p}")
            if len(paths) > 3:
                print("      ... (생략)")


if __name__ == "__main__":
    main()
