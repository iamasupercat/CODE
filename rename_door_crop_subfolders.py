#!/usr/bin/env python3
"""
frontdoor의 bad 폴더 아래에서
  crop_high, crop_mid, crop_low
각 영역의 하위 폴더 이름을
  0,1,2,3  →  1,2,3,4
로 +1씩 변경하는 원오프 스크립트.

대상:
- /home/work/datasets/0807 ~ 1109 (일반 폴더)
- /home/work/datasets/OBB/0616, 0721 (OBB 폴더)
"""

import os


BASE_PATH = "/home/work/datasets"
OBB_BASE_PATH = os.path.join(BASE_PATH, "OBB")


def collect_date_range_folders(base_path: str, start: str, end: str):
    """
    base_path 아래 날짜 폴더 중 start~end 범위(포함)의 절대경로 리스트 반환.
    - 지원 포맷: 4자리(MMDD) 또는 8자리(YYYYMMDD)
    - 입력 길이에 맞는 폴더만 비교 대상으로 포함
    """
    if not (start.isdigit() and end.isdigit()):
        raise ValueError("date-range는 숫자만 가능합니다. 예: 0715 0805 또는 20240715 20240805")
    if len(start) != len(end) or len(start) not in (4, 8):
        raise ValueError("date-range는 4자리(MMDD) 또는 8자리(YYYYMMDD)로 동일 길이여야 합니다.")

    s_val, e_val = int(start), int(end)
    if s_val > e_val:
        s_val, e_val = e_val, s_val

    found = []
    try:
        for name in os.listdir(base_path):
            full = os.path.join(base_path, name)
            if not os.path.isdir(full):
                continue
            if not (name.isdigit() and len(name) == len(start)):
                continue
            val = int(name)
            if s_val <= val <= e_val:
                found.append(os.path.abspath(full))
    except FileNotFoundError:
        print(f"기본 경로가 존재하지 않습니다: {base_path}")
        return []

    found.sort(key=lambda p: int(os.path.basename(p)))
    return found


def rename_area_subfolders(area_path: str):
    """
    area_path 안에서 하위 폴더 0,1,2,3을 1,2,3,4로 rename.
    덮어쓰기 방지를 위해 3→4, 2→3, 1→2, 0→1 순서로 처리.
    """
    if not os.path.isdir(area_path):
        return

    print(f"  영역 폴더 처리: {area_path}")
    # 역순으로 처리하여 이름 충돌 방지
    for old_name in ["3", "2", "1", "0"]:
        src = os.path.join(area_path, old_name)
        if not os.path.isdir(src):
            continue
        new_name = str(int(old_name) + 1)
        dst = os.path.join(area_path, new_name)
        if os.path.exists(dst):
            print(f"    [경고] 이미 존재해서 건너뜀: {dst}")
            continue
        print(f"    rename: {src} -> {dst}")
        os.rename(src, dst)


def process_frontdoor_bad(target_dir: str):
    """
    target_dir 아래 frontdoor/bad 에서
    crop_high, crop_mid, crop_low 하위 폴더 rename.
    """
    frontdoor_bad = os.path.join(target_dir, "frontdoor", "bad")
    if not os.path.isdir(frontdoor_bad):
        print(f"  frontdoor/bad 없음: {frontdoor_bad}")
        return

    print(f"\n[처리] {frontdoor_bad}")
    # 원본 크롭 폴더와 증강 크롭 폴더 모두 처리
    for area in ["crop_high", "crop_mid", "crop_low",
                 "crop_high_aug", "crop_mid_aug", "crop_low_aug"]:
        area_path = os.path.join(frontdoor_bad, area)
        if not os.path.isdir(area_path):
            print(f"  영역 폴더 없음 (건너뜀): {area_path}")
            continue
        rename_area_subfolders(area_path)


def main():
    # 1) 일반 폴더: 0807 ~ 1109
    normal_dirs = collect_date_range_folders(BASE_PATH, "0807", "1109")
    print(f"일반 폴더 대상: {[os.path.basename(p) for p in normal_dirs]}")

    # 2) OBB 폴더: 0616, 0721 만
    obb_dates = ["0616", "0721"]
    obb_dirs = []
    for d in obb_dates:
        path = os.path.join(OBB_BASE_PATH, d)
        if os.path.isdir(path):
            obb_dirs.append(os.path.abspath(path))
        else:
            print(f"[경고] OBB 날짜 폴더가 없음: {path}")

    print(f"OBB 폴더 대상: {[os.path.basename(p) for p in obb_dirs]}")

    all_targets = normal_dirs + obb_dirs
    if not all_targets:
        print("대상 폴더가 없습니다.")
        return

    for target_dir in all_targets:
        print(f"\n=== 날짜 폴더: {target_dir} ===")
        process_frontdoor_bad(target_dir)

    print("\n모든 작업 완료.")


if __name__ == "__main__":
    main()


