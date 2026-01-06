#!/usr/bin/env python3
"""
YOLO 라벨 파일을 BB(5개 값)와 OBB(6개 값) 모드 간 전환하는 스크립트



사용법:
    python bb_to_obb.py --to-bb    # 전체를 BB 형식(5개 값)으로 변환
    python bb_to_obb.py --to-obb   # 전체를 OBB 형식(6개 값)으로 변환





동작:
    - 0718부터 0806까지의 폴더를 자동으로 처리
    - --to-bb: 모든 라인을 BB 형식(5개 값)으로 변환 (6개 값이면 마지막 값 제거)
    - --to-obb: 모든 라인을 OBB 형식(6개 값)으로 변환 (5개 값이면 0 추가)
    BB 형식: class x_center y_center width height
    OBB 형식: class x_center y_center width height angle
"""

import os
import glob
import argparse
from pathlib import Path


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


def process_txt_file(txt_path: str, mode: str):
    """
    txt 파일을 읽어서 BB↔OBB 전환 처리
    
    Args:
        txt_path: 처리할 txt 파일 경로
        mode: 'bb' 또는 'obb'
            - 'bb': 모든 라인을 BB 형식(5개 값)으로 변환
            - 'obb': 모든 라인을 OBB 형식(6개 값)으로 변환
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  오류: {txt_path} 읽기 실패 - {e}")
        return False, 0, 0

    modified_lines = []
    converted_count = 0

    for line in lines:
        line = line.strip()
        if not line:  # 빈 줄은 그대로 유지
            modified_lines.append('')
            continue

        parts = line.split()
        
        if mode == 'bb':
            # BB 모드: 모든 라인을 5개 값으로 변환
            if len(parts) == 6:
                # 6개 값이면 마지막 값 제거
                new_line = ' '.join(parts[:5])
                modified_lines.append(new_line)
                converted_count += 1
            elif len(parts) == 5:
                # 이미 5개 값이면 그대로 유지
                modified_lines.append(line)
            else:
                # 5개도 6개도 아니면 그대로 유지
                modified_lines.append(line)
        
        elif mode == 'obb':
            # OBB 모드: 모든 라인을 6개 값으로 변환
            if len(parts) == 5:
                # 5개 값이면 0 추가
                new_line = ' '.join(parts) + ' 0'
                modified_lines.append(new_line)
                converted_count += 1
            elif len(parts) == 6:
                # 이미 6개 값이면 그대로 유지
                modified_lines.append(line)
            else:
                # 5개도 6개도 아니면 그대로 유지
                modified_lines.append(line)

    # 파일이 변경되었는지 확인
    if converted_count == 0:
        return False, 0, 0

    # 파일 쓰기
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            for line in modified_lines:
                f.write(line + '\n')
        return True, converted_count, 0
    except Exception as e:
        print(f"  오류: {txt_path} 쓰기 실패 - {e}")
        return False, 0, 0


def process_labels_folder(labels_path: str, mode: str):
    """labels 폴더의 모든 txt 파일을 처리"""
    if not os.path.isdir(labels_path):
        return 0, 0, 0

    txt_files = glob.glob(os.path.join(labels_path, '*.txt'))
    # .bak 파일 제외
    txt_files = [f for f in txt_files if not f.endswith('.bak')]
    total_files = len(txt_files)
    total_converted = 0
    modified_files = 0

    for txt_file in txt_files:
        modified, converted, _ = process_txt_file(txt_file, mode)
        if modified:
            modified_files += 1
            total_converted += converted

    return total_files, modified_files, total_converted


def main():
    parser = argparse.ArgumentParser(description='YOLO 라벨 파일을 BB 또는 OBB 형식으로 변환')
    parser.add_argument('--to-bb', action='store_true', help='전체를 BB 형식(5개 값)으로 변환')
    parser.add_argument('--to-obb', action='store_true', help='전체를 OBB 형식(6개 값)으로 변환')
    args = parser.parse_args()

    # 모드 확인
    if args.to_bb and args.to_obb:
        print("오류: --to-bb와 --to-obb를 동시에 사용할 수 없습니다.")
        return
    if not args.to_bb and not args.to_obb:
        print("오류: --to-bb 또는 --to-obb 중 하나를 선택해야 합니다.")
        print("사용법:")
        print("  python toggle_obb_bb.py --to-bb    # 전체를 BB 형식으로 변환")
        print("  python toggle_obb_bb.py --to-obb   # 전체를 OBB 형식으로 변환")
        return

    mode = 'bb' if args.to_bb else 'obb'
    mode_name = 'BB' if mode == 'bb' else 'OBB'

    base_path = "/home/ciw/work/datasets"
    start_date = "0718"
    end_date = "0806"

    print("=" * 60)
    print(f"YOLO 라벨 → {mode_name} 형식 변환 스크립트")
    print("=" * 60)
    print(f"변환 모드: {mode_name} 형식 ({'5개 값' if mode == 'bb' else '6개 값'})")
    print(f"기본 경로: {base_path}")
    print(f"처리 날짜 범위: {start_date} ~ {end_date}")
    print()

    # 날짜 범위 폴더 수집
    target_folders = collect_date_range_folders(base_path, start_date, end_date)
    if not target_folders:
        print("처리할 폴더가 없습니다.")
        return

    print(f"처리할 폴더 ({len(target_folders)}개):")
    for folder in target_folders:
        print(f"  - {os.path.basename(folder)}")
    print()

    # 통계
    total_txt_files = 0
    total_modified_files = 0
    total_converted_lines = 0

    # 각 폴더 처리
    for folder_path in target_folders:
        folder_name = os.path.basename(folder_path)
        print(f"\n=== {folder_name} 처리 중 ===")

        # 폴더 내의 모든 하위 폴더 찾기 (예: frontfender, hood 등)
        subfolders = [d for d in os.listdir(folder_path) 
                     if os.path.isdir(os.path.join(folder_path, d)) 
                     and not d.endswith('_aug')]  # _aug 폴더는 제외

        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            print(f"  하위폴더: {subfolder}")

            # bad/labels, good/labels 처리
            for quality in ['bad', 'good']:
                labels_path = os.path.join(subfolder_path, quality, 'labels')
                if not os.path.isdir(labels_path):
                    continue

                files_count, modified_count, converted = process_labels_folder(labels_path, mode)
                if files_count > 0:
                    total_txt_files += files_count
                    total_modified_files += modified_count
                    total_converted_lines += converted

                    if modified_count > 0:
                        print(f"    {quality}/labels: {files_count}개 파일 중 {modified_count}개 수정")
                        print(f"      → {mode_name} 형식으로 변환: {converted}줄")
                    else:
                        print(f"    {quality}/labels: {files_count}개 파일 (변경 없음)")

    # 최종 통계
    print("\n" + "=" * 60)
    print("처리 완료")
    print("=" * 60)
    print(f"총 처리된 txt 파일: {total_txt_files}개")
    print(f"수정된 파일: {total_modified_files}개")
    print(f"{mode_name} 형식으로 변환된 줄: {total_converted_lines}줄")
    print()


if __name__ == '__main__':
    main()

