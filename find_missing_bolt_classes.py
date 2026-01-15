#!/usr/bin/env python3
"""
Bolt 모드에서 txt 파일 내에 클래스 0 또는 1이 아예 없는 경우를 찾는 스크립트

사용법:
    # 날짜 범위 지정
    python find_missing_bolt_classes.py \
        --date-range 0807 1109 \
        --subfolders frontfender hood trunklid \
        --output missing_bolt_classes.txt

    # OBB 폴더 포함
    python find_missing_bolt_classes.py \
        --date-range 0807 1109 \
        --obb-date-range 0616 0806 \
        --subfolders frontfender hood trunklid \
        --output missing_bolt_classes.txt

    # 특정 폴더 지정
    python find_missing_bolt_classes.py \
        --folders 0910 0920 \
        --subfolders frontfender \
        --output missing_bolt_classes.txt
"""

import os
import glob
import argparse
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


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


def parse_label_line(line):
    """
    라벨 라인 파싱: 클래스 ID 추출
    지원 형식:
    - BB: class x y w h (5개 값)
    - OBB: class x y w h angle (6개 값)
    - 변환된 OBB: class x1 y1 x2 y2 x3 y3 x4 y4 (9개 값)
    """
    line = line.strip()
    if not line:
        return None
    
    parts = line.split()
    if len(parts) < 1:
        return None
    
    try:
        cls = int(float(parts[0]))
        return cls
    except (ValueError, IndexError):
        return None


def check_bolt_classes(label_path):
    """
    라벨 파일에서 클래스 0 또는 1이 있는지 확인
    
    Returns:
        tuple: (has_class_0, has_class_1, all_classes) 또는 None (파일 읽기 실패)
    """
    # .bak 파일이 있으면 우선 사용
    bak_path = label_path + '.bak'
    actual_label_path = bak_path if os.path.exists(bak_path) else label_path
    
    if not os.path.exists(actual_label_path):
        return None
    
    try:
        has_class_0 = False
        has_class_1 = False
        all_classes = set()
        
        with open(actual_label_path, 'r', encoding='utf-8') as f:
            for line in f:
                cls = parse_label_line(line)
                if cls is not None:
                    all_classes.add(cls)
                    if cls == 0:
                        has_class_0 = True
                    elif cls == 1:
                        has_class_1 = True
        
        return (has_class_0, has_class_1, all_classes)
    except Exception as e:
        print(f"⚠️  파일 읽기 실패: {actual_label_path} - {e}")
        return None


def find_missing_bolt_classes(base_folders, subfolder_names, is_obb=False):
    """
    클래스 0 또는 1이 없는 라벨 파일들을 찾는 함수
    
    Returns:
        list: (이미지 경로, 라벨 경로, 발견된 클래스들) 튜플 리스트
    """
    missing_files = []
    
    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"기본 폴더가 존재하지 않습니다: {base_folder}")
            continue
        
        folder_type = "OBB" if is_obb else "일반"
        print(f"\n=== [{folder_type}] {base_folder}에서 클래스 0/1 누락 검색 ===")
        
        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(base_folder, subfolder_name)
            
            if not os.path.isdir(subfolder_path):
                print(f"  하위폴더가 존재하지 않습니다: {subfolder_name}")
                continue
            
            print(f"  하위폴더: {subfolder_name}")
            
            for quality in ['bad', 'good']:
                quality_path = os.path.join(subfolder_path, quality)
                if not os.path.isdir(quality_path):
                    continue
                
                images_path = os.path.join(quality_path, 'images')
                labels_path = os.path.join(quality_path, 'labels')
                
                if not os.path.isdir(images_path):
                    continue
                
                if not os.path.isdir(labels_path):
                    print(f"    ⚠️  {quality}/labels 폴더가 존재하지 않습니다.")
                    continue
                
                # 이미지 파일들 검색
                img_files = glob.glob(os.path.join(images_path, '*'))
                img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in IMG_EXTS]
                
                missing_count = 0
                for img_file in img_files:
                    img_name = os.path.basename(img_file)
                    label_name = os.path.splitext(img_name)[0] + '.txt'
                    label_path = os.path.join(labels_path, label_name)
                    
                    result = check_bolt_classes(label_path)
                    if result is None:
                        # 파일이 없거나 읽기 실패
                        continue
                    
                    has_class_0, has_class_1, all_classes = result
                    
                    # 클래스 0과 1이 모두 없는 경우
                    if not has_class_0 and not has_class_1:
                        missing_files.append((
                            os.path.abspath(img_file),
                            label_path,
                            sorted(all_classes) if all_classes else []
                        ))
                        missing_count += 1
                
                if missing_count > 0:
                    print(f"    {quality}: {missing_count}개 이미지의 라벨에 클래스 0/1이 없습니다")
                else:
                    print(f"    {quality}: 모든 이미지의 라벨에 클래스 0 또는 1이 있습니다")
    
    return missing_files


def main():
    parser = argparse.ArgumentParser(description='Bolt 모드에서 클래스 0 또는 1이 없는 라벨 파일들을 찾습니다.')
    parser.add_argument('--folders', nargs='+',
                       help='분석할 기본 폴더 날짜들 (일반 폴더)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                       help='일반 폴더 날짜 구간 선택 (MMDD). 예: --date-range 0807 1103')
    parser.add_argument('--obb-folders', nargs='+',
                       help='분석할 OBB 폴더 날짜들')
    parser.add_argument('--obb-date-range', nargs=2, metavar=('START', 'END'),
                       help='OBB 폴더 날짜 구간 선택 (MMDD). 예: --obb-date-range 0718 0806')
    parser.add_argument('--subfolders', nargs='+', required=True,
                       help='찾을 하위폴더들 (frontfender, hood, trunklid)')
    parser.add_argument('--output', type=str, default='missing_bolt_classes.txt',
                       help='출력 파일 경로 (기본값: missing_bolt_classes.txt)')
    
    args = parser.parse_args()
    
    base_path = "/workspace/datasets"
    
    # 일반 폴더 처리
    if args.date_range:
        start, end = args.date_range
        target_folders = collect_date_range_folders(base_path, start, end)
        print(f"일반 폴더 날짜 구간: {start} ~ {end}")
    elif args.folders:
        target_folders = [os.path.join(base_path, date) for date in args.folders]
    else:
        target_folders = []
    
    # OBB 폴더 처리
    obb_base_path = os.path.join(base_path, "OBB")
    obb_folders = []
    if args.obb_date_range:
        start, end = args.obb_date_range
        obb_folders = collect_date_range_folders(obb_base_path, start, end)
        print(f"OBB 폴더 날짜 구간: {start} ~ {end}")
    elif args.obb_folders:
        obb_folders = [os.path.join(obb_base_path, date) for date in args.obb_folders]
    
    all_folders = target_folders + obb_folders
    
    if target_folders:
        print(f"일반 폴더들: {[os.path.basename(p) for p in target_folders]}")
    if obb_folders:
        print(f"OBB 폴더들: {[os.path.basename(p) for p in obb_folders]}")
    print(f"찾을 하위폴더들: {args.subfolders}")
    
    # 누락된 클래스 검색
    missing_files = []
    
    if target_folders:
        missing_files.extend(find_missing_bolt_classes(target_folders, args.subfolders, is_obb=False))
    
    if obb_folders:
        missing_files.extend(find_missing_bolt_classes(obb_folders, args.subfolders, is_obb=True))
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"=== 검색 결과 ===")
    print(f"{'='*60}")
    print(f"총 {len(missing_files)}개 이미지의 라벨에 클래스 0 또는 1이 없습니다.")
    
    if missing_files:
        # 클래스별 통계
        class_stats = {}
        for _, _, classes in missing_files:
            if not classes:
                class_key = "빈 파일"
            else:
                class_key = ", ".join(map(str, classes))
            class_stats[class_key] = class_stats.get(class_key, 0) + 1
        
        print(f"\n발견된 클래스별 통계:")
        for class_key, count in sorted(class_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  - 클래스 {class_key}: {count}개")
        
        # 파일에 저장
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 클래스 0 또는 1이 없는 라벨 파일 목록\n")
            f.write(f"# 총 {len(missing_files)}개\n")
            f.write("# 형식: 이미지_경로 | 라벨_경로 | 발견된_클래스들\n\n")
            
            for img_path, label_path, classes in missing_files:
                classes_str = ", ".join(map(str, classes)) if classes else "없음"
                f.write(f"{img_path} | {label_path} | 클래스: {classes_str}\n")
        
        print(f"\n결과가 {output_path}에 저장되었습니다.")
        
        # 처음 10개 샘플 출력
        print(f"\n샘플 (처음 10개):")
        for i, (img_path, label_path, classes) in enumerate(missing_files[:10], 1):
            classes_str = ", ".join(map(str, classes)) if classes else "없음"
            print(f"  {i}. {os.path.basename(img_path)}")
            print(f"     라벨: {label_path}")
            print(f"     발견된 클래스: {classes_str}")
    else:
        print("\n✅ 모든 라벨 파일에 클래스 0 또는 1이 있습니다!")


if __name__ == '__main__':
    main()

