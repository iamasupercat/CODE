#!/usr/bin/env python3
"""
DINO 훈련용 split을 생성하는 스크립트 (Bolt 전용)
ResNet txt 형식과 동일하게 경로와 라벨을 저장합니다.

- Bolt 모드의 라벨 할당: quality 폴더와 subdir를 기준으로 2클래스 또는 4클래스 라벨 결정
- stratified_split 로직을 label 기준으로 강화하여 클래스 불균형 및 다중 클래스에 대응

Bolt 폴더 구조:
    target_dir/
    ├── subfolder/
    │   ├── bad/
    │   │   ├── crop_bolt/        
    │   │   │  ├──0/       # 정측면
    │   │   │  └──1/       # 측면
    │   │   └── crop_bolt_aug/        
    │   └── good/
    │       ├── crop_bolt/        
    │       │  ├──0/       # 정측면
    │       │  └──1/       # 측면
    │       └── crop_bolt_aug/        

사용법:
    # Bolt 모드 (기본 2클래스):
    python DINOsplit_bolt.py \
        --date-range 0807 1013 \
        --subfolders frontfender hood trunklid \
        --name Bolt

    # Bolt 모드 (4클래스):
    python DINOsplit_bolt.py \
        --bolt-4class \
        --bad-date-range 0807 1013 \
        --good-date-range 0616 1103 \
        --subfolders frontfender hood trunklid \
        --name Bolt_4class

    # 결과: 
    # TXT/train_dino_{name}.txt, TXT/val_dino_{name}.txt, TXT/test_dino_{name}.txt
    # 형식: 경로 라벨 (예: /path/to/image.jpg 0)
"""

import os
import random
import argparse
import glob
from pathlib import Path
from collections import defaultdict
import re

random.seed(42)

SPLIT_RATIO = [0.7, 0.1, 0.2]  # train, val, test (7:1:2)

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

def extract_image_id_bolt(img_name: str) -> str:
    """Bolt 모드: 이미지명에서 UUID까지 포함한 고유 ID 추출"""
    aug_suffixes = ['_invert', '_blur', '_bright', '_contrast', '_flip', '_gray', '_noise', '_rot']
    img_name_clean = img_name
    
    for suffix in aug_suffixes:
        # YOLOsplit과 동일하게, 예: xxx_flip.jpg, xxx_noise.png 에서 "_flip", "_noise" 부분만 제거
        for ext in ('.jpg', '.png'):
            full_suffix = suffix + ext  # "_flip.jpg"
            if img_name_clean.endswith(full_suffix):
                # 뒤에서 full_suffix 전체를 제거하고, 원래 확장자를 다시 붙임
                img_name_clean = img_name_clean[:-len(full_suffix)] + ext
                break
    
    parts = img_name_clean.split('_')
    for i, part in enumerate(parts):
        if len(part) == 8 and i + 4 < len(parts):
            if (len(parts[i+1]) == 4 and len(parts[i+2]) == 4 and 
                len(parts[i+3]) == 4 and len(parts[i+4]) == 12):
                return '_'.join(parts[:i+5])
    return os.path.splitext(img_name_clean)[0]

def collect_bolt_images(base_folders, subfolder_names, use_4class=False, quality_filter=None):
    """Bolt 모드: 크롭된 이미지들을 수집하는 함수
    
    Args:
        base_folders: 날짜 폴더들의 절대경로 리스트
        subfolder_names: frontfender/hood/trunklid 등 하위 폴더 이름 리스트
        use_4class: True면 4클래스 모드, False면 2클래스 모드
        quality_filter: None이면 bad/good 모두, 'bad' 또는 'good'이면 해당 quality만 수집
    """
    all_images = []
    
    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"기본 폴더가 존재하지 않습니다: {base_folder}")
            continue
            
        print(f"\n=== {os.path.basename(base_folder)}에서 Bolt 이미지 수집 ===")
        
        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(base_folder, subfolder_name)
            
            if not os.path.isdir(subfolder_path):
                print(f"  하위폴더가 존재하지 않습니다: {subfolder_name}")
                continue
                
            print(f"  하위폴더: {subfolder_name}")
            
            qualities = ['bad', 'good'] if quality_filter is None else [quality_filter]
            for quality in qualities:
                quality_path = os.path.join(subfolder_path, quality)
                
                if not os.path.isdir(quality_path):
                    print(f"    {quality} 폴더가 존재하지 않습니다")
                    continue
                
                crop_folders = ['crop_bolt', 'crop_bolt_aug']
                
                for crop_folder in crop_folders:
                    crop_path = os.path.join(quality_path, crop_folder)
                    
                    if not os.path.isdir(crop_path):
                        print(f"    {quality}/{crop_folder} 폴더가 존재하지 않습니다")
                        continue
                    
                    for subdir in ['0', '1']:  # 0: 정측면, 1: 측면
                        subdir_path = os.path.join(crop_path, subdir)
                        
                        if not os.path.isdir(subdir_path):
                            print(f"    {quality}/{crop_folder}/{subdir} 폴더가 존재하지 않습니다")
                            continue
                        
                        img_files = glob.glob(os.path.join(subdir_path, '*'))
                        img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in IMG_EXTS]
                        
                        if not img_files:
                            print(f"    {quality}/{crop_folder}/{subdir} 폴더에 이미지 파일이 없습니다")
                            continue
                        
                        print(f"    {quality}/{crop_folder}/{subdir}: {len(img_files)}개 이미지 발견")
                        
                        for img_file in img_files:
                            img_name = os.path.basename(img_file)
                            folder_name = os.path.basename(base_folder.rstrip(os.sep))
                            absolute_path = os.path.abspath(img_file)
                            
                            # --- *** 라벨 결정 로직 (Bolt 요구사항 반영) *** ---
                            if use_4class:
                                # 0: 양품 정측면, 1: 불량 정측면, 2: 양품 측면, 3: 불량 측면
                                if quality == 'good' and subdir == '0':
                                    label = 0  # 양품 정측면
                                elif quality == 'bad' and subdir == '0':
                                    label = 1  # 불량 정측면
                                elif quality == 'good' and subdir == '1':
                                    label = 2  # 양품 측면
                                else:  # quality == 'bad' and subdir == '1'
                                    label = 3  # 불량 측면
                            else:
                                # 2클래스: 0(good), 1(bad)
                                label = 1 if quality == 'bad' else 0
                            # -----------------------------------------------------------
                            
                            # 원본 이미지 ID 추출 로직
                            uuid_pattern = r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
                            match = re.search(uuid_pattern, img_name)
                            
                            if match:
                                uuid_end = match.end()
                                img_name_no_suffix = img_name[:uuid_end] + ('.jpg' if img_name.endswith('.jpg') else '.png')
                                original_image_id = extract_image_id_bolt(img_name_no_suffix)
                            else:
                                original_image_id = extract_image_id_bolt(img_name)
                            
                            img_info = {
                                'path': absolute_path,
                                'subfolder': subfolder_name,
                                'quality': quality,
                                'crop_folder': f"{crop_folder}/{subdir}",
                                'label': label,  # 최종 라벨
                                'is_augmented': '_aug' in crop_folder,
                                'original_image_id': original_image_id,
                                'base_folder': folder_name,
                                'image_id': extract_image_id_bolt(img_name)
                            }
                            all_images.append(img_info)
    
    return all_images

def stratified_split(images, ratios):
    """
    이미지를 stratified split하는 함수 (클래스 비율 유지)
    
    각 label별로 이미지를 비율에 맞게 분할하되,
    같은 original_image_id를 가진 이미지들은 모두 같은 split에 들어가도록 합니다.
    이렇게 하면 데이터 누수를 방지하면서도 클래스 비율을 유지할 수 있습니다.
    """
    # 1. 원본 이미지 ID (키) 추출 및 라벨 부여
    id_to_label = {}
    id_groups = defaultdict(list)
    
    for img in images:
        # 키는 (base_folder, subfolder, quality, original_image_id)를 사용
        id_key = (img['base_folder'], img['subfolder'], img['quality'], img['original_image_id'])
        
        # 해당 ID 그룹의 라벨을 결정
        id_to_label[id_key] = img['label']
        id_groups[id_key].append(img)

    # 2. 각 label별로 ID 그룹을 분류
    label_id_groups = defaultdict(list)
    for id_key, label in id_to_label.items():
        label_id_groups[label].append(id_key)
    
    train_id_set = set()
    val_id_set = set()
    test_id_set = set()
    
    # 3. 각 label별 ID 그룹 리스트를 비율에 맞게 분할
    for label, id_list in label_id_groups.items():
        random.shuffle(id_list)
        
        n_total = len(id_list)
        n_train = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])
        n_test = n_total - n_train - n_val
        
        train_id_set.update(id_list[:n_train])
        val_id_set.update(id_list[n_train:n_train+n_val])
        test_id_set.update(id_list[n_train+n_val:])
        
    # 4. 할당된 ID를 기반으로 실제 이미지 분배
    train_images = []
    val_images = []
    test_images = []
    
    for id_key, imgs in id_groups.items():
        if id_key in train_id_set:
            train_images.extend(imgs)
        elif id_key in val_id_set:
            val_images.extend(imgs)
        elif id_key in test_id_set:
            test_images.extend(imgs)
    
    random.shuffle(train_images)
    random.shuffle(val_images)
    random.shuffle(test_images)
    
    return train_images, val_images, test_images

def write_dino_split_files(splits, name=''):
    """분할된 이미지들을 DINO용 파일에 저장하는 함수"""
    txt_dir = Path('TXT')
    txt_dir.mkdir(parents=True, exist_ok=True)
    
    if name:
        train_file = txt_dir / f'train_dino_{name}.txt'
        val_file = txt_dir / f'val_dino_{name}.txt'
        test_file = txt_dir / f'test_dino_{name}.txt'
    else:
        train_file = txt_dir / 'train_dino.txt'
        val_file = txt_dir / 'val_dino.txt'
        test_file = txt_dir / 'test_dino.txt'
    
    train_images, val_images, test_images = splits
    missing_files = []
    
    def write_paths_with_label(file_path, imgs):
        written = 0
        with open(file_path, 'w') as f:
            for img in imgs:
                p = img['path']
                if os.path.isfile(p):
                    f.write(f"{p} {img['label']}\n")
                    written += 1
                else:
                    missing_files.append(p)
        return written
    
    train_written = write_paths_with_label(train_file, train_images)
    val_written = write_paths_with_label(val_file, val_images)
    test_written = write_paths_with_label(test_file, test_images)
    
    if missing_files:
        miss_file = txt_dir / f'missing_dino_{name if name else "default"}.txt'
        with open(miss_file, 'w') as mf:
            for p in missing_files:
                mf.write(p + '\n')
    
    print(f"\n=== DINO 분할 결과 ===")
    print(f"  train: {len(train_images)}개 (실제 기록: {train_written}) -> {train_file}")
    print(f"  val: {len(val_images)}개 (실제 기록: {val_written}) -> {val_file}")
    print(f"  test: {len(test_images)}개 (실제 기록: {test_written}) -> {test_file}")

def main():
    parser = argparse.ArgumentParser(description='DINO 훈련용 split을 생성합니다 (Bolt 전용).')
    parser.add_argument('--folders', nargs='+', 
                       help='분석할 기본 폴더 날짜들 (예: 0616 0718 0721, 일반 폴더)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                       help='일반 폴더 날짜 구간 선택 (MMDD). 예: --date-range 0807 1103')
    parser.add_argument('--obb-folders', nargs='+',
                       help='분석할 OBB 폴더 날짜들 (예: 0718 0806)')
    parser.add_argument('--obb-date-range', nargs=2, metavar=('START', 'END'),
                       help='OBB 폴더 날짜 구간 선택 (MMDD). 예: --obb-date-range 0718 0806')
    parser.add_argument('--subfolders', nargs='+', required=True,
                       help='찾을 하위폴더들 (여러 개 가능, 일반/OBB 공통)')
    parser.add_argument('--name', type=str, default='',
                       help='출력 파일명에 사용할 이름')
    
    # Bolt 모드 옵션
    parser.add_argument('--bolt-4class', action='store_true',
                       help='Bolt 모드: 4클래스 사용 (정측면 양품/불량, 측면 양품/불량)')
    parser.add_argument('--bad-date-range', nargs=2, metavar=('START', 'END'),
                       help='Bolt 모드: bad용 일반 폴더 날짜 구간 (MMDD). 예: --bad-date-range 0616 0806')
    parser.add_argument('--good-date-range', nargs=2, metavar=('START', 'END'),
                       help='Bolt 모드: good용 일반 폴더 날짜 구간 (MMDD). 예: --good-date-range 0807 1103')
    
    args = parser.parse_args()
    
    base_path = "/home/work/datasets"
    
    # 일반 폴더
    if args.date_range:
        start, end = args.date_range
        target_folders = collect_date_range_folders(base_path, start, end)
        print(f"일반 폴더 날짜 구간: {start} ~ {end}")
    elif args.folders:
        target_folders = [os.path.join(base_path, date) for date in args.folders]
    else:
        target_folders = []
    
    # OBB 폴더
    obb_base_path = os.path.join(base_path, "OBB")
    obb_folders = []
    if args.obb_date_range:
        start, end = args.obb_date_range
        obb_folders = collect_date_range_folders(obb_base_path, start, end)
        print(f"OBB 폴더 날짜 구간: {start} ~ {end}")
    elif args.obb_folders:
        obb_folders = [os.path.join(obb_base_path, date) for date in args.obb_folders]
    
    # 최종 대상 폴더 (일반 + OBB)
    all_folders = target_folders + obb_folders
    if not all_folders and not (args.bad_date_range or args.good_date_range):
        parser.error("--folders/--date-range 또는 --obb-folders/--obb-date-range 또는 --bad-date-range/--good-date-range 중 하나는 반드시 지정해야 합니다.")
    
    display_dates = [os.path.basename(p) for p in target_folders]
    obb_display_dates = [os.path.basename(p) for p in obb_folders]
    print(f"모드: Bolt")
    if display_dates:
        print(f"일반 폴더들: {display_dates}")
    if obb_display_dates:
        print(f"OBB 폴더들: {obb_display_dates}")
    print(f"찾을 하위폴더들: {args.subfolders}")
    if args.name:
        print(f"출력 파일 이름: {args.name}")
    
    # Bolt 모드 처리
    if args.bolt_4class:
        print("Bolt 모드: 4클래스 사용 (정측면 양품/불량, 측면 양품/불량)")
    else:
        print("Bolt 모드: 2클래스 사용 (good/bad)")
    
    # bad/good 각각 별도 날짜 범위가 지정된 경우
    if args.bad_date_range or args.good_date_range:
        # 기본값은 전체 target_folders 사용
        bad_base_folders = target_folders
        good_base_folders = target_folders
        # bad 전용 범위
        if args.bad_date_range:
            b_start, b_end = args.bad_date_range
            bad_base_folders = collect_date_range_folders(base_path, b_start, b_end)
            print(f"  bad용 일반 폴더 날짜 구간: {b_start} ~ {b_end}")
        # good 전용 범위
        if args.good_date_range:
            g_start, g_end = args.good_date_range
            good_base_folders = collect_date_range_folders(base_path, g_start, g_end)
            print(f"  good용 일반 폴더 날짜 구간: {g_start} ~ {g_end}")
        # OBB 폴더는 bad/good 공통 사용
        bad_folders = bad_base_folders + obb_folders
        good_folders = good_base_folders + obb_folders
        dino_images = []
        # bad만 수집
        if bad_folders:
            dino_images.extend(collect_bolt_images(bad_folders, args.subfolders,
                                                   use_4class=args.bolt_4class,
                                                   quality_filter='bad'))
        # good만 수집
        if good_folders:
            dino_images.extend(collect_bolt_images(good_folders, args.subfolders,
                                                   use_4class=args.bolt_4class,
                                                   quality_filter='good'))
    else:
        # 기존 동작: bad/good 동일 폴더 범위
        dino_images = collect_bolt_images(all_folders, args.subfolders, args.bolt_4class)
    
    if not dino_images:
        print("수집된 DINO용 이미지가 없습니다.")
        return
    
    print(f"\n총 {len(dino_images)}개 DINO용 이미지 수집 완료")
    
    # quality별 수집 개수 출력
    quality_counts = defaultdict(int)
    for img in dino_images:
        quality_counts[img['quality']] += 1
    print(f"\n=== 수집된 이미지 quality별 개수 ===")
    for quality in sorted(quality_counts.keys()):
        print(f"  {quality}: {quality_counts[quality]}개")
    
    # 원본과 증강 분리
    original_images = [img for img in dino_images if not img['is_augmented']]
    aug_images = [img for img in dino_images if img['is_augmented']]
    
    print(f"  - 원본 이미지: {len(original_images)}개")
    print(f"  - 증강 이미지: {len(aug_images)}개")
    
    # label별 원본 이미지 개수 출력
    label_counts = defaultdict(int)
    quality_label_counts = defaultdict(lambda: defaultdict(int))
    for img in original_images:
        label_counts[img['label']] += 1
        quality_label_counts[img['quality']][img['label']] += 1
    print(f"\n=== 원본 이미지 label별 개수 ===")
    for label in sorted(label_counts.keys()):
        print(f"  Label {label}: {label_counts[label]}개")
    print(f"\n=== 원본 이미지 quality별 label별 개수 ===")
    for quality in sorted(quality_label_counts.keys()):
        print(f"  {quality}:")
        for label in sorted(quality_label_counts[quality].keys()):
            print(f"    Label {label}: {quality_label_counts[quality][label]}개")
    
    # 원본 이미지로 split 수행
    train_original, val_original, test_original = stratified_split(original_images, SPLIT_RATIO)
    
    # split 결과 label별 개수 출력
    def count_labels(imgs):
        counts = defaultdict(int)
        for img in imgs:
            counts[img['label']] += 1
        return counts
    
    train_label_counts = count_labels(train_original)
    val_label_counts = count_labels(val_original)
    test_label_counts = count_labels(test_original)
    
    print(f"\n=== Split 결과 label별 개수 ===")
    print(f"Train: {dict(sorted(train_label_counts.items()))}")
    print(f"Val: {dict(sorted(val_label_counts.items()))}")
    print(f"Test: {dict(sorted(test_label_counts.items()))}")
    
    # train에 선택된 원본 이미지의 키 추출
    train_original_keys = set()
    for img in train_original:
        key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['original_image_id']}"
        train_original_keys.add(key)
    
    # train에 속한 원본에서 생성된 증강 이미지들을 train에 추가
    train_aug = []
    for aug_img in aug_images:
        key = f"{aug_img['base_folder']}/{aug_img['subfolder']}/{aug_img['quality']}/{aug_img['original_image_id']}"
        if key in train_original_keys:
            train_aug.append(aug_img)
    
    print(f"\n=== 증강 이미지 매칭 ===")
    print(f"train에 추가된 증강 이미지: {len(train_aug)}개")
    
    # 최종 split 구성
    train_final = train_original + train_aug
    val_final = val_original
    test_final = test_original
    
    random.shuffle(train_final)
    
    # 파일에 저장
    write_dino_split_files((train_final, val_final, test_final), args.name)

if __name__ == '__main__':
    main()

