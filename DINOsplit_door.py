#!/usr/bin/env python3
"""
DINO 훈련용 split을 생성하는 스크립트 (Door 전용 최종 수정)

- Door / Door Area 모드의 이미지 수집 로직 안정화 (glob 기반)
- 라벨 결정: 하위 폴더명(subdir)을 그대로 클래스 번호로 사용 (good: 0, bad: 1~4)
- stratified_split을 통해 최종 라벨 비율 유지

python DINOsplit_door.py \
      --mode door_area \
      --areas high mid low \
      --date-range 0807 1109 \
      --obb-date-range 0616 0806 \
      --subfolders frontdoor \
      --name Door
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

"""
예시:
python DINOsplit_door.py \
    --mode door_area \
    --areas high mid low \
    --date-range 0807 1109 \
    --obb-date-range 0616 0806 \
    --subfolders frontdoor \
    --name Door_5class
"""

# --- 도우미 함수 ---
# (코드의 일관성을 위해 필요한 최소한의 도우미 함수를 포함합니다.)

def collect_date_range_folders(base_path: str, start: str, end: str):
    """지정된 날짜 범위의 폴더 경로를 수집합니다."""
    # (내부 로직은 유지)
    if not (start.isdigit() and end.isdigit()): return []
    if len(start) != len(end) or len(start) not in (4, 8): return []
    s_val, e_val = int(start), int(end)
    if s_val > e_val: s_val, e_val = e_val, s_val
    found = []
    try:
        for name in os.listdir(base_path):
            full = os.path.join(base_path, name)
            if os.path.isdir(full) and name.isdigit() and len(name) == len(start):
                val = int(name)
                if s_val <= val <= e_val: found.append(os.path.abspath(full))
    except FileNotFoundError: return []
    found.sort(key=lambda p: int(os.path.basename(p)))
    return found

def extract_image_id_door(img_name: str) -> str:
    """Door 모드: 이미지명에서 원본 이미지 파일명 추출"""
    img_name_without_ext = os.path.splitext(img_name)[0]
    crop_aug_types = ['bright', 'contrast', 'flip', 'gray', 'noise', 'rot']
    
    # 크롭 증강 이미지: 원본파일명_라벨링번호_증강기법
    if '_' in img_name_without_ext:
        parts = img_name_without_ext.split('_')
        if len(parts) >= 3 and parts[-1] in crop_aug_types and parts[-2].isdigit():
            return '_'.join(parts[:-2])
    
    # 크롭 이미지: 원본파일명_라벨링번호 (0, 1, 2)
    if '_' in img_name_without_ext:
        parts = img_name_without_ext.split('_')
        if len(parts) >= 2 and parts[-1] in ['0', '1', '2']:
            return '_'.join(parts[:-1])
    
    return img_name_without_ext

def collect_original_folders(base_folders, subfolder_names):
    """원본 폴더명들을 수집하는 함수 (Door 모드용)"""
    original_folders = set()
    for base_folder in base_folders:
        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(base_folder, subfolder_name)
            if not os.path.isdir(subfolder_path): continue
            for quality in ['bad', 'good']:
                quality_path = os.path.join(subfolder_path, quality)
                if not os.path.isdir(quality_path): continue
                images_path = os.path.join(quality_path, 'images')
                if not os.path.isdir(images_path): continue
                img_files = glob.glob(os.path.join(images_path, '*'))
                for img_file in img_files:
                    if os.path.splitext(img_file)[1].lower() in IMG_EXTS:
                        original_folders.add(os.path.splitext(os.path.basename(img_file))[0])
    return original_folders


def collect_door_images(base_folders, subfolder_names, original_folders, target_areas=None):
    """
    Door 모드: 크롭된 이미지들을 수집하는 함수 (Glob 기반, 라벨링 로직 수정)
    - 최종 라벨은 subdir(0~3)을 기준으로 결정
    - merge_classes가 True면 최종 라벨은 무조건 0 또는 1이 되도록 보장
    """
    all_images = []
    
    # 사용할 영역 설정 로직 유지
    if target_areas is None:
        crop_areas = ['crop_high', 'crop_mid', 'crop_low']
    else:
        area_map = {'high': 'crop_high', 'mid': 'crop_mid', 'low': 'crop_low'}
        crop_areas = [area_map[a] for a in target_areas if a in area_map]
        if not crop_areas: return all_images
    
    for base_folder in base_folders:
        if not os.path.isdir(base_folder): continue
        print(f"\n=== {os.path.basename(base_folder)}에서 Door 이미지 수집 ===")
        
        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(base_folder, subfolder_name)
            if not os.path.isdir(subfolder_path): continue
            print(f"  하위폴더: {subfolder_name}")
            
            for quality in ['bad', 'good']:
                quality_path = os.path.join(subfolder_path, quality)
                if not os.path.isdir(quality_path): continue
                
                crop_folders = []
                for area in crop_areas:
                    crop_folders.extend([area, area + '_aug'])
                
                for crop_folder in crop_folders:
                    crop_path = os.path.join(quality_path, crop_folder)
                    if not os.path.isdir(crop_path): continue
                    
                    # good: 주로 0, bad: 1~4 사용 (0~4 전체를 탐색)
                    for subdir in ['0', '1', '2', '3', '4']:
                        subdir_path = os.path.join(crop_path, subdir)
                        if not os.path.isdir(subdir_path): continue
                        
                        img_files = glob.glob(os.path.join(subdir_path, '*'))
                        img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in IMG_EXTS]
                        
                        crop_count = len(img_files)
                        if crop_count == 0: continue
                        
                        print(f"    {quality}/{crop_folder}/{subdir}: {crop_count}개 이미지 발견")

                        # --- *** 라벨 결정 로직 (5-class 고정) *** ---
                        # 하위 폴더명을 그대로 클래스 번호로 사용
                        #   good: 0(양품)
                        #   bad: 1(출고실링), 2(실링없음), 3(작업실링), 4(테이프실링)
                        label = int(subdir)
                        # ---------------------------------------------
                        
                        for img_file in img_files:
                            img_name = os.path.basename(img_file)
                            original_name_without_ext = extract_image_id_door(img_name) 
                            folder_name = os.path.basename(base_folder.rstrip(os.sep))
                            absolute_path = os.path.abspath(img_file)
                            
                            img_info = {
                                'path': absolute_path,
                                'subfolder': subfolder_name,
                                'quality': quality, 
                                'crop_folder': f"{crop_folder}/{subdir}",
                                'label': label, # 최종 라벨 (0 또는 1, 또는 0~3)
                                'is_augmented': '_aug' in crop_folder,
                                'original_image_id': original_name_without_ext,
                                'base_folder': folder_name,
                                'image_id': extract_image_id_door(img_name)
                            }
                            all_images.append(img_info)
    
    return all_images

# (중략: stratified_split, write_dino_split_files 등 나머지 함수는 이전 코드와 동일하게 유지)

def stratified_split(images, ratios):
    """이미지를 stratified split하는 함수 (클래스 비율 유지)"""
    id_to_label = {}
    id_groups = defaultdict(list)
    for img in images:
        id_key = (img['base_folder'], img['subfolder'], img['quality'], img['original_image_id'])
        id_to_label[id_key] = img['label']
        id_groups[id_key].append(img)
    label_id_groups = defaultdict(list)
    for id_key, label in id_to_label.items():
        label_id_groups[label].append(id_key)
    train_id_set, val_id_set, test_id_set = set(), set(), set()
    for label, id_list in label_id_groups.items():
        random.shuffle(id_list)
        n_total = len(id_list)
        n_train = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])
        n_test = n_total - n_train - n_val
        train_id_set.update(id_list[:n_train])
        val_id_set.update(id_list[n_train:n_train+n_val])
        test_id_set.update(id_list[n_train+n_val:])
    train_images, val_images, test_images = [], [], []
    for id_key, imgs in id_groups.items():
        if id_key in train_id_set: train_images.extend(imgs)
        elif id_key in val_id_set: val_images.extend(imgs)
        elif id_key in test_id_set: test_images.extend(imgs)
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
                else: missing_files.append(p)
        return written
    train_written = write_paths_with_label(train_file, train_images)
    val_written = write_paths_with_label(val_file, val_images)
    test_written = write_paths_with_label(test_file, test_images)
    if missing_files:
        miss_file = txt_dir / f'missing_dino_{name if name else "default"}.txt'
        with open(miss_file, 'w') as mf:
            for p in missing_files: mf.write(p + '\n')
    print(f"\n=== DINO 분할 결과 ===")
    print(f"  train: {len(train_images)}개 (실제 기록: {train_written}) -> {train_file}")
    print(f"  val: {len(val_images)}개 (실제 기록: {val_written}) -> {val_file}")
    print(f"  test: {len(test_images)}개 (실제 기록: {test_written}) -> {test_file}")

def main():
    parser = argparse.ArgumentParser(description='DINO 훈련용 split을 생성합니다.')
    parser.add_argument('--mode', choices=['door', 'door_area'], required=True,
                       help='모드 선택: door 또는 door_area')
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
    
    # Door Area 모드 옵션
    parser.add_argument('--areas', nargs='+', choices=['high', 'mid', 'low'],
                       help='Door Area 모드: 처리할 영역 선택 (기본값: high mid low 모두)')
    
    args = parser.parse_args()
    
    base_path = "/home/work/datasets"
    
    # 폴더 수집
    if args.date_range:
        start, end = args.date_range
        target_folders = collect_date_range_folders(base_path, start, end)
    elif args.folders:
        target_folders = [os.path.join(base_path, date) for date in args.folders]
    else:
        target_folders = []
    
    obb_base_path = os.path.join(base_path, "OBB")
    obb_folders = []
    if args.obb_date_range:
        start, end = args.obb_date_range
        obb_folders = collect_date_range_folders(obb_base_path, start, end)
    elif args.obb_folders:
        obb_folders = [os.path.join(obb_base_path, date) for date in args.obb_folders]
    
    all_folders = target_folders + obb_folders
    if not all_folders:
        parser.error("--folders/--date-range 또는 --obb-folders/--obb-date-range 중 하나는 반드시 지정해야 합니다.")
    
    # ... (생략: display_dates 출력 로직) ...
    
    # 모드별 처리
    if args.mode == 'door':
        print("Door 모드: 0~4 (양품/출고실링/실링없음/작업실링/테이프실링) 5-class 그대로 사용")
        original_folders = collect_original_folders(all_folders, args.subfolders)
        if not original_folders:
            print("수집된 원본 폴더명이 없습니다.")
            return

        # Door 모드는 모든 영역을 한 파일에 수집 (5-class 고정)
        dino_images = collect_door_images(
            all_folders,
            args.subfolders,
            original_folders,
            target_areas=['high', 'mid', 'low'],
        )

        if not dino_images:
            print("수집된 DINO용 이미지가 없습니다.")
            return

        # 후처리 및 저장
        original_images = [img for img in dino_images if not img['is_augmented']]
        aug_images = [img for img in dino_images if img['is_augmented']]
    
        train_original, val_original, test_original = stratified_split(original_images, SPLIT_RATIO)
        
        train_original_keys = set()
        for img in train_original:
            key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['original_image_id']}"
            train_original_keys.add(key)
        
        train_aug = []
        for aug_img in aug_images:
            key = f"{aug_img['base_folder']}/{aug_img['subfolder']}/{aug_img['quality']}/{aug_img['original_image_id']}"
            if key in train_original_keys:
                train_aug.append(aug_img)
        
        train_final = train_original + train_aug
        val_final = val_original
        test_final = test_original
        
        random.shuffle(train_final)
        
        write_dino_split_files((train_final, val_final, test_final), args.name)
        
    elif args.mode == 'door_area':
        # Door Area 모드는 영역별로 처리 및 파일 저장 (각 영역도 5-class 고정)
        if args.areas:
            areas = args.areas
        else:
            areas = ['high', 'mid', 'low']
        
        original_folders = collect_original_folders(all_folders, args.subfolders)
        if not original_folders:
            print("수집된 원본 폴더명이 없습니다.")
            return

        for area in areas:
            print(f"\n=== {area} 영역 처리 (5-class 그대로 사용: 0~4) ===")
            
            dino_images_area = collect_door_images(
                all_folders,
                args.subfolders,
                original_folders,
                target_areas=[area],
            )
            
            if not dino_images_area:
                print(f"{area} 영역에 수집된 DINO용 이미지가 없습니다.")
                continue
            
            original_images = [img for img in dino_images_area if not img['is_augmented']]
            aug_images = [img for img in dino_images_area if img['is_augmented']]
            
            train_original, val_original, test_original = stratified_split(original_images, SPLIT_RATIO)
            
            train_original_keys = set()
            for img in train_original:
                key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['original_image_id']}"
                train_original_keys.add(key)
            
            train_aug = []
            for aug_img in aug_images:
                key = f"{aug_img['base_folder']}/{aug_img['subfolder']}/{aug_img['quality']}/{aug_img['original_image_id']}"
                if key in train_original_keys:
                    train_aug.append(aug_img)
            
            train_final = train_original + train_aug
            val_final = val_original
            test_final = test_original
            
            random.shuffle(train_final)
            
            area_suffix = f"{args.name}_{area}" if args.name else f"door_{area}"
            write_dino_split_files((train_final, val_final, test_final), area_suffix)

        return

if __name__ == '__main__':
    main()