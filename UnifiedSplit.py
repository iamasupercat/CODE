#!/usr/bin/env python3
"""
YOLO와 DINO를 위한 통합 학습 데이터셋 split 생성 스크립트
원본 이미지 기준으로 분할하고, 대응되는 크롭 이미지를 같은 split에 포함시킵니다.

사용법:
    # Bolt 모드
    python unified_split.py \
        --mode bolt \
        --bolt-4class \
        --bad-date-range 0807 1013 \
        --good-date-range 0616 1103 \
        --subfolders frontfender hood trunklid \
        --name Bolt_4class

    # Door 모드
    python unified_split.py \
        --mode door \
        --date-range 0807 1109 \
        --obb-date-range 0616 0806 \
        --subfolders frontdoor \
        --name Door

결과:
    YOLO용: TXT/train_{name}.txt, TXT/val_{name}.txt, TXT/test_{name}.txt
    DINO용: TXT/train_dino_{name}.txt, TXT/val_dino_{name}.txt, TXT/test_dino_{name}.txt
"""

import os
import glob
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict

random.seed(42)

SPLIT_RATIO = [0.7, 0.1, 0.2]  # train, val, test
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


def extract_image_id(img_name: str) -> str:
    """이미지명에서 UUID까지 포함한 고유 ID 추출 (bad_XXXX_..._UUID)"""
    aug_suffixes = ['_invert', '_blur', '_bright', '_contrast', '_flip', '_gray', '_noise', '_rot']
    img_name_clean = img_name

    for suffix in aug_suffixes:
        for ext in ('.jpg', '.png'):
            full_suffix = suffix + ext
            if img_name_clean.endswith(full_suffix):
                img_name_clean = img_name_clean[:-len(full_suffix)] + ext
                break

    parts = img_name_clean.split('_')
    for i, part in enumerate(parts):
        if len(part) == 8 and i + 4 < len(parts):
            if (len(parts[i+1]) == 4 and len(parts[i+2]) == 4 and
                len(parts[i+3]) == 4 and len(parts[i+4]) == 12):
                return '_'.join(parts[:i+5])
    return os.path.splitext(img_name_clean)[0]


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


def collect_yolo_images_from_folders(base_folders, subfolder_names, is_obb=False, quality_filter=None):
    """YOLO 훈련용 원본/증강 이미지들을 수집"""
    all_images = []

    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"기본 폴더가 존재하지 않습니다: {base_folder}")
            continue

        folder_type = "OBB" if is_obb else "일반"
        print(f"\n=== [{folder_type}] {base_folder}에서 YOLO용 이미지 수집 ===")

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

                images_path = os.path.join(quality_path, 'images')

                if not os.path.isdir(images_path):
                    print(f"    {quality}/images 폴더가 존재하지 않습니다")
                    continue

                # 원본 이미지
                img_files = glob.glob(os.path.join(images_path, '*'))
                img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in IMG_EXTS]
                print(f"    {quality} 원본 이미지: {len(img_files)}개")

                for img_file in img_files:
                    img_name = os.path.basename(img_file)
                    folder_name = os.path.basename(base_folder.rstrip(os.sep))
                    if is_obb:
                        folder_name = f"OBB_{folder_name}"
                    absolute_path = os.path.abspath(img_file)
                    img_info = {
                        'path': absolute_path,
                        'subfolder': subfolder_name,
                        'quality': quality,
                        'base_folder': folder_name,
                        'img_name': img_name,
                        'image_id': extract_image_id(img_name),
                        'is_augmented': False
                    }
                    all_images.append(img_info)

                # 증강 이미지
                aug_subfolder_path = os.path.join(base_folder, f"{subfolder_name}_aug", quality, 'images')
                if os.path.isdir(aug_subfolder_path):
                    aug_img_files = glob.glob(os.path.join(aug_subfolder_path, '*'))
                    aug_img_files = [f for f in aug_img_files if os.path.splitext(f)[1].lower() in IMG_EXTS]
                    print(f"    {quality} 증강 이미지: {len(aug_img_files)}개")

                    for img_file in aug_img_files:
                        img_name = os.path.basename(img_file)
                        folder_name = os.path.basename(base_folder.rstrip(os.sep))
                        absolute_path = os.path.abspath(img_file)
                        image_id = extract_image_id(img_name)
                        img_info = {
                            'path': absolute_path,
                            'subfolder': subfolder_name,
                            'quality': quality,
                            'base_folder': folder_name,
                            'img_name': img_name,
                            'image_id': image_id,
                            'is_augmented': True
                        }
                        all_images.append(img_info)

    return all_images


def collect_original_folders(base_folders, subfolder_names):
    """원본 폴더명들을 수집하는 함수 (Door 모드용)"""
    original_folders = set()
    
    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            continue
            
        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(base_folder, subfolder_name)
            
            if not os.path.isdir(subfolder_path):
                continue
            
            for quality in ['bad', 'good']:
                quality_path = os.path.join(subfolder_path, quality)
                
                if not os.path.isdir(quality_path):
                    continue
                
                images_path = os.path.join(quality_path, 'images')
                
                if not os.path.isdir(images_path):
                    continue
                
                img_files = glob.glob(os.path.join(images_path, '*'))
                img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in IMG_EXTS]
                
                for img_file in img_files:
                    img_name = os.path.basename(img_file)
                    original_name_without_ext = os.path.splitext(img_name)[0]
                    original_folders.add(original_name_without_ext)
    
    return original_folders


def collect_bolt_dino_images(base_folders, subfolder_names, use_4class=False, quality_filter=None):
    """Bolt 모드: DINO용 크롭된 이미지들을 수집하는 함수"""
    all_images = []
    
    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"기본 폴더가 존재하지 않습니다: {base_folder}")
            continue
            
        print(f"\n=== {base_folder}에서 Bolt DINO 이미지 수집 ===")
        
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
                    
                    for subdir in ['0', '1']:
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
                            
                            # 라벨 결정
                            if use_4class:
                                if quality == 'good' and subdir == '0':
                                    label = 0  # 정측면 양품
                                elif quality == 'bad' and subdir == '0':
                                    label = 1  # 정측면 불량
                                elif quality == 'good' and subdir == '1':
                                    label = 2  # 측면 양품
                                else:
                                    label = 3  # 측면 불량
                            else:
                                label = 1 if quality == 'bad' else 0
                            
                            # 원본 이미지 ID 추출
                            uuid_pattern = r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
                            match = re.search(uuid_pattern, img_name)
                            
                            if match:
                                uuid_end = match.end()
                                img_name_no_suffix = img_name[:uuid_end] + ('.jpg' if img_name.endswith('.jpg') else '.png')
                                original_image_id = extract_image_id(img_name_no_suffix)
                            else:
                                original_image_id = extract_image_id(img_name)
                            
                            img_info = {
                                'path': absolute_path,
                                'subfolder': subfolder_name,
                                'quality': quality,
                                'crop_folder': f"{crop_folder}/{subdir}",
                                'label': label,
                                'is_augmented': '_aug' in crop_folder,
                                'original_image_id': original_image_id,
                                'base_folder': folder_name,
                                'image_id': extract_image_id(img_name)
                            }
                            all_images.append(img_info)
    
    return all_images


def collect_door_dino_images(base_folders, subfolder_names, original_folders, merge_classes=False, target_areas=None):
    """Door 모드: DINO용 크롭된 이미지들을 수집하는 함수"""
    all_images = []
    
    if target_areas is None:
        crop_areas = ['crop_high', 'crop_mid', 'crop_low']
    else:
        area_map = {
            'high': 'crop_high',
            'mid': 'crop_mid',
            'low': 'crop_low'
        }
        crop_areas = [area_map[a] for a in target_areas if a in area_map]
        if not crop_areas:
            print(f"유효한 영역이 없습니다: {target_areas}")
            return all_images
    
    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"기본 폴더가 존재하지 않습니다: {base_folder}")
            continue
            
        print(f"\n=== {base_folder}에서 Door DINO 이미지 수집 ===")
        
        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(base_folder, subfolder_name)
            
            if not os.path.isdir(subfolder_path):
                print(f"  하위폴더가 존재하지 않습니다: {subfolder_name}")
                continue
                
            print(f"  하위폴더: {subfolder_name}")
            
            for quality in ['bad', 'good']:
                quality_path = os.path.join(subfolder_path, quality)
                
                if not os.path.isdir(quality_path):
                    print(f"    {quality} 폴더가 존재하지 않습니다")
                    continue
                
                crop_folders = []
                for area in crop_areas:
                    crop_folders.append(area)
                    crop_folders.append(area + '_aug')
                
                for crop_folder in crop_folders:
                    crop_path = os.path.join(quality_path, crop_folder)
                    
                    if not os.path.isdir(crop_path):
                        print(f"    {quality}/{crop_folder} 폴더가 존재하지 않습니다")
                        continue
                    
                    area_labels = {'crop_high': '0', 'crop_mid': '1', 'crop_low': '2'}
                    area_label = area_labels.get(crop_folder.replace('_aug', ''), None)
                    
                    # good은 0만, bad는 1,2,3,4만
                    if quality == 'good':
                        subdirs = ['0']
                    else:
                        subdirs = ['1', '2', '3', '4']
                    
                    for subdir in subdirs:
                        subdir_path = os.path.join(crop_path, subdir)
                        
                        if not os.path.isdir(subdir_path):
                            continue
                        
                        crop_count = 0
                        for original_name_without_ext in original_folders:
                            # 일반 크롭 이미지
                            crop_filename = f"{original_name_without_ext}_{area_label}.jpg"
                            crop_file_path = os.path.join(subdir_path, crop_filename)
                            
                            if os.path.exists(crop_file_path):
                                if base_folder.startswith('/home/ciw/work/datasets/'):
                                    folder_name = base_folder[len('/home/ciw/work/datasets/'):]
                                else:
                                    folder_name = os.path.basename(base_folder.rstrip(os.sep))
                                
                                absolute_path = f"/home/ciw/work/datasets/{folder_name}/{subfolder_name}/{quality}/{crop_folder}/{subdir}/{crop_filename}"
                                
                                # 라벨 결정
                                original_label = int(subdir)
                                if merge_classes and original_label in [1, 2, 3]:
                                    label = 1
                                else:
                                    label = original_label
                                
                                # original_image_id를 YOLO의 image_id와 동일하게 만들기 위해 extract_image_id 사용
                                original_img_name = f"{original_name_without_ext}.jpg"
                                original_image_id = extract_image_id(original_img_name)
                                
                                img_info = {
                                    'path': absolute_path,
                                    'subfolder': subfolder_name,
                                    'quality': quality,
                                    'crop_folder': f"{crop_folder}/{subdir}",
                                    'label': label,
                                    'area_label': area_label,
                                    'is_augmented': '_aug' in crop_folder,
                                    'original_img_name': original_img_name,
                                    'original_image_id': original_image_id,
                                    'base_folder': folder_name,
                                    'image_id': extract_image_id_door(crop_filename)
                                }
                                all_images.append(img_info)
                                crop_count += 1
                        
                        # 크롭 증강 이미지
                        if '_aug' in crop_folder:
                            crop_aug_types = ['bright', 'contrast', 'flip', 'gray', 'noise', 'rot']
                            
                            for original_name_without_ext in original_folders:
                                for aug_type in crop_aug_types:
                                    crop_aug_filename = f"{original_name_without_ext}_{area_label}_{aug_type}.jpg"
                                    crop_aug_file_path = os.path.join(subdir_path, crop_aug_filename)
                                    
                                    if os.path.exists(crop_aug_file_path):
                                        if base_folder.startswith('/home/ciw/work/datasets/'):
                                            folder_name = base_folder[len('/home/ciw/work/datasets/'):]
                                        else:
                                            folder_name = os.path.basename(base_folder.rstrip(os.sep))
                                        
                                        absolute_path = f"/home/ciw/work/datasets/{folder_name}/{subfolder_name}/{quality}/{crop_folder}/{subdir}/{crop_aug_filename}"
                                        
                                        original_label = int(subdir)
                                        if merge_classes and original_label in [1, 2, 3]:
                                            label = 1
                                        else:
                                            label = original_label
                                        
                                        # original_image_id를 YOLO의 image_id와 동일하게 만들기 위해 extract_image_id 사용
                                        original_img_name = f"{original_name_without_ext}.jpg"
                                        original_image_id = extract_image_id(original_img_name)
                                        
                                        img_info = {
                                            'path': absolute_path,
                                            'subfolder': subfolder_name,
                                            'quality': quality,
                                            'crop_folder': f"{crop_folder}/{subdir}",
                                            'label': label,
                                            'area_label': area_label,
                                            'is_augmented': True,
                                            'original_img_name': original_img_name,
                                            'original_image_id': original_image_id,
                                            'base_folder': folder_name,
                                            'image_id': extract_image_id_door(crop_aug_filename)
                                        }
                                        all_images.append(img_info)
                                        crop_count += 1
                        
                        if crop_count > 0:
                            print(f"    {quality}/{crop_folder}/{subdir}: {crop_count}개 이미지 발견")
    
    return all_images


def unified_stratified_split(yolo_images, dino_images, ratios):
    """YOLO와 DINO 이미지를 동일한 기준으로 분할하는 함수 (sealing_split.py 로직 참고)"""
    # 1. YOLO 이미지를 기준으로 분할 키 생성
    split_keys = set()
    for img in yolo_images:
        if not img.get('is_augmented', False):  # 원본만 사용
            key = img['image_id']
            split_keys.add(key)
    
    print(f"\n=== 분할 키 생성 ===")
    print(f"총 분할 키 수: {len(split_keys)}개")
    
    # 2. 각 키별로 good/bad 비율을 유지하면서 stratified split 수행
    key_quality_groups = defaultdict(lambda: {'good': [], 'bad': []})
    for img in yolo_images:
        if not img.get('is_augmented', False):
            key = img['image_id']
            quality = img['quality']
            key_quality_groups[key][quality].append(img)
    
    # 각 키별로 good/bad 비율 계산
    key_ratios = {}
    for key in split_keys:
        good_count = len(key_quality_groups[key]['good'])
        bad_count = len(key_quality_groups[key]['bad'])
        total_count = good_count + bad_count
        if total_count > 0:
            good_ratio = good_count / total_count
            key_ratios[key] = good_ratio
        else:
            key_ratios[key] = 0.0
    
    # good/bad 비율별로 키들을 그룹화
    ratio_groups = defaultdict(list)
    for key, ratio in key_ratios.items():
        ratio_groups[ratio].append(key)
    
    print(f"good/bad 비율 그룹 수: {len(ratio_groups)}개")
    for ratio, keys in list(ratio_groups.items())[:5]:
        print(f"  비율 {ratio:.2f}: {len(keys)}개 키")
    
    # 각 비율 그룹별로 stratified split 수행
    train_keys = set()
    val_keys = set()
    test_keys = set()
    
    for ratio, keys in ratio_groups.items():
        random.shuffle(keys)
        n_total = len(keys)
        n_train = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])
        n_test = n_total - n_train - n_val
        
        for key in keys[:n_train]:
            train_keys.add(key)
        for key in keys[n_train:n_train+n_val]:
            val_keys.add(key)
        for key in keys[n_train+n_val:]:
            test_keys.add(key)
    
    print(f"train 키: {len(train_keys)}개")
    print(f"val 키: {len(val_keys)}개")
    print(f"test 키: {len(test_keys)}개")
    
    # 3. YOLO 이미지 분할
    yolo_original = [img for img in yolo_images if not img.get('is_augmented', False)]
    yolo_augmented = [img for img in yolo_images if img.get('is_augmented', False)]
    
    yolo_train = []
    yolo_val = []
    yolo_test = []
    
    for img in yolo_original:
        key = img['image_id']
        if key in train_keys:
            yolo_train.append(img)
        elif key in val_keys:
            yolo_val.append(img)
        elif key in test_keys:
            yolo_test.append(img)
    
    # YOLO 증강 이미지 추가 (train에만)
    yolo_aug_matching_count = 0
    for aug_img in yolo_augmented:
        key = aug_img['image_id']
        if key in train_keys:
            yolo_train.append(aug_img)
            yolo_aug_matching_count += 1
    
    print(f"YOLO train에 추가된 증강 이미지: {yolo_aug_matching_count}개")
    
    # 4. DINO 이미지 분할
    dino_original = [img for img in dino_images if not img.get('is_augmented', False)]
    dino_aug = [img for img in dino_images if img.get('is_augmented', False)]
    
    print(f"\n=== DINO 이미지 분류 ===")
    print(f"원본 DINO 이미지: {len(dino_original)}개")
    print(f"증강 DINO 이미지: {len(dino_aug)}개")
    
    dino_train = []
    dino_val = []
    dino_test = []
    
    matched_count = 0
    unmatched_count = 0
    
    for img in dino_original:
        key = img['original_image_id']
        if key in train_keys:
            dino_train.append(img)
            matched_count += 1
        elif key in val_keys:
            dino_val.append(img)
            matched_count += 1
        elif key in test_keys:
            dino_test.append(img)
            matched_count += 1
        else:
            unmatched_count += 1
    
    print(f"매칭된 DINO 원본 이미지: {matched_count}개")
    print(f"매칭되지 않은 DINO 원본 이미지: {unmatched_count}개")
    
    # train에 선택된 원본 이미지들의 키 추출
    train_original_keys = set()
    for img in dino_train:
        key = img['original_image_id']
        train_original_keys.add(key)
    
    print(f"train에 선택된 원본 이미지 키: {len(train_original_keys)}개")
    
    # train에 속한 원본에서 생성된 증강 이미지들을 train에 추가
    matching_aug_count = 0
    for aug_img in dino_aug:
        key = aug_img['original_image_id']
        if key in train_original_keys:
            dino_train.append(aug_img)
            matching_aug_count += 1
    
    print(f"train에 추가된 증강 이미지: {matching_aug_count}개")
    
    print(f"\n=== 최종 분할 결과 ===")
    print(f"YOLO - train: {len(yolo_train)}, val: {len(yolo_val)}, test: {len(yolo_test)}")
    print(f"DINO - train: {len(dino_train)}, val: {len(dino_val)}, test: {len(dino_test)}")
    
    # 최종 리스트를 섞기
    random.shuffle(yolo_train)
    random.shuffle(yolo_val)
    random.shuffle(yolo_test)
    random.shuffle(dino_train)
    random.shuffle(dino_val)
    random.shuffle(dino_test)
    
    return (yolo_train, yolo_val, yolo_test), (dino_train, dino_val, dino_test)


def write_split_files(yolo_splits, dino_splits, name=''):
    """분할된 이미지들을 파일에 저장하는 함수"""
    txt_dir = Path('TXT')
    txt_dir.mkdir(parents=True, exist_ok=True)
    
    if name:
        yolo_train_file = txt_dir / f'train_{name}.txt'
        yolo_val_file = txt_dir / f'val_{name}.txt'
        yolo_test_file = txt_dir / f'test_{name}.txt'
        
        dino_train_file = txt_dir / f'train_dino_{name}.txt'
        dino_val_file = txt_dir / f'val_dino_{name}.txt'
        dino_test_file = txt_dir / f'test_dino_{name}.txt'
    else:
        yolo_train_file = txt_dir / 'train.txt'
        yolo_val_file = txt_dir / 'val.txt'
        yolo_test_file = txt_dir / 'test.txt'
        
        dino_train_file = txt_dir / 'train_dino.txt'
        dino_val_file = txt_dir / 'val_dino.txt'
        dino_test_file = txt_dir / 'test_dino.txt'
    
    # YOLO용 파일 저장 (경로만)
    yolo_train, yolo_val, yolo_test = yolo_splits
    missing_yolo = []
    
    def write_paths(file_path, imgs):
        written = 0
        with open(file_path, 'w') as f:
            for img in imgs:
                p = img['path']
                if os.path.isfile(p):
                    f.write(f"{p}\n")
                    written += 1
                else:
                    missing_yolo.append(p)
        return written
    
    yolo_train_written = write_paths(yolo_train_file, yolo_train)
    yolo_val_written = write_paths(yolo_val_file, yolo_val)
    yolo_test_written = write_paths(yolo_test_file, yolo_test)
    
    if missing_yolo:
        miss_file = txt_dir / f'missing_yolo_{name if name else "default"}.txt'
        with open(miss_file, 'w') as mf:
            for p in missing_yolo:
                mf.write(p + '\n')
    
    # DINO용 파일 저장 (경로 + 라벨)
    dino_train, dino_val, dino_test = dino_splits
    missing_dino = []
    
    def write_paths_with_label(file_path, imgs):
        written = 0
        with open(file_path, 'w') as f:
            for img in imgs:
                p = img['path']
                if os.path.isfile(p):
                    f.write(f"{p} {img['label']}\n")
                    written += 1
                else:
                    missing_dino.append(p)
        return written
    
    dino_train_written = write_paths_with_label(dino_train_file, dino_train)
    dino_val_written = write_paths_with_label(dino_val_file, dino_val)
    dino_test_written = write_paths_with_label(dino_test_file, dino_test)
    
    if missing_dino:
        miss_file = txt_dir / f'missing_dino_{name if name else "default"}.txt'
        with open(miss_file, 'w') as mf:
            for p in missing_dino:
                mf.write(p + '\n')
    
    print(f"\n=== 분할 결과 ===")
    print(f"YOLO용:")
    print(f"  train: {len(yolo_train)}개 (실제 기록: {yolo_train_written}) -> {yolo_train_file}")
    print(f"  val: {len(yolo_val)}개 (실제 기록: {yolo_val_written}) -> {yolo_val_file}")
    print(f"  test: {len(yolo_test)}개 (실제 기록: {yolo_test_written}) -> {yolo_test_file}")
    print(f"DINO용:")
    print(f"  train: {len(dino_train)}개 (실제 기록: {dino_train_written}) -> {dino_train_file}")
    print(f"  val: {len(dino_val)}개 (실제 기록: {dino_val_written}) -> {dino_val_file}")
    print(f"  test: {len(dino_test)}개 (실제 기록: {dino_test_written}) -> {dino_test_file}")


def main():
    parser = argparse.ArgumentParser(description='YOLO와 DINO를 위한 통합 학습 데이터셋 split을 생성합니다.')
    parser.add_argument('--mode', choices=['bolt', 'door'], required=True,
                       help='모드 선택: bolt 또는 door')
    parser.add_argument('--folders', nargs='+',
                       help='분석할 기본 폴더 날짜들 (일반 폴더)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                       help='일반 폴더 날짜 구간 선택 (MMDD). 예: --date-range 0807 1103')
    parser.add_argument('--obb-folders', nargs='+',
                       help='분석할 OBB 폴더 날짜들')
    parser.add_argument('--obb-date-range', nargs=2, metavar=('START', 'END'),
                       help='OBB 폴더 날짜 구간 선택 (MMDD). 예: --obb-date-range 0718 0806')
    parser.add_argument('--subfolders', nargs='+', required=True,
                       help='찾을 하위폴더들 (여러 개 가능)')
    parser.add_argument('--name', type=str, default='',
                       help='출력 파일명에 사용할 이름')
    
    # Bolt 모드 옵션
    parser.add_argument('--bolt-4class', action='store_true',
                       help='Bolt 모드: 4클래스 사용 (정측면 양품/불량, 측면 양품/불량)')
    parser.add_argument('--bad-date-range', nargs=2, metavar=('START', 'END'),
                       help='Bolt 모드: bad용 일반 폴더 날짜 구간 (MMDD)')
    parser.add_argument('--good-date-range', nargs=2, metavar=('START', 'END'),
                       help='Bolt 모드: good용 일반 폴더 날짜 구간 (MMDD)')
    
    # Door 모드 옵션
    parser.add_argument('--merge-classes', action='store_true',
                       help='Door 모드: 클래스 1,2,3을 1로 합침')
    
    args = parser.parse_args()
    
    base_path = "/home/ciw/work/datasets"
    
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
    
    print(f"모드: {args.mode}")
    if target_folders:
        print(f"일반 폴더들: {[os.path.basename(p) for p in target_folders]}")
    if obb_folders:
        print(f"OBB 폴더들: {[os.path.basename(p) for p in obb_folders]}")
    print(f"찾을 하위폴더들: {args.subfolders}")
    if args.name:
        print(f"출력 파일 이름: {args.name}")
    
    # YOLO 이미지 수집
    all_yolo_images = []
    
    if args.mode == 'bolt' and (args.bad_date_range or args.good_date_range):
        # bad/good 각각 별도 날짜 범위
        bad_base_folders = target_folders
        good_base_folders = target_folders
        
        if args.bad_date_range:
            b_start, b_end = args.bad_date_range
            bad_base_folders = collect_date_range_folders(base_path, b_start, b_end)
        if args.good_date_range:
            g_start, g_end = args.good_date_range
            good_base_folders = collect_date_range_folders(base_path, g_start, g_end)
        
        # bad 일반 폴더 수집
        if bad_base_folders:
            bad_imgs = collect_yolo_images_from_folders(bad_base_folders, args.subfolders, is_obb=False, quality_filter='bad')
            all_yolo_images.extend(bad_imgs)
        # bad OBB 폴더 수집
        if obb_folders:
            bad_obb_imgs = collect_yolo_images_from_folders(obb_folders, args.subfolders, is_obb=True, quality_filter='bad')
            all_yolo_images.extend(bad_obb_imgs)
        # good 일반 폴더 수집
        if good_base_folders:
            good_imgs = collect_yolo_images_from_folders(good_base_folders, args.subfolders, is_obb=False, quality_filter='good')
            all_yolo_images.extend(good_imgs)
        # good OBB 폴더 수집
        if obb_folders:
            good_obb_imgs = collect_yolo_images_from_folders(obb_folders, args.subfolders, is_obb=True, quality_filter='good')
            all_yolo_images.extend(good_obb_imgs)
    else:
        # 일반 처리
        if target_folders:
            normal_imgs = collect_yolo_images_from_folders(target_folders, args.subfolders, is_obb=False)
            all_yolo_images.extend(normal_imgs)
        if obb_folders:
            obb_imgs = collect_yolo_images_from_folders(obb_folders, args.subfolders, is_obb=True)
            all_yolo_images.extend(obb_imgs)
    
    if not all_yolo_images:
        print("수집된 YOLO용 이미지가 없습니다.")
        return
    
    print(f"\n총 {len(all_yolo_images)}개 YOLO용 이미지 수집 완료")
    
    # DINO 이미지 수집
    if args.mode == 'bolt':
        if args.bolt_4class:
            print("Bolt 모드: 4클래스 사용")
        else:
            print("Bolt 모드: 2클래스 사용")
        
        if args.bad_date_range or args.good_date_range:
            bad_base_folders = target_folders
            good_base_folders = target_folders
            
            if args.bad_date_range:
                b_start, b_end = args.bad_date_range
                bad_base_folders = collect_date_range_folders(base_path, b_start, b_end)
            if args.good_date_range:
                g_start, g_end = args.good_date_range
                good_base_folders = collect_date_range_folders(base_path, g_start, g_end)
            
            bad_folders = bad_base_folders + obb_folders
            good_folders = good_base_folders + obb_folders
            
            dino_images = []
            if bad_folders:
                dino_images.extend(collect_bolt_dino_images(bad_folders, args.subfolders,
                                                           use_4class=args.bolt_4class,
                                                           quality_filter='bad'))
            if good_folders:
                dino_images.extend(collect_bolt_dino_images(good_folders, args.subfolders,
                                                           use_4class=args.bolt_4class,
                                                           quality_filter='good'))
        else:
            dino_images = collect_bolt_dino_images(all_folders, args.subfolders, args.bolt_4class)
    
    elif args.mode == 'door':
        merge_classes = args.merge_classes
        if merge_classes:
            print("Door 모드: 클래스 1,2,3을 1로 합침")
        else:
            print("Door 모드: 클래스 1,2,3을 각각 유지")
        
        # 원본 폴더명 수집
        original_folders = collect_original_folders(all_folders, args.subfolders)
        print(f"\n총 {len(original_folders)}개 원본 폴더명 수집 완료")
        
        if not original_folders:
            print("수집된 원본 폴더명이 없습니다.")
            return
        
        dino_images = collect_door_dino_images(all_folders, args.subfolders, original_folders, merge_classes)
    
    if not dino_images:
        print("수집된 DINO용 이미지가 없습니다.")
        return
    
    print(f"\n총 {len(dino_images)}개 DINO용 이미지 수집 완료")
    
    # 통합 분할 수행
    yolo_splits, dino_splits = unified_stratified_split(all_yolo_images, dino_images, SPLIT_RATIO)
    
    # 파일에 저장
    write_split_files(yolo_splits, dino_splits, args.name)


if __name__ == '__main__':
    main()

