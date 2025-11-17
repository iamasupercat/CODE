#!/usr/bin/env python3
"""
사용법:
    # YOLO용과 ResNet용 txt 파일 동시 생성 (name 옵션 사용):
    python sealing_split.py \
        --folders 0616 0721 0728 0729 0731 0801 0804 0805 0806 \
        --subfolders frontdoor \
        --name frontdoor

    # 결과: 
    # YOLO용: TXT/train_frontdoor.txt, TXT/val_frontdoor.txt, TXT/test_frontdoor.txt
    # ResNet용: TXT/train_resnet_high_frontdoor.txt, TXT/val_resnet_high_frontdoor.txt, TXT/test_resnet_high_frontdoor.txt



YOLO와 ResNet 훈련용 split을 동시에 생성하는 통합 스크립트
원본 이미지와 크롭된 이미지를 동일한 기준으로 분할하여 일관성을 보장합니다.

라벨번호: 0(high), 1(mid), 2(low) #영역구분용 라벨
서브폴더 번호: 0(출고실링), 1(실링없음), 2(작업실링), 3(테이프실링) #ResNet용 라벨  

크롭이미지 파일명 규칙) 원본이미지파일명_라벨번호.jpg
high: 원본이미지파일명_0.jpg
mid: 원본이미지파일명_1.jpg
low: 원본이미지파일명_2.jpg

크롭 증강이미지 파일명 규칙) 원본이미지파일명_라벨번호_증강기법.jpg
증강종류) bright, contrast, flip, gray, noise, rot
예시: 원본이미지파일명_0_rot.jpg

원본 증강이미지 파일명 규칙) 원본이미지파일명_증강기법.jpg
증강종류) flip, noise, invert
예시: 원본이미지파일명_flip.jpg


폴더 구조:
    target_dir/
    ├── subfolder/
    │   ├── bad/
    │   │   ├── images/          # 원본 이미지들
    │   │   ├── labels/          # YOLO 라벨 파일들
    │   │   ├── crop_high/        
    │   │   │  ├──0/       # 출고실링(양품)
    │   │   │  ├──1/       # 실링없음
    │   │   │  ├──2/       # 작업실링
    │   │   │  └──3/       # 테이프실링
    │   │   ├── crop_mid/        
    │   │   │  ├──0/       # 출고실링(양품)
    │   │   │  ├──1/       # 실링없음
    │   │   │  ├──2/       # 작업실링
    │   │   │  └──3/       # 테이프실링
    │   │   ├── crop_low/        
    │   │   │  ├──0/       # 출고실링(양품)
    │   │   │  ├──1/       # 실링없음
    │   │   │  ├──2/       # 작업실링
    │   │   │  └──3/       # 테이프실링
    │   │   ├── crop_high_aug/        
    │   │   │  ├──0/       # 출고실링(양품)_aug
    │   │   │  ├──1/       # 실링없음_aug
    │   │   │  ├──2/       # 작업실링_aug
    │   │   │  └──3/       # 테이프실링_aug
    │   │   ├── crop_mid_aug/        
    │   │   │  ├──0/       # 출고실링(양품)_aug
    │   │   │  ├──1/       # 실링없음_aug
    │   │   │  ├──2/       # 작업실링_aug
    │   │   │  └──3/       # 테이프실링_aug
    │   │   ├── crop_low_aug/        
    │   │   │  ├──0/       # 출고실링(양품)_aug
    │   │   │  ├──1/       # 실링없음_aug
    │   │   │  ├──2/       # 작업실링_aug
    │   │   │  └──3/       # 테이프실링_aug
    │   │   └── debug_crop/      # 디버그용 이미지들 (출력)
    │   └── good/
    │       ├── images/          # 원본 이미지들
    │       ├── labels/          # YOLO 라벨 파일들
    │       ├── crop_high/        
    │       │  ├──0/       # 출고실링(양품)
    │       │  ├──1/       # 실링없음
    │       │  ├──2/       # 작업실링
    │       │  └──3/       # 테이프실링
    │       ├── crop_mid/        
    │       │  ├──0/       # 출고실링(양품)
    │       │  ├──1/       # 실링없음
    │       │  ├──2/       # 작업실링
    │       │  └──3/       # 테이프실링
    │       ├── crop_low/        
    │       │  ├──0/       # 출고실링(양품)
    │       │  ├──1/       # 실링없음
    │       │  ├──2/       # 작업실링
    │       │  └──3/       # 테이프실링
    │       ├── crop_high_aug/        
    │       │  ├──0/       # 출고실링(양품)_aug
    │       │  ├──1/       # 실링없음_aug
    │       │  ├──2/       # 작업실링_aug
    │       │  └──3/       # 테이프실링_aug
    │       ├── crop_mid_aug/        
    │       │  ├──0/       # 출고실링(양품)_aug
    │       │  ├──1/       # 실링없음_aug
    │       │  ├──2/       # 작업실링_aug
    │       │  └──3/       # 테이프실링_aug
    │       ├── crop_low_aug/        
    │       │  ├──0/       # 출고실링(양품)_aug
    │       │  ├──1/       # 실링없음_aug
    │       │  ├──2/       # 작업실링_aug
    │       │  └──3/       # 테이프실링_aug
    │       └── debug_crop/      # 디버그용 이미지들 (출력)

"""

import os
import random
import argparse
import glob
from pathlib import Path
from collections import defaultdict

random.seed(42)

SPLIT_RATIO = [0.8, 0.1, 0.1]  # train, val, test (8:1:1)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def extract_image_id(img_name: str) -> str:
    """이미지명에서 원본 이미지 파일명 추출"""
    # 확장자 제거
    img_name_without_ext = os.path.splitext(img_name)[0]
    
    # 증강기법 목록
    # 원본 증강: flip, noise, invert
    # 크롭 증강: bright, contrast, flip, gray, noise, rot
    original_aug_types = ['flip', 'noise', 'invert']
    crop_aug_types = ['bright', 'contrast', 'flip', 'gray', 'noise', 'rot']
    
    # 크롭 증강 이미지: 원본파일명_라벨링번호_증강기법
    if '_' in img_name_without_ext:
        parts = img_name_without_ext.split('_')
        if len(parts) >= 3 and parts[-1] in crop_aug_types and parts[-2].isdigit():
            # 원본파일명만 추출
            return '_'.join(parts[:-2])
    
    # 원본 증강 이미지: 원본파일명_증강기법
    if '_' in img_name_without_ext:
        parts = img_name_without_ext.split('_')
        if len(parts) >= 2 and parts[-1] in original_aug_types:
            # 원본파일명만 추출
            return '_'.join(parts[:-1])
    
    # 크롭 이미지: 원본파일명_라벨링번호 (0, 1, 2)
    if '_' in img_name_without_ext:
        parts = img_name_without_ext.split('_')
        if len(parts) >= 2 and parts[-1] in ['0', '1', '2']:
            # 원본파일명만 추출
            return '_'.join(parts[:-1])
    
    # 원본 이미지: 그대로 반환
    return img_name_without_ext

def collect_original_folders(base_folders, subfolder_names):
    """원본 폴더명들을 수집하는 함수"""
    original_folders = set()
    
    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"기본 폴더가 존재하지 않습니다: {base_folder}")
            continue
            
        print(f"\n=== {base_folder}에서 원본 폴더명 수집 ===")
        
        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(base_folder, subfolder_name)
            
            if not os.path.isdir(subfolder_path):
                print(f"  하위폴더가 존재하지 않습니다: {subfolder_name}")
                continue
                
            print(f"  하위폴더: {subfolder_name}")
            
            # bad와 good 폴더 확인
            for quality in ['bad', 'good']:
                quality_path = os.path.join(subfolder_path, quality)
                
                if not os.path.isdir(quality_path):
                    print(f"    {quality} 폴더가 존재하지 않습니다")
                    continue
                
                images_path = os.path.join(quality_path, 'images')
                
                if not os.path.isdir(images_path):
                    print(f"    {quality}/images 폴더가 존재하지 않습니다")
                    continue
                
                # images 폴더의 모든 이미지 파일 찾기
                img_files = glob.glob(os.path.join(images_path, '*'))
                img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in IMG_EXTS]
                
                if not img_files:
                    print(f"    {quality}/images 폴더에 이미지 파일이 없습니다")
                    continue
                
                print(f"    {quality}: {len(img_files)}개 이미지 발견")
                
                # 원본 폴더명 수집 (확장자 제거)
                for img_file in img_files:
                    img_name = os.path.basename(img_file)
                    original_name_without_ext = os.path.splitext(img_name)[0]
                    original_folders.add(original_name_without_ext)
    
    print(f"\n총 {len(original_folders)}개 원본 폴더명 수집 완료")
    return original_folders

def collect_yolo_images_from_original_folders(base_folders, subfolder_names, original_folders):
    """원본 폴더명을 기반으로 YOLO 훈련용 이미지들을 수집하는 함수"""
    all_images = []
    
    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"기본 폴더가 존재하지 않습니다: {base_folder}")
            continue
            
        print(f"\n=== {base_folder}에서 YOLO용 이미지 수집 ===")
        
        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(base_folder, subfolder_name)
            
            if not os.path.isdir(subfolder_path):
                print(f"  하위폴더가 존재하지 않습니다: {subfolder_name}")
                continue
                
            print(f"  하위폴더: {subfolder_name}")
            
            # bad와 good 폴더 확인
            for quality in ['bad', 'good']:
                quality_path = os.path.join(subfolder_path, quality)
                
                if not os.path.isdir(quality_path):
                    print(f"    {quality} 폴더가 존재하지 않습니다")
                    continue
                
                images_path = os.path.join(quality_path, 'images')
                
                if not os.path.isdir(images_path):
                    print(f"    {quality}/images 폴더가 존재하지 않습니다")
                    continue
                
                # 원본 폴더명을 기반으로 이미지 찾기
                original_count = 0
                aug_count = 0
                
                for original_name_without_ext in original_folders:
                    # 원본 이미지 확인
                    original_filename = f"{original_name_without_ext}.jpg"
                    original_file_path = os.path.join(images_path, original_filename)
                    
                    if os.path.exists(original_file_path):
                        # 상대경로를 절대경로로 변환
                        if base_folder.startswith('./'):
                            folder_name = base_folder[2:]  # ./ 제거
                        elif base_folder.startswith('.'):
                            folder_name = base_folder[1:]  # . 제거
                        elif base_folder.startswith('/home/work/datasets/'):
                            # 절대경로인 경우 /home/work/datasets/ 제거
                            folder_name = base_folder[len('/home/work/datasets/'):]
                        else:
                            folder_name = base_folder
                        
                        # 절대경로 생성
                        absolute_path = f"/home/work/datasets/{folder_name}/{subfolder_name}/{quality}/images/{original_filename}"
                        
                        # 이미지 정보 저장
                        img_info = {
                            'path': absolute_path,
                            'subfolder': subfolder_name,
                            'quality': quality,
                            'base_folder': folder_name,
                            'img_name': original_filename,
                            'image_id': original_name_without_ext,
                            'is_augmented': False  # 원본 이미지
                        }
                        all_images.append(img_info)
                        original_count += 1
                
                # 증강본 수집 - 원본폴더명 기반으로 존재하는 증강 이미지 확인
                aug_subfolder_path = os.path.join(base_folder, f"{subfolder_name}_aug", quality, 'images')
                if os.path.isdir(aug_subfolder_path):
                    # 원본 증강 기법들
                    original_aug_types = ['flip', 'noise', 'invert']
                    
                    for original_name_without_ext in original_folders:
                        for aug_type in original_aug_types:
                            aug_filename = f"{original_name_without_ext}_{aug_type}.jpg"
                            aug_file_path = os.path.join(aug_subfolder_path, aug_filename)
                            
                            if os.path.exists(aug_file_path):
                                # 상대경로를 절대경로로 변환
                                if base_folder.startswith('./'):
                                    folder_name = base_folder[2:]  # ./ 제거
                                elif base_folder.startswith('.'):
                                    folder_name = base_folder[1:]  # . 제거
                                elif base_folder.startswith('/home/work/datasets/'):
                                    # 절대경로인 경우 /home/work/datasets/ 제거
                                    folder_name = base_folder[len('/home/work/datasets/'):]
                                else:
                                    folder_name = base_folder
                                
                                # 절대경로 생성
                                absolute_path = f"/home/work/datasets/{folder_name}/{subfolder_name}_aug/{quality}/images/{aug_filename}"
                                
                                # 이미지 정보 저장
                                img_info = {
                                    'path': absolute_path,
                                    'subfolder': subfolder_name,
                                    'quality': quality,
                                    'base_folder': folder_name,
                                    'img_name': aug_filename,
                                    'image_id': original_name_without_ext,
                                    'is_augmented': True  # 증강 이미지
                                }
                                all_images.append(img_info)
                                aug_count += 1
                
                print(f"    {quality}: 원본 {original_count}개, 증강 {aug_count}개 발견")

    return all_images

def collect_resnet_images_from_original_folders(base_folders, subfolder_names, original_folders):
    """원본 폴더명을 기반으로 ResNet 훈련용 크롭된 이미지들을 수집하는 함수"""
    all_images = []
    
    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"기본 폴더가 존재하지 않습니다: {base_folder}")
            continue
            
        print(f"\n=== {base_folder}에서 ResNet용 크롭 이미지 수집 ===")
        
        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(base_folder, subfolder_name)
            
            if not os.path.isdir(subfolder_path):
                print(f"  하위폴더가 존재하지 않습니다: {subfolder_name}")
                continue
                
            print(f"  하위폴더: {subfolder_name}")
            
            # bad와 good 폴더 확인
            for quality in ['bad', 'good']:
                quality_path = os.path.join(subfolder_path, quality)
                
                if not os.path.isdir(quality_path):
                    print(f"    {quality} 폴더가 존재하지 않습니다")
                    continue
                
                # crop_high, crop_mid, crop_low와 crop_high_aug, crop_mid_aug, crop_low_aug 폴더 확인
                crop_areas = ['crop_high', 'crop_mid', 'crop_low']
                crop_folders = []
                for area in crop_areas:
                    crop_folders.append(area)
                    crop_folders.append(area + '_aug')
                
                for crop_folder in crop_folders:
                    crop_path = os.path.join(quality_path, crop_folder)
                    
                    if not os.path.isdir(crop_path):
                        print(f"    {quality}/{crop_folder} 폴더가 존재하지 않습니다")
                        continue
                    
                    # 0, 1, 2, 3 서브폴더 확인 (실링 클래스별)
                    for subdir in ['0', '1', '2', '3']:
                        subdir_path = os.path.join(crop_path, subdir)
                        
                        if not os.path.isdir(subdir_path):
                            print(f"    {quality}/{crop_folder}/{subdir} 폴더가 존재하지 않습니다")
                            continue
                        
                        # 영역별 라벨 매핑
                        area_labels = {'crop_high': '0', 'crop_mid': '1', 'crop_low': '2'}
                        area_label = area_labels.get(crop_folder.replace('_aug', ''), None)
                        
                        # 원본 폴더명을 기반으로 크롭 이미지 수집
                        crop_count = 0
                        for original_name_without_ext in original_folders:
                            # 일반 크롭 이미지 확인
                            crop_filename = f"{original_name_without_ext}_{area_label}.jpg"
                            crop_file_path = os.path.join(subdir_path, crop_filename)
                            
                            if os.path.exists(crop_file_path):
                                # 상대경로를 절대경로로 변환
                                if base_folder.startswith('./'):
                                    folder_name = base_folder[2:]  # ./ 제거
                                elif base_folder.startswith('.'):
                                    folder_name = base_folder[1:]  # . 제거
                                elif base_folder.startswith('/home/work/datasets/'):
                                    # 절대경로인 경우 /home/work/datasets/ 제거
                                    folder_name = base_folder[len('/home/work/datasets/'):]
                                else:
                                    folder_name = base_folder
                                
                                # 절대경로 생성
                                absolute_path = f"/home/work/datasets/{folder_name}/{subfolder_name}/{quality}/{crop_folder}/{subdir}/{crop_filename}"
                                
                                # 라벨 결정 (서브폴더 번호가 실링 클래스 번호)
                                label = int(subdir)  # 실링 클래스 번호를 라벨로 사용
                                
                                # 이미지 정보 저장
                                img_info = {
                                    'path': absolute_path,
                                    'subfolder': subfolder_name,
                                    'quality': quality,
                                    'crop_folder': f"{crop_folder}/{subdir}",
                                    'label': label,  # 실링 클래스 번호
                                    'area_label': area_label,  # 영역 라벨 (0:high, 1:mid, 2:low)
                                    'is_augmented': '_aug' in crop_folder,
                                    'original_img_name': f"{original_name_without_ext}.jpg",
                                    'original_image_id': original_name_without_ext,
                                    'base_folder': folder_name,
                                    'image_id': extract_image_id(crop_filename)
                                }
                                all_images.append(img_info)
                                crop_count += 1
                        
                        # 크롭 증강 이미지 수집 (crop_*_aug 폴더인 경우)
                        if '_aug' in crop_folder:
                            crop_aug_types = ['bright', 'contrast', 'flip', 'gray', 'noise', 'rot']
                            
                            for original_name_without_ext in original_folders:
                                for aug_type in crop_aug_types:
                                    crop_aug_filename = f"{original_name_without_ext}_{area_label}_{aug_type}.jpg"
                                    crop_aug_file_path = os.path.join(subdir_path, crop_aug_filename)
                                    
                                    if os.path.exists(crop_aug_file_path):
                                        # 상대경로를 절대경로로 변환
                                        if base_folder.startswith('./'):
                                            folder_name = base_folder[2:]  # ./ 제거
                                        elif base_folder.startswith('.'):
                                            folder_name = base_folder[1:]  # . 제거
                                        elif base_folder.startswith('/home/work/datasets/'):
                                            # 절대경로인 경우 /home/work/datasets/ 제거
                                            folder_name = base_folder[len('/home/work/datasets/'):]
                                        else:
                                            folder_name = base_folder
                                        
                                        # 절대경로 생성
                                        absolute_path = f"/home/work/datasets/{folder_name}/{subfolder_name}/{quality}/{crop_folder}/{subdir}/{crop_aug_filename}"
                                        
                                        # 라벨 결정 (서브폴더 번호가 실링 클래스 번호)
                                        label = int(subdir)  # 실링 클래스 번호를 라벨로 사용
                                        
                                        # 이미지 정보 저장
                                        img_info = {
                                            'path': absolute_path,
                                            'subfolder': subfolder_name,
                                            'quality': quality,
                                            'crop_folder': f"{crop_folder}/{subdir}",
                                            'label': label,  # 실링 클래스 번호
                                            'area_label': area_label,  # 영역 라벨 (0:high, 1:mid, 2:low)
                                            'is_augmented': True,
                                            'original_img_name': f"{original_name_without_ext}.jpg",
                                            'original_image_id': original_name_without_ext,
                                            'base_folder': folder_name,
                                            'image_id': extract_image_id(crop_aug_filename)
                                        }
                                        all_images.append(img_info)
                                        crop_count += 1
                        
                        if crop_count > 0:
                            print(f"    {quality}/{crop_folder}/{subdir}: {crop_count}개 이미지 발견")
    
    return all_images

def unified_stratified_split(yolo_images, resnet_images, ratios):
    """YOLO와 ResNet 이미지를 동일한 기준으로 분할하는 함수"""
    # 1. YOLO 이미지를 기준으로 분할 키 생성
    split_keys = set()
    for img in yolo_images:
        # 분할 키: image_id만 사용 (경로 제외)
        key = img['image_id']
        split_keys.add(key)
    
    print(f"\n=== 분할 키 생성 ===")
    print(f"총 분할 키 수: {len(split_keys)}개")
    
    # 2. 각 키별로 good/bad 비율을 유지하면서 stratified split 수행
    # 키별로 good/bad 이미지 수집
    key_quality_groups = defaultdict(lambda: {'good': [], 'bad': []})
    for img in yolo_images:
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
    for ratio, keys in ratio_groups.items():
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
        
        # train 키들 추가
        for key in keys[:n_train]:
            train_keys.add(key)
        
        # val 키들 추가
        for key in keys[n_train:n_train+n_val]:
            val_keys.add(key)
        
        # test 키들 추가
        for key in keys[n_train+n_val:]:
            test_keys.add(key)
    
    print(f"train 키: {len(train_keys)}개")
    print(f"val 키: {len(val_keys)}개")
    print(f"test 키: {len(test_keys)}개")
    
    # 디버깅: 몇 개의 키 샘플 출력
    print(f"\n=== 키 샘플 ===")
    print("YOLO train 키 샘플 (처음 5개):")
    for i, key in enumerate(list(train_keys)[:5]):
        print(f"  {key}")
    
    print("\nResNet 원본 이미지 키 샘플 (처음 5개):")
    # resnet_original이 아직 정의되지 않았으므로 resnet_images에서 원본만 필터링
    resnet_original_sample = [img for img in resnet_images if not img['is_augmented']][:5]
    for i, img in enumerate(resnet_original_sample):
        key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['original_img_name']}"
        print(f"  {key}")
    
    # 4. YOLO 이미지 분할 (원본만 먼저 분할)
    yolo_original = []
    yolo_augmented = []
    
    for img in yolo_images:
        if img.get('is_augmented', False):
            yolo_augmented.append(img)
        else:
            yolo_original.append(img)
    
    # 원본 이미지만 먼저 분할
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
    
    # 5. ResNet 이미지 분할 (원본 이미지 기준으로 매칭)
    resnet_train = []
    resnet_val = []
    resnet_test = []
    
    # 원본 데이터와 증강 데이터 분리
    resnet_original = []
    resnet_aug = []
    
    for img in resnet_images:
        if img['is_augmented']:
            resnet_aug.append(img)
        else:
            resnet_original.append(img)
    
    print(f"\n=== ResNet 이미지 분류 ===")
    print(f"원본 ResNet 이미지: {len(resnet_original)}개")
    print(f"증강 ResNet 이미지: {len(resnet_aug)}개")
    
    # 원본 이미지들을 키 기준으로 분할
    matched_count = 0
    unmatched_count = 0
    
    for img in resnet_original:
        # 원본 이미지 ID만 사용
        key = img['original_image_id']
        if key in train_keys:
            resnet_train.append(img)
            matched_count += 1
        elif key in val_keys:
            resnet_val.append(img)
            matched_count += 1
        elif key in test_keys:
            resnet_test.append(img)
            matched_count += 1
        else:
            unmatched_count += 1
            print(f"  매칭되지 않은 ResNet 이미지: {key}")
    
    print(f"매칭된 ResNet 원본 이미지: {matched_count}개")
    print(f"매칭되지 않은 ResNet 원본 이미지: {unmatched_count}개")
    
    # train에 선택된 원본 이미지들의 키 추출
    train_original_keys = set()
    for img in resnet_train:
        key = img['original_image_id']
        train_original_keys.add(key)
    
    print(f"train에 선택된 원본 이미지 키: {len(train_original_keys)}개")
    
    # train에 선택된 원본 이미지와 매칭되는 증강 이미지들을 train에 추가
    matching_aug_count = 0
    for aug_img in resnet_aug:
        key = aug_img['original_image_id']
        if key in train_original_keys:
            resnet_train.append(aug_img)
            matching_aug_count += 1
    
    print(f"train에 추가된 증강 이미지: {matching_aug_count}개")
    
    # 6. YOLO 증강 이미지 추가 (train에만 포함) - bolt_split.py 참고
    yolo_augmented = [img for img in yolo_images if img.get('is_augmented', False)]
    yolo_aug_matching_count = 0
    
    for aug_img in yolo_augmented:
        key = aug_img['image_id']
        if key in train_keys:
            yolo_train.append(aug_img)
            yolo_aug_matching_count += 1
    
    print(f"YOLO train에 추가된 증강 이미지: {yolo_aug_matching_count}개")
    # print(f"YOLO train에는 원본 이미지만 포함 (증강 이미지 제외)")
    
    print(f"\n=== 최종 분할 결과 ===")
    print(f"YOLO - train: {len(yolo_train)}, val: {len(yolo_val)}, test: {len(yolo_test)}")
    print(f"ResNet - train: {len(resnet_train)}, val: {len(resnet_val)}, test: {len(resnet_test)}")
    
    # 최종 리스트를 섞기
    random.shuffle(yolo_train)
    random.shuffle(yolo_val)
    random.shuffle(yolo_test)
    random.shuffle(resnet_train)
    random.shuffle(resnet_val)
    random.shuffle(resnet_test)
    
    return (yolo_train, yolo_val, yolo_test), (resnet_train, resnet_val, resnet_test)

def write_split_files(yolo_splits, resnet_splits, name=''):
    """분할된 이미지들을 파일에 저장하는 함수"""
    # TXT 폴더가 없으면 생성
    txt_dir = Path('TXT')
    txt_dir.mkdir(parents=True, exist_ok=True)
    
    # name 매개변수를 사용해서 파일명 생성
    if name:
        # YOLO용 파일명
        yolo_train_file = txt_dir / f'train_{name}.txt'
        yolo_val_file = txt_dir / f'val_{name}.txt'
        yolo_test_file = txt_dir / f'test_{name}.txt'
        
        # ResNet용 파일명 (영역별 분리)
        resnet_high_train_file = txt_dir / f'train_resnet_high_{name}.txt'
        resnet_high_val_file = txt_dir / f'val_resnet_high_{name}.txt'
        resnet_high_test_file = txt_dir / f'test_resnet_high_{name}.txt'
        
        resnet_mid_train_file = txt_dir / f'train_resnet_mid_{name}.txt'
        resnet_mid_val_file = txt_dir / f'val_resnet_mid_{name}.txt'
        resnet_mid_test_file = txt_dir / f'test_resnet_mid_{name}.txt'
        
        resnet_low_train_file = txt_dir / f'train_resnet_low_{name}.txt'
        resnet_low_val_file = txt_dir / f'val_resnet_low_{name}.txt'
        resnet_low_test_file = txt_dir / f'test_resnet_low_{name}.txt'
    else:
        # YOLO용 파일명
        yolo_train_file = txt_dir / 'train.txt'
        yolo_val_file = txt_dir / 'val.txt'
        yolo_test_file = txt_dir / 'test.txt'
        
        # ResNet용 파일명 (영역별 분리)
        resnet_high_train_file = txt_dir / 'train_resnet_high.txt'
        resnet_high_val_file = txt_dir / 'val_resnet_high.txt'
        resnet_high_test_file = txt_dir / 'test_resnet_high.txt'
        
        resnet_mid_train_file = txt_dir / 'train_resnet_mid.txt'
        resnet_mid_val_file = txt_dir / 'val_resnet_mid.txt'
        resnet_mid_test_file = txt_dir / 'test_resnet_mid.txt'
        
        resnet_low_train_file = txt_dir / 'train_resnet_low.txt'
        resnet_low_val_file = txt_dir / 'val_resnet_low.txt'
        resnet_low_test_file = txt_dir / 'test_resnet_low.txt'

    # YOLO용 파일 저장 (경로만)
    yolo_train, yolo_val, yolo_test = yolo_splits
    
    with open(yolo_train_file, 'w') as f:
        for img in yolo_train:
            f.write(f"{img['path']}\n")
    
    with open(yolo_val_file, 'w') as f:
        for img in yolo_val:
            f.write(f"{img['path']}\n")
    
    with open(yolo_test_file, 'w') as f:
        for img in yolo_test:
            f.write(f"{img['path']}\n")
    
    # ResNet용 파일 저장 (영역별 분리, 실링 클래스 라벨)
    resnet_train, resnet_val, resnet_test = resnet_splits
    
    # High 영역 파일들
    with open(resnet_high_train_file, 'w') as f:
        for img in resnet_train:
            if img['area_label'] == '0':  # high 영역
                f.write(f"{img['path']} {img['label']}\n")
    
    with open(resnet_high_val_file, 'w') as f:
        for img in resnet_val:
            if img['area_label'] == '0':  # high 영역
                f.write(f"{img['path']} {img['label']}\n")
    
    with open(resnet_high_test_file, 'w') as f:
        for img in resnet_test:
            if img['area_label'] == '0':  # high 영역
                f.write(f"{img['path']} {img['label']}\n")
    
    # Mid 영역 파일들
    with open(resnet_mid_train_file, 'w') as f:
        for img in resnet_train:
            if img['area_label'] == '1':  # mid 영역
                f.write(f"{img['path']} {img['label']}\n")
    
    with open(resnet_mid_val_file, 'w') as f:
        for img in resnet_val:
            if img['area_label'] == '1':  # mid 영역
                f.write(f"{img['path']} {img['label']}\n")
    
    with open(resnet_mid_test_file, 'w') as f:
        for img in resnet_test:
            if img['area_label'] == '1':  # mid 영역
                f.write(f"{img['path']} {img['label']}\n")
    
    # Low 영역 파일들
    with open(resnet_low_train_file, 'w') as f:
        for img in resnet_train:
            if img['area_label'] == '2':  # low 영역
                f.write(f"{img['path']} {img['label']}\n")
    
    with open(resnet_low_val_file, 'w') as f:
        for img in resnet_val:
            if img['area_label'] == '2':  # low 영역
                f.write(f"{img['path']} {img['label']}\n")
    
    with open(resnet_low_test_file, 'w') as f:
        for img in resnet_test:
            if img['area_label'] == '2':  # low 영역
                f.write(f"{img['path']} {img['label']}\n")
    
    # 각 영역별 이미지 수 계산
    high_train_count = len([img for img in resnet_train if img['area_label'] == '0'])
    high_val_count = len([img for img in resnet_val if img['area_label'] == '0'])
    high_test_count = len([img for img in resnet_test if img['area_label'] == '0'])
    
    mid_train_count = len([img for img in resnet_train if img['area_label'] == '1'])
    mid_val_count = len([img for img in resnet_val if img['area_label'] == '1'])
    mid_test_count = len([img for img in resnet_test if img['area_label'] == '1'])
    
    low_train_count = len([img for img in resnet_train if img['area_label'] == '2'])
    low_val_count = len([img for img in resnet_val if img['area_label'] == '2'])
    low_test_count = len([img for img in resnet_test if img['area_label'] == '2'])
    
    print(f"\n=== 분할 결과 ===")
    print(f"YOLO용:")
    print(f"  train: {len(yolo_train)}개 -> {yolo_train_file}")
    print(f"  val: {len(yolo_val)}개 -> {yolo_val_file}")
    print(f"  test: {len(yolo_test)}개 -> {yolo_test_file}")
    print(f"ResNet용 (High 영역):")
    print(f"  train: {high_train_count}개 -> {resnet_high_train_file}")
    print(f"  val: {high_val_count}개 -> {resnet_high_val_file}")
    print(f"  test: {high_test_count}개 -> {resnet_high_test_file}")
    print(f"ResNet용 (Mid 영역):")
    print(f"  train: {mid_train_count}개 -> {resnet_mid_train_file}")
    print(f"  val: {mid_val_count}개 -> {resnet_mid_val_file}")
    print(f"  test: {mid_test_count}개 -> {resnet_mid_test_file}")
    print(f"ResNet용 (Low 영역):")
    print(f"  train: {low_train_count}개 -> {resnet_low_train_file}")
    print(f"  val: {low_val_count}개 -> {resnet_low_val_file}")
    print(f"  test: {low_test_count}개 -> {resnet_low_test_file}")

def main():
    parser = argparse.ArgumentParser(description='YOLO와 ResNet 훈련용 split을 동시에 생성합니다.')
    parser.add_argument('--folders', nargs='+', default=['0616', '0721', '0728', '0729', '0731', '0801', '0804', '0805', '0806'], 
                       help='분석할 기본 폴더 날짜들 (기본값: 0616 0721 0728 0729 0731 0801 0804 0805 0806)')
    parser.add_argument('--subfolders', nargs='+', required=True,
                       help='찾을 하위폴더들 (여러 개 가능)')
    parser.add_argument('--name', type=str, default='',
                       help='출력 파일명에 사용할 이름 (기본값: train.txt, val.txt, test.txt)')
    
    args = parser.parse_args()
    
    # 날짜를 절대경로로 변환
    base_path = "/home/work/datasets"
    target_folders = [os.path.join(base_path, date) for date in args.folders]
    
    print(f"분석할 기본 폴더들: {args.folders}")
    print(f"절대경로: {target_folders}")
    print(f"찾을 하위폴더들: {args.subfolders}")
    if args.name:
        print(f"출력 파일 이름: {args.name}")
    else:
        print("출력 파일: train.txt, val.txt, test.txt")
    
    # 1. 원본 폴더명 수집
    original_folders = collect_original_folders(target_folders, args.subfolders)
    
    if not original_folders:
        print("수집된 원본 폴더명이 없습니다.")
        return
    
    # 2. YOLO용 이미지 수집 (원본 폴더명 기반)
    yolo_images = collect_yolo_images_from_original_folders(target_folders, args.subfolders, original_folders)
    
    if not yolo_images:
        print("수집된 YOLO용 이미지가 없습니다.")
        return
    
    print(f"\n총 {len(yolo_images)}개 YOLO용 이미지 수집 완료")
    
    # 3. ResNet용 크롭 이미지 수집 (원본 폴더명 기반)
    resnet_images = collect_resnet_images_from_original_folders(target_folders, args.subfolders, original_folders)
    
    if not resnet_images:
        print("수집된 ResNet용 이미지가 없습니다.")
        return
    
    print(f"\n총 {len(resnet_images)}개 ResNet용 이미지 수집 완료")
    
    # 4. 통합 분할 수행
    yolo_splits, resnet_splits = unified_stratified_split(yolo_images, resnet_images, SPLIT_RATIO)
    
    # 5. 파일에 저장
    write_split_files(yolo_splits, resnet_splits, args.name)

if __name__ == '__main__':
    main() 