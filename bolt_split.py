#!/usr/bin/env python3
"""
YOLO와 ResNet 훈련용 split을 동시에 생성하는 통합 스크립트
원본 이미지와 크롭된 이미지를 동일한 기준으로 분할하여 일관성을 보장합니다.

# ResNet용 라벨
0 good
1 bad


폴더 구조:
    target_dir/
    ├── subfolder/
    │   ├── bad/
    │   │   ├── images/          # 원본 이미지들
    │   │   ├── labels/          # YOLO 라벨 파일들
    │   │   ├── crop_bolt/        
    │   │   │  ├──0/       # 정측면
    │   │   │  └──1/       # 정면
    │   │   │  └──2/       # 측면
    │   │   ├── crop_bolt_aug/        
    │   │   │  ├──0/       # 정측면_aug
    │   │   │  ├──1/       # 정면_aug
    │   │   │  └──2/       # 측면_aug
    │   │   └── debug_crop/      
    │   └── good/
    │       ├── images/          # 원본 이미지들
    │       ├── labels/          # YOLO 라벨 파일들
    │       ├── crop_bolt/        
    │       │  ├──0/       # 정측면
    │       │  ├──1/       # 정면  
    │       │  └──2/       # 측면
    │       ├── crop_bolt_aug/        
    │       │  ├──0/       # 정측면_aug
    │       │  ├──1/       # 정면_aug
    │       │  └──2/       # 측면_aug
    │       └── debug_crop/      



사용법:
    # 기본 사용 (YOLO용과 ResNet용 txt 파일 동시 생성):
    python bolt_split.py \
        --folders 0616 0718 0721 0725 0728 0729 0731 0801 0804 0805 0806 \
        --subfolders frontfender hood trunklid \
        --name bolt

    # name 옵션 사용:
        --name exp1

    # 결과: 
    # YOLO용: TXT/train_exp1.txt, TXT/val_exp1.txt, TXT/test_exp1.txt
    # ResNet용: TXT/train_resnet_exp1.txt, TXT/val_resnet_exp1.txt, TXT/test_resnet_exp1.txt
"""

import os
import random
import argparse
import glob
from pathlib import Path
from collections import defaultdict
import re

random.seed(42)

SPLIT_RATIO = [0.7, 0.2, 0.1]  # train, val, test

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

def extract_image_id(img_name: str) -> str:
    """이미지명에서 UUID까지 포함한 고유 ID 추출 (bad_XXXX_..._UUID)"""
    # 증강 suffix 제거 (_invert, _blur, _bright 등)
    aug_suffixes = ['_invert', '_blur', '_bright', '_contrast', '_flip', '_gray', '_noise', '_rot']
    img_name_clean = img_name
    
    for suffix in aug_suffixes:
        if img_name_clean.endswith(suffix + '.jpg') or img_name_clean.endswith(suffix + '.png'):
            img_name_clean = img_name_clean[:-len(suffix)] + ('.jpg' if img_name_clean.endswith('.jpg') else '.png')
            break
    
    parts = img_name_clean.split('_')
    for i, part in enumerate(parts):
        if len(part) == 8 and i + 4 < len(parts):
            if (len(parts[i+1]) == 4 and len(parts[i+2]) == 4 and 
                len(parts[i+3]) == 4 and len(parts[i+4]) == 12):
                return '_'.join(parts[:i+5])
    return os.path.splitext(img_name_clean)[0]  # fallback

def collect_yolo_images_from_folders(base_folders, subfolder_names):
    """YOLO 훈련용 원본 이미지들을 수집하는 함수"""
    all_images = []
    
    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"기본 폴더가 존재하지 않습니다: {base_folder}")
            continue
            
        print(f"\n=== {base_folder}에서 YOLO용 원본 이미지 수집 ===")
        
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
                labels_path = os.path.join(quality_path, 'labels')
                
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
                
                # 이미지 파일들을 수집
                for img_file in img_files:
                    img_name = os.path.basename(img_file)
                    
                    # 상대경로를 절대경로로 변환
                    if base_folder.startswith('./'):
                        folder_name = base_folder[2:]  # ./ 제거
                    elif base_folder.startswith('.'):
                        folder_name = base_folder[1:]  # . 제거
                    else:
                        folder_name = base_folder
                    
                    # 절대경로 생성
                    absolute_path = f"/home/work/datasets/{folder_name}/{subfolder_name}/{quality}/images/{img_name}"
                    
                    # 이미지 정보 저장
                    img_info = {
                        'path': absolute_path,
                        'subfolder': subfolder_name,
                        'quality': quality,
                        'base_folder': folder_name,
                        'img_name': img_name,
                        'image_id': extract_image_id(img_name), # image_id 추가
                        'is_augmented': False  # 원본임을 명시
                    }
                    all_images.append(img_info)
                
                # 증강본 수집
                aug_subfolder_path = os.path.join(base_folder, f"{subfolder_name}_aug", quality, 'images')
                if os.path.isdir(aug_subfolder_path):
                    aug_img_files = glob.glob(os.path.join(aug_subfolder_path, '*'))
                    aug_img_files = [f for f in aug_img_files if os.path.splitext(f)[1].lower() in IMG_EXTS]
                    
                    print(f"    {quality} 증강 이미지: {len(aug_img_files)}개 발견")

                    for img_file in aug_img_files:
                        img_name = os.path.basename(img_file)
                        # 증강 이미지도 절대경로로 생성
                        absolute_path = f"/home/work/datasets/{folder_name}/{subfolder_name}_aug/{quality}/images/{img_name}"
                        image_id = extract_image_id(img_name)
                        
                        img_info = {
                            'path': absolute_path,
                            'subfolder': subfolder_name,
                            'quality': quality,
                            'base_folder': folder_name,
                            'img_name': img_name,
                            'image_id': image_id,
                            'is_augmented': True  # 증강 이미지 표시
                        }
                        all_images.append(img_info)
    
    return all_images

def collect_resnet_images_from_folders(base_folders, subfolder_names):
    """ResNet 훈련용 크롭된 이미지들을 수집하는 함수"""
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
                
                # crop_bolt와 crop_bolt_aug 폴더 확인
                crop_folders = ['crop_bolt', 'crop_bolt_aug']
                
                for crop_folder in crop_folders:
                    crop_path = os.path.join(quality_path, crop_folder)
                    
                    if not os.path.isdir(crop_path):
                        print(f"    {quality}/{crop_folder} 폴더가 존재하지 않습니다")
                        continue
                    
                    # 0, 1, 2 서브폴더 확인
                    for subdir in ['0', '1', '2']:
                        subdir_path = os.path.join(crop_path, subdir)
                        
                        if not os.path.isdir(subdir_path):
                            print(f"    {quality}/{crop_folder}/{subdir} 폴더가 존재하지 않습니다")
                            continue
                        
                        # 크롭 폴더의 모든 이미지 파일 찾기
                        img_files = glob.glob(os.path.join(subdir_path, '*'))
                        img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in IMG_EXTS]
                        
                        if not img_files:
                            print(f"    {quality}/{crop_folder}/{subdir} 폴더에 이미지 파일이 없습니다")
                            continue
                        
                        print(f"    {quality}/{crop_folder}/{subdir}: {len(img_files)}개 이미지 발견")
                        
                        # 이미지 파일들을 수집
                        for img_file in img_files:
                            img_name = os.path.basename(img_file)
                            
                            # 상대경로를 절대경로로 변환
                            if base_folder.startswith('./'):
                                folder_name = base_folder[2:]  # ./ 제거
                            elif base_folder.startswith('.'):
                                folder_name = base_folder[1:]  # . 제거
                            else:
                                folder_name = base_folder
                            
                            # 절대경로 생성
                            absolute_path = f"/home/work/datasets/{folder_name}/{subfolder_name}/{quality}/{crop_folder}/{subdir}/{img_name}"
                            
                            # 라벨 결정 (bad는 0, good은 1)
                            # quality가 'bad'면 0, 'good'이면 1
                            label = 1 if quality == 'bad' else 0
                        
                            # 원본 이미지명 추출 (복잡한 크롭 이미지명에서 원본 추출)
                            original_img_name = img_name
                            
                            # 복잡한 크롭 이미지명 처리 (예: bad_0177_1_11_3af317b5-9b79-4c62-be99-baff9b7ae4a0_0_0_bright.png)
                            if '_' in img_name:
                                parts = img_name.split('_')
                                # UUID 부분을 찾아서 제거 (8-4-4-4-12 형태)
                                uuid_start = -1
                                for i, part in enumerate(parts):
                                    if len(part) == 8 and i + 4 < len(parts):
                                        # UUID 패턴 확인 (8-4-4-4-12)
                                        if (len(parts[i+1]) == 4 and len(parts[i+2]) == 4 and 
                                            len(parts[i+3]) == 4 and len(parts[i+4]) == 12):
                                            uuid_start = i
                                            break
                                
                                if uuid_start != -1:
                                    # UUID까지 포함한 전체 이름 사용 (quality + ID + UUID)
                                    base_parts = parts[:uuid_start+5]  # UUID까지 포함
                                    if len(base_parts) >= 5:
                                        # quality + ID + UUID 조합 (예: bad_0177_1_11_3af317b5-9b79-4c62-be99-baff9b7ae4a0)
                                        original_img_name = f"{base_parts[0]}_{base_parts[1]}_{base_parts[2]}_{base_parts[3]}_{base_parts[4]}.jpg"
                                else:
                                    # UUID가 없는 경우, 마지막 증강법 제거
                                    if len(parts) >= 2:
                                        # 마지막 부분이 증강법인지 확인
                                        aug_types = ['bright', 'contrast', 'flip', 'gray', 'noise', 'rot']
                                        if parts[-1].split('.')[0] in aug_types:
                                            # 증강법 제거
                                            base_parts = parts[:-1]
                                            if len(base_parts) >= 2:
                                                original_img_name = f"{base_parts[0]}_{base_parts[1]}.jpg"
                            
                            # 디버깅: 원본 이미지명 추출 확인
                            if img_name != original_img_name:
                                print(f"      크롭 이미지: {img_name} -> 원본 이미지: {original_img_name}")
                            
                            # suffix 제거 후 original_image_id 추출
                            # 증강 이미지에서도 extract_image_id 함수를 사용하여 일관성 보장
                            # UUID 패턴을 찾아서 그 이후의 suffix 제거
                            uuid_pattern = r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
                            match = re.search(uuid_pattern, img_name)
                            
                            if match:
                                uuid_end = match.end()
                                # UUID까지 포함한 부분만 추출 (suffix 완전 제거)
                                img_name_no_suffix = img_name[:uuid_end] + ('.jpg' if img_name.endswith('.jpg') else '.png')
                                original_image_id = extract_image_id(img_name_no_suffix)
                            else:
                                # UUID가 없는 경우 기존 방식 사용
                                original_image_id = extract_image_id(img_name)
                            
                            # 이미지 정보 저장
                            img_info = {
                                'path': absolute_path,
                                'subfolder': subfolder_name,
                                'quality': quality,
                                'crop_folder': f"{crop_folder}/{subdir}",
                                'label': label,
                                'is_augmented': '_aug' in crop_folder,
                                'original_img_name': original_img_name,
                                'original_image_id': original_image_id,  # suffix 제거된 image_id 사용
                                'base_folder': folder_name,
                                'image_id': extract_image_id(img_name) # image_id 추가
                            }
                            all_images.append(img_info)
    
    return all_images

def unified_stratified_split(yolo_images, resnet_images, ratios):
    """YOLO와 ResNet 이미지를 동일한 기준으로 분할하는 함수"""
    # 1. YOLO 이미지를 기준으로 분할 키 생성
    split_keys = set()
    for img in yolo_images:
        # 분할 키: base_folder/subfolder/quality/image_id (image_id 기반)
        key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['image_id']}"
        split_keys.add(key)
    
    print(f"\n=== 분할 키 생성 ===")
    print(f"총 분할 키 수: {len(split_keys)}개")
    
    # 2. YOLO 이미지를 하위폴더별로 그룹화
    yolo_groups = defaultdict(list)
    for img in yolo_images:
        key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}"
        yolo_groups[key].append(img)
    
    # 3. 각 그룹별로 stratified split 수행
    train_keys = set()
    val_keys = set()
    test_keys = set()
    
    for group_key, group_images in yolo_groups.items():
        # quality별로 다시 그룹화
        quality_groups = defaultdict(list)
        for img in group_images:
            quality_groups[img['quality']].append(img)
        
        # 각 quality별로 split 수행
        for quality in ['good', 'bad']:
            if quality in quality_groups:
                group_imgs = quality_groups[quality]
                random.shuffle(group_imgs)
                
                n_total = len(group_imgs)
                n_train = int(n_total * ratios[0])
                n_val = int(n_total * ratios[1])
                n_test = n_total - n_train - n_val
                
                # train 키들 추가
                for img in group_imgs[:n_train]:
                    key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['image_id']}"
                    train_keys.add(key)
                
                # val 키들 추가
                for img in group_imgs[n_train:n_train+n_val]:
                    key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['image_id']}"
                    val_keys.add(key)
                
                # test 키들 추가
                for img in group_imgs[n_train+n_val:]:
                    key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['image_id']}"
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
    
    # 4. YOLO 이미지 분할 (원본 이미지만 분할)
    yolo_train = []
    yolo_val = []
    yolo_test = []
    
    # 원본 YOLO 이미지만 분할 (증강 이미지 제외)
    yolo_original = [img for img in yolo_images if not img.get('is_augmented', False)]
    
    for img in yolo_original:
        key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['image_id']}"
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
        # 원본 폴더명 사용
        key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['original_image_id']}"
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
        key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['original_image_id']}"
        train_original_keys.add(key)
    
    print(f"train에 선택된 원본 이미지 키: {len(train_original_keys)}개")
    
    # 증강 이미지 매칭 디버깅
    print(f"\n=== train 증강 이미지 매칭 디버깅 ===")
    # train에 속한 원본 이미지에서 생성된 증강 이미지들만 필터링
    train_aug_images = []
    for aug_img in resnet_aug:
        key = f"{aug_img['base_folder']}/{aug_img['subfolder']}/{aug_img['quality']}/{aug_img['original_image_id']}"
        if key in train_original_keys:  # train에 속한 원본에서 생성된 증강 이미지만
            train_aug_images.append(aug_img)

    # 이제 train_aug_images만 사용
    for aug_img in train_aug_images[:10]:
        # 매칭 실패 확인
        key = f"{aug_img['base_folder']}/{aug_img['subfolder']}/{aug_img['quality']}/{aug_img['original_image_id']}"
        if key not in train_original_keys:
            print(f"❌ 증강 이미지 매칭 실패: {key}")
            print(f" - base_folder: {aug_img['base_folder']}")
            print(f" - subfolder: {aug_img['subfolder']}")
            print(f" - quality: {aug_img['quality']}")
            print(f" - original_image_id: {aug_img['original_image_id']}")
    
    # train에 선택된 원본 이미지와 매칭되는 증강 이미지들을 train에 추가
    matching_aug_count = 0
    for aug_img in resnet_aug:
        key = f"{aug_img['base_folder']}/{aug_img['subfolder']}/{aug_img['quality']}/{aug_img['original_image_id']}"
        if key in train_original_keys:
            resnet_train.append(aug_img)
            matching_aug_count += 1
    
    print(f"train에 추가된 증강 이미지: {matching_aug_count}개")
    
    # 6. YOLO 증강 이미지 추가 (train에만 포함)
    yolo_augmented = [img for img in yolo_images if img.get('is_augmented', False)]
    yolo_aug_matching_count = 0
    
    for aug_img in yolo_augmented:
        key = f"{aug_img['base_folder']}/{aug_img['subfolder']}/{aug_img['quality']}/{aug_img['image_id']}"
        if key in train_keys:
            yolo_train.append(aug_img)
            yolo_aug_matching_count += 1
    
    print(f"YOLO train에 추가된 증강 이미지: {yolo_aug_matching_count}개")
    
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
        
        # ResNet용 파일명
        resnet_train_file = txt_dir / f'train_resnet_{name}.txt'
        resnet_val_file = txt_dir / f'val_resnet_{name}.txt'
        resnet_test_file = txt_dir / f'test_resnet_{name}.txt'
    else:
        # YOLO용 파일명
        yolo_train_file = txt_dir / 'train.txt'
        yolo_val_file = txt_dir / 'val.txt'
        yolo_test_file = txt_dir / 'test.txt'
        
        # ResNet용 파일명
        resnet_train_file = txt_dir / 'train_resnet.txt'
        resnet_val_file = txt_dir / 'val_resnet.txt'
        resnet_test_file = txt_dir / 'test_resnet.txt'

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
    
    # ResNet용 파일 저장 (경로 라벨)
    resnet_train, resnet_val, resnet_test = resnet_splits
    
    with open(resnet_train_file, 'w') as f:
        for img in resnet_train:
            f.write(f"{img['path']} {img['label']}\n")
    
    with open(resnet_val_file, 'w') as f:
        for img in resnet_val:
            f.write(f"{img['path']} {img['label']}\n")
    
    with open(resnet_test_file, 'w') as f:
        for img in resnet_test:
            f.write(f"{img['path']} {img['label']}\n")
    
    print(f"\n=== 분할 결과 ===")
    print(f"YOLO용:")
    print(f"  train: {len(yolo_train)}개 -> {yolo_train_file}")
    print(f"  val: {len(yolo_val)}개 -> {yolo_val_file}")
    print(f"  test: {len(yolo_test)}개 -> {yolo_test_file}")
    print(f"ResNet용:")
    print(f"  train: {len(resnet_train)}개 -> {resnet_train_file}")
    print(f"  val: {len(resnet_val)}개 -> {resnet_val_file}")
    print(f"  test: {len(resnet_test)}개 -> {resnet_test_file}")

def main():
    parser = argparse.ArgumentParser(description='YOLO와 ResNet 훈련용 split을 동시에 생성합니다.')
    parser.add_argument('--folders', nargs='+', default=['0616', '0718', '0721', '0725', '0728', '0729', '0731', '0801', '0804', '0805', '0806'], 
                       help='분석할 기본 폴더 날짜들 (기본값: 0616 0718 0721 0725 0728 0729 0731 0801 0804 0805 0806)')
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
    
    # YOLO용 원본 이미지 수집
    yolo_images = collect_yolo_images_from_folders(target_folders, args.subfolders)
    
    if not yolo_images:
        print("수집된 YOLO용 이미지가 없습니다.")
        return
    
    print(f"\n총 {len(yolo_images)}개 YOLO용 이미지 수집 완료")
    
    # ResNet용 크롭 이미지 수집
    resnet_images = collect_resnet_images_from_folders(target_folders, args.subfolders)
    
    if not resnet_images:
        print("수집된 ResNet용 이미지가 없습니다.")
        return
    
    print(f"\n총 {len(resnet_images)}개 ResNet용 이미지 수집 완료")
    
    # 통합 분할 수행
    yolo_splits, resnet_splits = unified_stratified_split(yolo_images, resnet_images, SPLIT_RATIO)
    
    # 파일에 저장
    write_split_files(yolo_splits, resnet_splits, args.name)

if __name__ == '__main__':
    main() 