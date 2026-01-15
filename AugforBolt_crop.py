#!/usr/bin/env python3
"""
크롭된 볼트 이미지에 데이터 증강을 적용하는 스크립트



# 이 밖의 자세한 사용법은 USAGE.md 파일을 참조하세요.
사용법:
    python AugforBolt_crop.py \
        --date-range 0616 1109 \
        --subfolders frontfender hood trunklid \
        --set_types bad good
    
    python AugforBolt_crop.py \
        --obb-date-range 0616 0806 \
        --subfolders frontfender hood trunklid \
        --set_types bad good


    


[로직 설명]
6가지 증강 기법을 적용하여 각 이미지당 6개의 증강된 이미지를 생성합니다.
(rot, flip, noise, gray, bright, contrast)
rot: 랜덤 회전 (-180° ~ +180°)
noise: mean 0, std 10
bright: +40 (hsv의 v채널에 +40)
contrast: 1.5배 (hsv의 s채널에 1.5배)


폴더 구조:
    target_dir/
    ├── subfolder/
    │   ├── bad/
    │   │   ├── images/          # 원본 이미지들
    │   │   ├── labels/          # YOLO 라벨 파일들
    │   │   ├── crop_bolt/        
    │   │   │  ├──0/       # 정측면
    │   │   │  └──1/       # 측면
    │   │   ├── crop_bolt_aug/        
    │   │   │  ├──0/       # 정측면_aug
    │   │   │  └──1/       # 측면_aug
    │   │   └── debug_crop/      
    │   └── good/
    │       ├── images/          # 원본 이미지들
    │       ├── labels/          # YOLO 라벨 파일들
    │       ├── crop_bolt/        
    │       │  ├──0/       # 정측면
    │       │  └──1/       # 측면
    │       ├── crop_bolt_aug/        
    │       │  ├──0/       # 정측면_aug
    │       │  └──1/       # 측면_aug
    │       └── debug_crop/  

증강 기법:
    - rot: 랜덤 회전 (-180° ~ +180°)
    - flip: 수평 뒤집기
    - noise: 가우시안 노이즈 추가
    - gray: 그레이스케일 변환
    - bright: 밝기 증가 (+40)
    - contrast: 대비 증가 (1.5배)

출력:
    - crop_bad_aug/: 증강된 결함 객체들 (원본 1개 + 증강 6개 = 총 7개)
    - crop_good_aug/: 증강된 정상 객체들 (원본 1개 + 증강 6개 = 총 7개)
    - 파일명 형식: 원본파일명_증강기법.확장자 (예: image1_rot.jpg, image1_flip.jpg)
"""

import os
import cv2
import numpy as np
import argparse
from glob import glob

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

# 증강 함수들
def small_rotation(img):
    angle = np.random.uniform(-180, 180)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def horizontal_flip(img):
    return cv2.flip(img, 1)

def add_noise(img):
    row, col, ch = img.shape
    mean = 0
    sigma = 10
    gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch)
    noisy = img + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def increase_brightness(img, value=40):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_bright

def increase_contrast(img, factor=1.5):
    """
    이미지의 대비를 factor 배로 증가시킵니다.
    factor=1.0이면 변화 없음, 1.5면 50% 증가
    """
    img = img.astype(np.float32)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = (img - mean) * factor + mean
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# 증강 기법과 파일명 접미사 매핑
audict = {
    'rot': small_rotation,
    'flip': horizontal_flip,
    'noise': add_noise,
    'gray': to_grayscale,
    'bright': increase_brightness,
    'contrast': increase_contrast
}

# bad/good, crop_* 폴더 순회
def process_all(target_dirs, subfolders, set_types):
    for target_dir in target_dirs:
        target_dir = os.path.abspath(target_dir)
        print(f"\n처리 중인 대상 폴더: {target_dir}")
        
        for subfolder in subfolders:
            for set_type in set_types:
                # crop_bolt 폴더 내의 0, 1, 2 서브폴더 처리
                src_base_dir = os.path.join(target_dir, subfolder, set_type, 'crop_bolt')
                dst_base_dir = os.path.join(target_dir, subfolder, set_type, 'crop_bolt_aug')
                
                if not os.path.exists(src_base_dir):
                    print(f"소스 디렉토리가 존재하지 않음: {src_base_dir}")
                    continue
                
                # 0,1 서브폴더 처리 (정측면, 측면)
                for subdir in ['0', '1']:
                    src_dir = os.path.join(src_base_dir, subdir)
                    dst_dir = os.path.join(dst_base_dir, subdir)
                    
                    if not os.path.exists(src_dir):
                        print(f"소스 디렉토리가 존재하지 않음: {src_dir}")
                        continue
                        
                    os.makedirs(dst_dir, exist_ok=True)
                    img_paths = glob(os.path.join(src_dir, '*.jpg'))
                    
                    if not img_paths:
                        print(f"이미지 파일이 없음: {src_dir}")
                        continue
                        
                    print(f"처리 중: {src_dir} -> {dst_dir} ({len(img_paths)}개 파일)")
                    
                    for img_path in img_paths:
                        print(f'Reading: {img_path}')
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f'Failed to read: {img_path}')
                            continue
                        fname = os.path.splitext(os.path.basename(img_path))[0]
                        for key, func in audict.items():
                            aug_img = func(img)
                            if key == 'gray':
                                out_path = os.path.join(dst_dir, f'{fname}_{key}.png')
                                result = cv2.imwrite(out_path, aug_img)
                            else:
                                out_path = os.path.join(dst_dir, f'{fname}_{key}.jpg')
                                result = cv2.imwrite(out_path, aug_img)
                            print(f'Saving: {out_path} - {"Success" if result else "Failed"}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='크롭된 이미지에 데이터 증강을 적용합니다.')
    parser.add_argument('--target_dir', nargs='+', 
                       help='대상 폴더 날짜들 (예: 0616 0718 0721, 일반 폴더)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                       help='일반 폴더 날짜 구간 선택 (MMDD). 예: --date-range 0616 0806')
    parser.add_argument('--obb-folders', nargs='+',
                       help='분석할 OBB 폴더 날짜들 (예: 0718 0806)')
    parser.add_argument('--obb-date-range', nargs=2, metavar=('START', 'END'),
                       help='OBB 폴더 날짜 구간 선택 (MMDD). 예: --obb-date-range 0718 0806')
    parser.add_argument('--subfolders', nargs='+', 
                       default=['hood_revised', 'trunk_revised', 'front_revised'],
                       help='처리할 서브폴더들 (기본값: hood_revised trunk_revised front_revised)')
    parser.add_argument('--set_types', nargs='+', 
                       default=['bad', 'good'],
                       help='처리할 set 타입들 (기본값: bad good)')
    
    args = parser.parse_args()
    
    # 날짜를 절대경로로 변환
       # base_path = "/workspace/datasets"
    base_path = "/workspace/datasets"
    obb_base_path = os.path.join(base_path, "OBB")
    
    # 일반 폴더 수집
    if args.date_range:
        start, end = args.date_range
        target_dirs = collect_date_range_folders(base_path, start, end)
        print(f"일반 폴더 날짜 구간: {start} ~ {end}")
    elif args.target_dir:
        target_dirs = [os.path.join(base_path, date) for date in args.target_dir]
    else:
        target_dirs = []
    
    # OBB 폴더 수집
    obb_dirs = []
    if args.obb_date_range:
        start, end = args.obb_date_range
        obb_dirs = collect_date_range_folders(obb_base_path, start, end)
        print(f"OBB 폴더 날짜 구간: {start} ~ {end}")
    elif args.obb_folders:
        obb_dirs = [os.path.join(obb_base_path, date) for date in args.obb_folders]
    
    # 최종 대상 폴더 (일반 + OBB)
    target_dirs = target_dirs + obb_dirs
    if not target_dirs:
        parser.error("--target_dir/--date-range 또는 --obb-folders/--obb-date-range 중 하나는 반드시 지정해야 합니다.")
    
    display_dates = [os.path.basename(p) for p in target_dirs if p.startswith(base_path) and not p.startswith(obb_base_path)]
    obb_display_dates = [os.path.basename(p) for p in target_dirs if p.startswith(obb_base_path)]
    
    print(f"일반 폴더들: {display_dates if display_dates else '없음'}")
    if obb_display_dates:
        print(f"OBB 폴더들: {obb_display_dates}")
    print(f"처리할 서브폴더들: {args.subfolders}")
    print(f"처리할 set 타입들: {args.set_types}")
    
    process_all(target_dirs, args.subfolders, args.set_types)
