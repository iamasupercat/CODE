#!/usr/bin/env python3
"""
크롭된 앞도어 이미지에 데이터 증강을 적용하는 스크립트



# 이 밖의 자세한 사용법은 USAGE.md 파일을 참조하세요.
사용법:
    python AugforDoor_crop.py \
        --date-range 0807 1103 \
        --subfolders frontdoor \
        --set_types bad
    
    python AugforDoor_crop.py \
    --date-range 0807 1109 \
    --obb-date-range 0616 0806 \
    --subfolders frontdoor \
    --set_types good bad




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

증강 기법:
    - rot: 랜덤 회전 (-10° ~ +10°)
    - flip: 수평 뒤집기
    - noise: 가우시안 노이즈 추가
    - gray: 그레이스케일 변환
    - bright: 밝기 증가 (+40)
    - contrast: 대비 증가 (1.5배)

출력:
    - crop_high_aug/: 증강된 결함 객체들 (원본 1개 + 증강 6개 = 총 7개)
    - crop_mid_aug/: 증강된 정상 객체들 (원본 1개 + 증강 6개 = 총 7개)
    - crop_low_aug/: 증강된 정상 객체들 (원본 1개 + 증강 6개 = 총 7개)
    - 파일명 형식: 원본파일명_라벨번호_증강기법.확장자 (예: image_0_rot.jpg, image_1_flip.jpg)
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
    angle = np.random.uniform(-10, 10)
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

def invert_color(img):
    """색상 반전 (Invert)"""
    return cv2.bitwise_not(img)

# 증강 기법과 파일명 접미사 매핑
audict = {
    'flip': horizontal_flip,
    'noise': add_noise,
    'invert': invert_color
}

# bad/good, crop_* 폴더 순회
def process_all(target_dirs, subfolders, set_types):
    for target_dir in target_dirs:
        target_dir = os.path.abspath(target_dir)
        print(f"\n처리 중인 대상 폴더: {target_dir}")
        
        for subfolder in subfolders:
            for set_type in set_types:
                # crop_high, crop_mid, crop_low 폴더 처리
                crop_areas = ['crop_high', 'crop_mid', 'crop_low']
                
                for crop_area in crop_areas:
                    src_base_dir = os.path.join(target_dir, subfolder, set_type, crop_area)
                    dst_base_dir = os.path.join(target_dir, subfolder, set_type, crop_area + '_aug')
                    
                    if not os.path.exists(src_base_dir):
                        print(f"소스 디렉토리가 존재하지 않음: {src_base_dir}")
                        continue
                    
                    # 0, 1, 2, 3 서브폴더 처리 (실링 클래스별)
                    for subdir in ['0', '1', '2', '3']:
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
                            
                            # 파일명에서 원본 이미지명과 라벨 번호 추출
                            fname = os.path.basename(img_path)
                            fname_without_ext = os.path.splitext(fname)[0]
                            
                            # 파일명이 "원본이미지명_라벨번호.jpg" 형태인지 확인
                            if '_' in fname_without_ext:
                                parts = fname_without_ext.split('_')
                                if len(parts) >= 2 and parts[-1].isdigit():
                                    # 마지막 부분이 라벨 번호인 경우
                                    original_name = '_'.join(parts[:-1])  # 라벨 번호 제외
                                    label_num = parts[-1]
                                else:
                                    # 라벨 번호가 없는 경우
                                    original_name = fname_without_ext
                                    label_num = subdir  # 폴더 번호를 라벨 번호로 사용
                            else:
                                # 언더스코어가 없는 경우
                                original_name = fname_without_ext
                                label_num = subdir  # 폴더 번호를 라벨 번호로 사용
                            
                            for key, func in audict.items():
                                aug_img = func(img)
                                # 파일명 형식: 원본이미지명_라벨번호_증강기법.확장자
                                if key == 'gray':
                                    out_path = os.path.join(dst_dir, f'{original_name}_{label_num}_{key}.png')
                                    result = cv2.imwrite(out_path, aug_img)
                                else:
                                    out_path = os.path.join(dst_dir, f'{original_name}_{label_num}_{key}.jpg')
                                    result = cv2.imwrite(out_path, aug_img)
                                print(f'Saving: {out_path} - {"Success" if result else "Failed"}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='크롭된 이미지에 데이터 증강을 적용합니다.')
    parser.add_argument('--target_dir', nargs='+',
                       help='대상 폴더 날짜들 (예: 0616 0721 0728)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                       help='날짜 구간 선택 (MMDD). 예: --date-range 0807 1103')
    parser.add_argument('--obb-folders', nargs='+',
                       help='OBB 폴더 날짜들 (예: 0718 0806)')
    parser.add_argument('--obb-date-range', nargs=2, metavar=('START', 'END'),
                       help='OBB 폴더 날짜 구간 선택 (MMDD). 예: --obb-date-range 0718 0806')
    parser.add_argument('--subfolders', nargs='+',
                       default=['frontdoor'],
                       help='처리할 서브폴더들 (기본값: frontdoor)')
    parser.add_argument('--set_types', nargs='+',
                       default=['bad', 'good'],
                       help='처리할 set 타입들 (기본값: bad good)')
    
    args = parser.parse_args()
    
    # 날짜를 절대경로로 변환
    base_path = "/home/work/datasets"
    obb_base_path = os.path.join(base_path, "OBB")

    # 일반 폴더
    if args.date_range:
        start, end = args.date_range
        target_dirs = collect_date_range_folders(base_path, start, end)
        print(f"일반 폴더 날짜 구간: {start} ~ {end}")
    elif args.target_dir:
        target_dirs = [os.path.join(base_path, date) for date in args.target_dir]
    else:
        target_dirs = []

    # OBB 폴더
    obb_dirs = []
    if args.obb_date_range:
        start, end = args.obb_date_range
        obb_dirs = collect_date_range_folders(obb_base_path, start, end)
        print(f"OBB 폴더 날짜 구간: {start} ~ {end}")
    elif args.obb-folders:
        obb_dirs = [os.path.join(obb_base_path, date) for date in args.obb_folders]

    target_dirs = target_dirs + obb_dirs
    if not target_dirs:
        parser.error("--target_dir/--date-range 또는 --obb-folders/--obb-date-range 중 하나는 반드시 지정해야 합니다.")
    
    display_dates = [os.path.basename(p) for p in target_dirs]
    print(f"대상 폴더들: {display_dates}")
    print(f"절대경로: {target_dirs}")
    print(f"처리할 서브폴더들: {args.subfolders}")
    print(f"처리할 set 타입들: {args.set_types}")
    
    process_all(target_dirs, args.subfolders, args.set_types)
