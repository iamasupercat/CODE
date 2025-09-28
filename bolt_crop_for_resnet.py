#!/usr/bin/env python3
"""
라벨 번호 0,1,2번에 대해서만 크롭
이거 돌리면 덮어씌우기 됨 (크롭 폴더 굳이 삭제 안해도 됨)
근데 만일 cls번호가 달라지면 다른 크롭이라 생각하여 덮어씌워지기가 안됨 -> 크롭 폴더 삭제 후 크롭 진행


사용법:
    # 크롭 처리
    python bolt_crop_for_resnet.py \
        --target_dir 0616 0718 0721 0725 0728 0729 0731 0801 0804 0805 0806 \
        --subfolders frontfender hood trunklid \
        --set_types bad good
    
    # 폴더 삭제 (crop_bolt, crop_bolt_aug, debug_crop 폴더들 삭제)
    python bolt_crop_for_resnet.py \
        --target_dir 0616 0718 0721 0725 0728 0729 0731 0801 0804 0805 0806 \
        --subfolders frontfender hood trunklid \
        --set_types bad good \
        --clean


옵션 설명:
    --target_dir: 대상 폴더 경로들 (기본값: 현재 디렉토리)
    --subfolders: 처리할 서브폴더들 (기본값: hood_revised trunk_revised front_revised)
    --set_types: 처리할 set 타입들 (기본값: bad good)

폴더 구조:
    target_dir/
    ├── subfolder/
    │   ├── bad/
    │   │   ├── images/          # 원본 이미지들
    │   │   ├── labels/          # YOLO 라벨 파일들
    │   │   ├── crop_bolt/        
    │   │   │  ├──0/       # 정측면
    │   │   │  ├──1/       # 정면
    │   │   │  └──2/       # 측면
    │   │   └── debug_crop/     
    │   └── good/
    │       ├── images/          # 원본 이미지들
    │       ├── labels/          # YOLO 라벨 파일들
    │       ├── crop_bolt/        
    │       │  ├──0/       # 정측면
    │       │  ├──1/       # 정면
    │       │  └──2/       # 측면
    │       └── debug_crop/     


"""

import os
import argparse
import shutil
from PIL import Image, ImageDraw, ImageOps

def yolo_to_bbox(yolo_line, img_width, img_height, img_name=None, line_num=None):
    parts = yolo_line.strip().split()
    cls = int(parts[0])
    vals = list(map(float, parts[1:]))
    # 값이 0~1 범위가 아니면 경고
    for i, v in enumerate(vals):
        if not (0.0 <= v <= 1.0):
            print(f"[경고] {img_name or ''} 라벨 {line_num or ''} 값 {v}가 0~1 범위가 아님: {yolo_line.strip()}")
    x_center = vals[0] * img_width
    y_center = vals[1] * img_height
    w = vals[2] * img_width
    h = vals[3] * img_height
    x1 = int(round(x_center - w / 2))
    y1 = int(round(y_center - h / 2))
    x2 = int(round(x_center + w / 2))
    y2 = int(round(y_center + h / 2))
    # 이미지 경계로 clipping
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    return cls, x1, y1, x2, y2

def clean_folders(set_type):
    """crop_bolt, crop_bolt_aug, debug_crop 폴더들을 삭제합니다."""
    base_dir = set_type
    folders_to_clean = ['crop_bolt', 'crop_bolt_aug', 'debug_crop']
    
    for folder in folders_to_clean:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            print(f"삭제 중: {folder_path}")
            shutil.rmtree(folder_path)
        else:
            print(f"폴더가 존재하지 않음: {folder_path}")

def process_set(set_type):
    base_dir = set_type
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    debug_dir = os.path.join(base_dir, 'debug_crop')
    os.makedirs(debug_dir, exist_ok=True)
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        if not os.path.exists(label_path):
            continue
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # EXIF 회전 보정 추가
        img_width, img_height = img.size
        # 디버그용 복사본
        debug_img = img.copy()
        draw = ImageDraw.Draw(debug_img)
        with open(label_path, 'r') as f:
            for idx, line in enumerate(f):  # 라벨 파일의 각 라인(각 박스)마다 반복
                if not line.strip():
                    continue
                cls, x1, y1, x2, y2 = yolo_to_bbox(line, img_width, img_height, img_name, idx+1)
                # 라벨 번호 0,1,2번에 대해서만 크롭
                if cls != 0 and cls != 1 and cls != 2:
                    continue
                # 유효한 crop 영역만 저장
                if x2 > x1 and y2 > y1:
                    crop = img.crop((x1, y1, x2, y2))
                    save_name = f"{os.path.splitext(img_name)[0]}_{cls}_{idx}.jpg"
                    # 0,1,2 폴더 생성
                    crop_dir = os.path.join(base_dir, 'crop_bolt', str(cls))
                    os.makedirs(crop_dir, exist_ok=True)
                    save_path = os.path.join(crop_dir, save_name)
                    crop.save(save_path)
                    # 디버그용 박스 그리기
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        # 디버그 이미지 저장
        debug_save_path = os.path.join(debug_dir, img_name)
        debug_img.save(debug_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO 라벨을 사용하여 이미지에서 객체를 크롭합니다.')
    parser.add_argument('--target_dir', nargs='+', default=['0616', '0718', '0721', '0725', '0728', '0729', '0731', '0801', '0804', '0805', '0806'], 
                       help='대상 폴더 날짜들 (기본값: 0616 0718 0721 0725 0728 0729 0731 0801 0804 0805 0806)')
    parser.add_argument('--subfolders', nargs='+', 
                       default=['hood_revised', 'trunk_revised', 'front_revised'],
                       help='처리할 서브폴더들 (기본값: hood_revised trunk_revised front_revised)')
    parser.add_argument('--set_types', nargs='+', 
                       default=['bad', 'good'],
                       help='처리할 set 타입들 (기본값: bad good)')
    parser.add_argument('--clean', action='store_true',
                       help='crop_bolt, crop_bolt_aug, debug_crop 폴더들을 삭제합니다')
    
    args = parser.parse_args()
    
    # 날짜를 절대경로로 변환
    base_path = "/home/work/datasets"
    target_dirs = [os.path.join(base_path, date) for date in args.target_dir]
    
    print(f"대상 폴더들: {args.target_dir}")
    print(f"절대경로: {target_dirs}")
    print(f"처리할 서브폴더들: {args.subfolders}")
    print(f"처리할 set 타입들: {args.set_types}")
    
    if args.clean:
        print("\n=== 폴더 삭제 모드 ===")
        for target_dir in target_dirs:
            print(f"\n삭제 중인 대상 폴더: {target_dir}")
            
            for part in args.subfolders:
                for set_type in args.set_types:
                    base_path = os.path.join(target_dir, part, set_type)
                    if os.path.exists(base_path):
                        print(f"삭제 처리 중: {base_path}")
                        clean_folders(base_path)
                    else:
                        print(f"경로가 존재하지 않음: {base_path}")
    else:
        print("\n=== 크롭 진행 모드 ===")
        for target_dir in target_dirs:
            print(f"\n처리 중인 대상 폴더: {target_dir}")
            
            for part in args.subfolders:
                for set_type in args.set_types:
                    base_path = os.path.join(target_dir, part, set_type)
                    if os.path.exists(base_path):
                        print(f"처리 중: {base_path}")
                        process_set(base_path)
                    else:
                        print(f"경로가 존재하지 않음: {base_path}")
