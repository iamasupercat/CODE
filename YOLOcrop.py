#!/usr/bin/env python3
"""
YOLO OBB(x, y, w, h, angle) 라벨을 고려하여 객체를 각도까지 반영해 크롭합니다.

참고 코드: CODE/bolt_crop_for_resnet.py
차이점:
- 라벨 포맷: class x y w h angle (정규화, angle은 라디안 가정)
- 모든 클래스에 대해 크롭 수행
- 결과 저장 경로: crop_obb/<cls>/
- debug_crop 폴더에 모든 클래스의 크롭 영역 표시된 디버그 이미지 생성

사용법:
    # 크롭 처리
    python YOLOcrop.py \
        --target_dir 0807 \
        --subfolders frontfender hood trunklid \
        --set_types bad good

    # 날짜 구간으로 선택 (예: 0715부터 0805까지, 해당 범위 폴더 자동 선택)
    python YOLOcrop.py \
        --date-range 0807 0821 \
        --subfolders frontfender hood trunklid \
        --set_types bad good

    # 폴더 삭제 (crop_obb, debug_crop 폴더들 삭제)
    python YOLOcrop.py \
        --target_dir 0807 \
        --subfolders frontfender hood trunklid \
        --set_types bad good \
        --clean
"""

import os
import argparse
import shutil
from math import cos, sin
from PIL import Image, ImageOps, ImageDraw
import math


def parse_obb_label(line):
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    cls = int(float(parts[0]))
    x = float(parts[1])
    y = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    angle = float(parts[5])  # 라디안 가정
    return cls, x, y, w, h, angle


def compute_rotated_rect_corners(cx, cy, w, h, angle):
    # 중심 기준 좌표
    dx = w / 2.0
    dy = h / 2.0
    # 네 꼭짓점 (시계 또는 반시계)
    local_pts = [
        (-dx, -dy),
        ( dx, -dy),
        ( dx,  dy),
        (-dx,  dy),
    ]
    c = cos(angle)
    s = sin(angle)
    corners = []
    for lx, ly in local_pts:
        rx = c * lx - s * ly + cx
        ry = s * lx + c * ly + cy
        corners.append((rx, ry))
    return corners


def crop_obb_from_image(img, cx, cy, w, h, angle):
    """
    절차:
    1) 회전된 사각형 꼭짓점의 AABB로 1차 크롭 (여유를 두고)
    2) 1차 패치를 -angle로 회전 (expand=True로 패치 확장)
    3) 회전 후 패치에서 회전된 사각형의 꼭짓점 위치를 다시 계산
    4) 그 꼭짓점들을 포함하는 최소 사각형으로 최종 크롭
    """
    img_w, img_h = img.size
    corners = compute_rotated_rect_corners(cx, cy, w, h, angle)
    
    # AABB 계산 (여유를 두고)
    padding = max(w, h) * 0.2  # 크기의 20% 여유
    min_x = max(0, int(min(p[0] for p in corners) - padding))
    max_x = min(img_w, int(max(p[0] for p in corners) + padding))
    min_y = max(0, int(min(p[1] for p in corners) - padding))
    max_y = min(img_h, int(max(p[1] for p in corners) + padding))

    if min_x >= max_x or min_y >= max_y:
        return None

    # 1차 패치
    patch = img.crop((min_x, min_y, max_x, max_y))
    patch_w, patch_h = patch.size
    
    # 패치 내에서의 원본 중심 좌표 (패치 좌표계)
    local_cx = cx - min_x
    local_cy = cy - min_y
    
    # 패치 내에서의 회전된 사각형 꼭짓점 (패치 좌표계)
    local_corners = []
    for corner_x, corner_y in corners:
        local_corner_x = corner_x - min_x
        local_corner_y = corner_y - min_y
        local_corners.append((local_corner_x, local_corner_y))

    # 2) -angle로 회전 (degrees로 변환)
    degrees = -angle * 180.0 / math.pi
    
    # Pillow rotate(expand=True)는 원본 이미지의 중심을 기준으로 회전하고
    # 회전 후 이미지의 중심은 원본 이미지의 중심과 같습니다.
    # 회전 후 패치의 크기는 원본 패치의 대각선 길이 이상입니다.
    
    # 회전 행렬 (원본 패치 중심 기준)
    c = cos(-angle)
    s = sin(-angle)
    
    # 원본 패치 중심
    patch_center_x = patch_w / 2.0
    patch_center_y = patch_h / 2.0
    
    # 회전 후 패치에서의 꼭짓점 계산
    rotated_corners = []
    for lx, ly in local_corners:
        # 패치 중심 기준 상대 좌표
        dx = lx - patch_center_x
        dy = ly - patch_center_y
        
        # 회전 적용
        rx = c * dx - s * dy
        ry = s * dx + c * dy
        
        # 회전 후 패치 중심 기준 절대 좌표
        # expand=True일 때 회전 후 패치의 중심은 원본 패치의 중심과 같음
        final_x = rx + patch_center_x
        final_y = ry + patch_center_y
        
        rotated_corners.append((final_x, final_y))
    
    # 회전된 패치 생성 (expand=True)
    patch = patch.rotate(degrees, resample=Image.BICUBIC, expand=True)
    
    # expand=True일 때 회전 후 패치 크기가 변하므로, 중심 위치도 조정 필요
    # Pillow는 회전 후 패치의 중심을 원본 패치의 중심과 같게 유지하려고 하지만,
    # 실제로는 패치 크기가 변하면서 중심 위치가 달라질 수 있습니다.
    # 회전 후 패치의 실제 크기
    rotated_patch_w, rotated_patch_h = patch.size
    
    # 회전 후 패치에서 원본 패치 중심의 위치
    # expand=True일 때, 원본 패치의 중심은 회전 후 패치의 중심과 같습니다
    rotated_center_x = rotated_patch_w / 2.0
    rotated_center_y = rotated_patch_h / 2.0
    
    # 회전 후 패치에서의 꼭짓점 위치 재계산 (회전 후 패치 중심 기준)
    final_corners = []
    for lx, ly in local_corners:
        # 원본 패치 중심 기준 상대 좌표
        dx = lx - patch_center_x
        dy = ly - patch_center_y
        
        # 회전 적용
        rx = c * dx - s * dy
        ry = s * dx + c * dy
        
        # 회전 후 패치 중심 기준 절대 좌표
        final_x = rx + rotated_center_x
        final_y = ry + rotated_center_y
        
        final_corners.append((final_x, final_y))
    
    # 회전 후 패치에서 꼭짓점들을 포함하는 최소 사각형으로 크롭
    if not final_corners:
        return None
    
    crop_min_x = max(0, int(min(p[0] for p in final_corners)))
    crop_max_x = min(rotated_patch_w, int(max(p[0] for p in final_corners)) + 1)
    crop_min_y = max(0, int(min(p[1] for p in final_corners)))
    crop_max_y = min(rotated_patch_h, int(max(p[1] for p in final_corners)) + 1)

    if crop_min_x >= crop_max_x or crop_min_y >= crop_max_y:
        return None

    final_crop = patch.crop((crop_min_x, crop_min_y, crop_max_x, crop_max_y))
    return final_crop


def clean_folders(base_dir):
    """crop_obb, debug_crop 폴더들을 삭제합니다."""
    folders_to_clean = ['crop_obb', 'debug_crop']
    
    for folder in folders_to_clean:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            print(f"삭제 중: {folder_path}")
            shutil.rmtree(folder_path)
        else:
            print(f"폴더가 존재하지 않음: {folder_path}")

def process_set(base_dir):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    debug_dir = os.path.join(base_dir, 'debug_crop')
    
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"경로가 올바르지 않음: {images_dir} 또는 {labels_dir}")
        return
    
    os.makedirs(debug_dir, exist_ok=True)

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        if not os.path.exists(label_path):
            continue

        try:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            img_w, img_h = img.size
        except:
            print(f"이미지 로드 실패: {img_path}")
            continue

        # 디버그용 복사본
        debug_img = img.copy()
        draw = ImageDraw.Draw(debug_img)

        # 먼저 모든 라벨을 읽어서 디버그 박스 그리기
        labels_data = []
        with open(label_path, 'r') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                parsed = parse_obb_label(line)
                if not parsed:
                    continue
                cls, x, y, w, h, angle = parsed
                labels_data.append((cls, x, y, w, h, angle, idx))
                
                # 정규화 해제
                cx = x * img_w
                cy = y * img_h
                bw = w * img_w
                bh = h * img_h
                
                # 디버그용 회전된 박스 그리기 (모든 클래스)
                corners = compute_rotated_rect_corners(cx, cy, bw, bh, angle)
                # 다각형으로 그리기
                draw.polygon(corners, outline='red', width=3)
        
        # 모든 클래스에 대해 크롭 수행
        for cls, x, y, w, h, angle, idx in labels_data:
            # 정규화 해제
            cx = x * img_w
            cy = y * img_h
            bw = w * img_w
            bh = h * img_h

            crop = crop_obb_from_image(img, cx, cy, bw, bh, angle)
            if crop is None:
                continue

            save_name = f"{os.path.splitext(img_name)[0]}_{cls}_{idx}.jpg"
            crop_dir = os.path.join(base_dir, 'crop_obb', str(cls))
            os.makedirs(crop_dir, exist_ok=True)
            save_path = os.path.join(crop_dir, save_name)
            crop.save(save_path)
        
        # 디버그 이미지 저장
        debug_save_path = os.path.join(debug_dir, img_name)
        debug_img.save(debug_save_path)


def main():
    parser = argparse.ArgumentParser(description='YOLO OBB 라벨을 고려하여 이미지에서 객체를 크롭합니다.')
    parser.add_argument('--target_dir', nargs='*',
                        help='대상 폴더 날짜들 (예: 0718 0721)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                        help='날짜 구간 선택 (MMDD 또는 YYYYMMDD). 예: --date-range 0715 0805')
    parser.add_argument('--subfolders', nargs='+', required=True,
                        help='처리할 서브폴더들 (예: frontfender hood trunklid)')
    parser.add_argument('--set_types', nargs='+', default=['bad', 'good'],
                        help='처리할 set 타입들 (기본값: bad good)')
    parser.add_argument('--clean', action='store_true',
                        help='crop_obb, debug_crop 폴더들을 삭제합니다')
    args = parser.parse_args()

    base_path = "/home/work/datasets"
    def collect_date_range_folders(base_path: str, start: str, end: str):
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

    if args.date_range:
        start, end = args.date_range
        target_dirs = collect_date_range_folders(base_path, start, end)
        print(f"날짜 구간: {start} ~ {end}")
    else:
        if not args.target_dir:
            parser.error("--target_dir 또는 --date-range 중 하나는 반드시 지정해야 합니다.")
        target_dirs = [os.path.join(base_path, date) for date in args.target_dir]

    display_dates = [os.path.basename(p) for p in target_dirs]
    print(f"대상 폴더들: {display_dates}")
    print(f"절대경로: {target_dirs}")
    print(f"처리할 서브폴더들: {args.subfolders}")
    print(f"처리할 set 타입들: {args.set_types}")

    if args.clean:
        print("\n=== 폴더 삭제 모드 ===")
        for target_dir in target_dirs:
            print(f"\n삭제 중인 대상 폴더: {target_dir}")
            for part in args.subfolders:
                for set_type in args.set_types:
                    base_dir = os.path.join(target_dir, part, set_type)
                    if os.path.exists(base_dir):
                        print(f"삭제 처리 중: {base_dir}")
                        clean_folders(base_dir)
                    else:
                        print(f"경로가 존재하지 않음: {base_dir}")
    else:
        print("\n=== 크롭 진행 모드 ===")
        for target_dir in target_dirs:
            print(f"\n처리 중인 대상 폴더: {target_dir}")
            for part in args.subfolders:
                for set_type in args.set_types:
                    base_dir = os.path.join(target_dir, part, set_type)
                    if os.path.exists(base_dir):
                        print(f"처리 중: {base_dir}")
                        process_set(base_dir)
                    else:
                        print(f"경로가 존재하지 않음: {base_dir}")


if __name__ == "__main__":
    main()


