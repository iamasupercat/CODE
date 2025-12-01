#!/usr/bin/env python3
"""
YOLO OBB(Oriented Bounding Box) 라벨을 이용하여 회전된 객체를 crop하는 스크립트

[로직 설명]
- correct_orientation_constrained: 
  사용자가 지정한 로직 사용 (w, h 길이를 기준으로 가로/세로 판단 후 방향 보정)
- clean 옵션: 
  실행 시 지정된 폴더 내의 기존 결과물(crop_*, debug_crop) 삭제

사용법:
    # 단일 날짜 지정
    python CropforOBB.py \
        --target_dir 0805 \
        --mode door \
        --clean

    # 날짜 범위 지정 (일반 폴더)
    python CropforOBB.py \
        --date-range 0807 1103 \
        --mode door

    # 일반 폴더 + OBB 폴더 (별도 날짜 범위)
    python CropforOBB.py \
        --date-range 0807 1109 \
        --obb-date-range 0616 0806 \
        --mode door \
        --clean
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from math import cos, sin
from pathlib import Path
import math
import shutil
import glob


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


def parse_obb_label(line):
    """OBB 라벨 파싱: class x y w h a"""
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    cls = int(float(parts[0]))
    x = float(parts[1])
    y = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    angle = float(parts[5])
    return cls, x, y, w, h, angle


def compute_rotated_box_corners(cx, cy, w, h, angle):
    """회전된 박스의 4개 모서리 좌표 계산"""
    dx = w / 2.0
    dy = h / 2.0
    
    local_corners = [
        (-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)
    ]
    
    c = cos(angle)
    s = sin(angle)
    
    corners = []
    for lx, ly in local_corners:
        rx = c * lx - s * ly + cx
        ry = s * lx + c * ly + cy
        corners.append((rx, ry))
    
    return corners


def correct_orientation_constrained(w, h, angle):
    """
    [사용자 지정 로직] 형상 적응형 보정 (Shape-Adaptive)
    조건: 객체는 원래 방향(가로/세로)에서 +-45도 이내로만 기울어짐.
    목표: 
      1. w >= h (가로 객체) -> 각도를 오른쪽(-45~+45도)으로 맞춤
      2. h > w (세로 객체) -> 각도를 위쪽(-135~-45도)으로 맞춤
    """
    pi = math.pi
    
    # 1. 각도 1차 정규화 (-pi ~ +pi)
    angle = (angle + pi) % (2 * pi) - pi
    
    # 2. 객체 형태에 따른 방향 보정
    if w >= h:
        # [Case A] 가로가 긴 객체 (Horizontal)
        # 목표: 각도가 0도(오른쪽) 근처여야 함.
        # 만약 각도가 절대값 90도(pi/2)를 넘어가면 '왼쪽'을 보고 있다는 뜻이므로 뒤집음.
        if abs(angle) > pi / 2:
            angle -= pi  # 180도 회전
            
    else:
        # [Case B] 세로가 긴 객체 (Vertical)
        # 목표: 각도가 -90도(위쪽) 근처여야 함. (OpenCV 좌표계: -90도가 12시 방향)
        
        # 간단하게: "Y축 아래(양수 각도)"를 보고 있으면 무조건 위로 올림
        if angle > 0:  
            angle -= pi
        
        # -180도 근처(-pi)인 경우도 아래쪽(6시)에 가까우므로 위로 보냄
        # (단, +-45도 제한 조건 때문에 이 케이스는 드물겠지만 안전장치)
        if angle < -pi + (pi/4): # -135도보다 더 작으면 (예: -170도)
             angle += pi

    # 최종 각도 재정규화
    angle = (angle + pi) % (2 * pi) - pi
            
    return w, h, angle


def crop_rotated_object(img, cx, cy, w, h, angle):
    img_h, img_w = img.shape[:2]

    # [수정됨] 사용자가 요청한 로직 (w, h 길이 기준 판단)
    w, h, angle = correct_orientation_constrained(w, h, angle)
    
    # 각도가 0에 매우 가까우면 일반 crop
    if abs(angle) < 1e-6:
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(img_w, int(cx + w / 2))
        y2 = min(img_h, int(cy + h / 2))
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        crop = img[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
        return crop_resized
    
    src_corners = compute_rotated_box_corners(cx, cy, w, h, angle)
    src_points = np.array(src_corners, dtype=np.float32)
    
    dst_corners = [
        (0, 0), (w, 0), (w, h), (0, h)
    ]
    dst_points = np.array(dst_corners, dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    warped = cv2.warpPerspective(img, M, (int(w), int(h)), 
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
    return warped


def load_excel_data(excel_path):
    if not os.path.exists(excel_path):
        return None
    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        print(f"엑셀 파일 로드 실패: {excel_path}, 오류: {e}")
        return None


def _normalize_image_name(name: str) -> str:
    """이미지 파일명에서 bad_/good_ 접두어와 확장자를 제거한 기본 이름을 반환"""
    base = os.path.splitext(name)[0]
    if base.startswith('bad_'):
        base = base[len('bad_'):]
    elif base.startswith('good_'):
        base = base[len('good_'):]
    return base


def get_defect_type_from_excel(df, image_filename):
    """
    엑셀에서 이미지파일명으로 직접 매칭하여 결함 타입(high/mid/low)을 가져옵니다.
    같은 차량번호에 여러 도어(좌/우)가 있을 수 있으므로, 차량번호가 아니라
    '이미지파일명' 열만 사용해서 한 행만 선택합니다.
    
    반환:
        (defect_dict 또는 None, needs_review: bool)
    """
    if df is None:
        return None, False
    
    img_key = _normalize_image_name(image_filename)
    df_keys = df['이미지파일명'].astype(str).apply(_normalize_image_name)
    matching_rows = df[
        (df_keys == img_key) &
        (df['상단'].notna() | df['중간'].notna() | df['하단'].notna())
    ]
    if len(matching_rows) == 0:
        return None, False
    
    needs_review = len(matching_rows) > 1
    row = matching_rows.iloc[0]
    high_val = row['상단']
    mid_val = row['중간']
    low_val = row['하단']
    
    result = {}
    if pd.notna(high_val): result['high'] = int(high_val) - 1
    if pd.notna(mid_val): result['mid'] = int(mid_val) - 1
    if pd.notna(low_val): result['low'] = int(low_val) - 1
    return (result if result else None), needs_review


def clean_directory(target_dir):
    """
    지정된 날짜 폴더 내의 모든 파트(frontdoor, hood 등)에서
    crop_* 및 debug_crop 폴더를 삭제합니다.
    """
    parts = ['frontdoor', 'frontfender', 'hood', 'trunklid']
    sub_dirs = ['bad', 'good']
    
    print(f"🧹 [{os.path.basename(target_dir)}] 청소(삭제) 시작...")
    
    cleaned_count = 0
    
    for part in parts:
        for sub in sub_dirs:
            base_path = os.path.join(target_dir, part, sub)
            if not os.path.exists(base_path):
                continue
            
            # 1. debug_crop 삭제
            debug_path = os.path.join(base_path, 'debug_crop')
            if os.path.exists(debug_path):
                try:
                    shutil.rmtree(debug_path)
                    print(f"  - 삭제됨: {debug_path}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"  ! 삭제 실패: {debug_path} ({e})")

            # 2. crop_* 삭제 (glob 사용)
            crop_pattern = os.path.join(base_path, 'crop_*')
            for crop_dir in glob.glob(crop_pattern):
                if os.path.isdir(crop_dir):
                    try:
                        shutil.rmtree(crop_dir)
                        print(f"  - 삭제됨: {crop_dir}")
                        cleaned_count += 1
                    except Exception as e:
                        print(f"  ! 삭제 실패: {crop_dir} ({e})")
                        
    if cleaned_count == 0:
        print("  (삭제할 폴더가 없습니다)")
    print("--------------------------------------------------")


def process_door_mode(base_dir, excel_path=None):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    debug_dir = os.path.join(base_dir, 'debug_crop')
    
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print(f"경로가 올바르지 않음: {images_dir} 또는 {labels_dir}")
        return
    
    os.makedirs(debug_dir, exist_ok=True)
    for cls_name in ['high', 'mid', 'low']:
        crop_dir = os.path.join(base_dir, f'crop_{cls_name}')
        for i in range(4):
            os.makedirs(os.path.join(crop_dir, str(i)), exist_ok=True)
    
    df = None
    is_bad = 'bad' in base_dir
    if is_bad and excel_path and os.path.exists(excel_path):
        df = load_excel_data(excel_path)
    
    review_needed = []
    cls_names = {0: 'high', 1: 'mid', 2: 'low'}
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        
        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        if not os.path.exists(label_path): continue
        
        try:
            img = cv2.imread(img_path)
            if img is None: continue
            img_h, img_w = img.shape[:2]
            debug_img = img.copy()
            
            labels_data = []
            with open(label_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    parsed = parse_obb_label(line)
                    if parsed is None: continue
                    cls, x, y, w, h, angle = parsed
                    if cls not in [0, 1, 2]: continue
                    labels_data.append((cls, x, y, w, h, angle))
            
            defect_types = None
            if is_bad and df is not None:
                defect_types, needs_review = get_defect_type_from_excel(df, img_name)
                if needs_review:
                    review_needed.append(img_name)
            
            for cls, x, y, w, h, angle in labels_data:
                cx, cy = x * img_w, y * img_h
                bw, bh = w * img_w, h * img_h
                
                crop = crop_rotated_object(img, cx, cy, bw, bh, angle)
                if crop is None: continue
                
                cls_name = cls_names[cls]
                folder_num = 0
                if is_bad and defect_types and cls_name in defect_types:
                    folder_num = defect_types[cls_name]
                
                crop_filename = f"{os.path.splitext(img_name)[0]}_{cls}.jpg"
                crop_path = os.path.join(base_dir, f'crop_{cls_name}', str(folder_num), crop_filename)
                cv2.imwrite(crop_path, crop)
                
                corners = compute_rotated_box_corners(cx, cy, bw, bh, angle)
                corners_int = np.array(corners, dtype=np.int32)
                cv2.polylines(debug_img, [corners_int], isClosed=True, color=(0, 0, 255), thickness=2)
            
            cv2.imwrite(os.path.join(debug_dir, img_name), debug_img)
            
        except Exception as e:
            print(f"오류: {img_path}, {e}")
            continue

    if review_needed:
        print(f"검수 필요: {len(review_needed)}개")


def process_bolt_mode(base_dir):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    debug_dir = os.path.join(base_dir, 'debug_crop')
    
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir): return
    
    os.makedirs(debug_dir, exist_ok=True)
    crop_bolt_dir = os.path.join(base_dir, 'crop_bolt')
    for i in range(2): os.makedirs(os.path.join(crop_bolt_dir, str(i)), exist_ok=True)
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        if not os.path.exists(label_path): continue
        
        try:
            img = cv2.imread(img_path)
            if img is None: continue
            img_h, img_w = img.shape[:2]
            debug_img = img.copy()
            
            labels_data = []
            cls_counters = {0: 0, 1: 0}
            
            with open(label_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    parsed = parse_obb_label(line)
                    if parsed is None: continue
                    cls, x, y, w, h, angle = parsed
                    if cls not in [0, 1]: continue
                    labels_data.append((cls, x, y, w, h, angle))
            
            for cls, x, y, w, h, angle in labels_data:
                cx, cy = x * img_w, y * img_h
                bw, bh = w * img_w, h * img_h
                
                crop = crop_rotated_object(img, cx, cy, bw, bh, angle)
                if crop is None: continue
                
                idx = cls_counters[cls]
                cls_counters[cls] += 1
                
                crop_filename = f"{os.path.splitext(img_name)[0]}_{cls}_{idx}.jpg"
                crop_path = os.path.join(crop_bolt_dir, str(cls), crop_filename)
                cv2.imwrite(crop_path, crop)
                
                corners = compute_rotated_box_corners(cx, cy, bw, bh, angle)
                corners_int = np.array(corners, dtype=np.int32)
                cv2.polylines(debug_img, [corners_int], isClosed=True, color=(0, 0, 255), thickness=2)
            
            cv2.imwrite(os.path.join(debug_dir, img_name), debug_img)
            
        except Exception as e:
            print(f"오류: {img_path}, {e}")
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', nargs='*',
                        help='일반 폴더 날짜들 (예: 0616 0718 0721)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                        help='일반 폴더 날짜 구간 (MMDD 또는 YYYYMMDD)')
    parser.add_argument('--obb-folders', nargs='*',
                        help='OBB 폴더 날짜들 (예: 0718 0806)')
    parser.add_argument('--obb-date-range', nargs=2, metavar=('START', 'END'),
                        help='OBB 폴더 날짜 구간 (MMDD 또는 YYYYMMDD)')
    parser.add_argument('--mode', choices=['door', 'bolt'], required=True)
    parser.add_argument('--clean', action='store_true', help='실행 전 기존 crop, debug 폴더 삭제')
    args = parser.parse_args()
    
    base_path = "/home/work/datasets"
    obb_base_path = os.path.join(base_path, "OBB")
    
    # 일반 폴더 수집
    if args.date_range:
        start, end = args.date_range
        target_dirs = collect_date_range_folders(base_path, start, end)
    elif args.target_dir:
        target_dirs = [os.path.join(base_path, d) for d in args.target_dir]
    else:
        target_dirs = []

    # OBB 폴더 수집
    obb_dirs = []
    if args.obb_date_range:
        start, end = args.obb_date_range
        obb_dirs = collect_date_range_folders(obb_base_path, start, end)
    elif args.obb_folders:
        obb_dirs = [os.path.join(obb_base_path, d) for d in args.obb_folders]

    # 최종 대상
    target_dirs = target_dirs + obb_dirs
    if not target_dirs:
        print("target_dir/date-range 또는 obb-folders/obb-date-range 중 하나는 필요합니다.")
        return

    print(f"대상: {[os.path.basename(p) for p in target_dirs]}")
    
    # Clean 옵션 처리
    if args.clean:
        print("\n=== [CLEAN MODE] 기존 결과물 삭제 ===")
        for target_dir in target_dirs:
            clean_directory(target_dir)
        print("=== 청소 완료, 데이터 처리를 시작합니다 ===\n")
    
    for target_dir in target_dirs:
        print(f"\n폴더: {target_dir}")
        if args.mode == 'door':
            frontdoor = os.path.join(target_dir, 'frontdoor')
            if os.path.exists(frontdoor):
                excel = os.path.join(frontdoor, 'frontdoor.xlsx')
                if os.path.exists(os.path.join(frontdoor, 'bad')): process_door_mode(os.path.join(frontdoor, 'bad'), excel)
                if os.path.exists(os.path.join(frontdoor, 'good')): process_door_mode(os.path.join(frontdoor, 'good'))
        elif args.mode == 'bolt':
            for part in ['frontfender', 'hood', 'trunklid']:
                p_dir = os.path.join(target_dir, part)
                if os.path.exists(os.path.join(p_dir, 'bad')): process_bolt_mode(os.path.join(p_dir, 'bad'))
                if os.path.exists(os.path.join(p_dir, 'good')): process_bolt_mode(os.path.join(p_dir, 'good'))

    print("\n처리 완료!")

if __name__ == "__main__":
    main()