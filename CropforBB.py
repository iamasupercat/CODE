#!/usr/bin/env python3
"""
볼트 크롭에 사용
- 볼트 라벨링에는 회전을 적용하지 않았음


# 이 밖의 자세한 사용법은 USAGE.md 파일을 참조하세요.
사용법:
    python CropforBB.py \
        --date-range 0616 1109 \
        --clean

python CropforBB.py \
        --obb-date-range 0616 0806 \
        --clean




[로직 설명]
YOLO OBB 라벨을 BB처럼 처리하여 볼트를 크롭하는 스크립트
- OBB 포맷이지만 angle을 무시하고 BB처럼 크롭
- 클래스 0(정측면), 1(측면)만 처리
    * 정면 볼트는 학습과 테스트 모두에 사용하지 않아 backup폴더에 따로 빼둠
    * 데이터가 부족할 경우, backup폴더 데이터를 가져와 사용할 수 있으며 이때 라벨링 번호는 
        0 정측면 1 정면 2 측면 순
        혹은 0 정측면 1 측면 8 정면 순
        (정면데이터를 사용하려할 경우, 라벨링 번호 확인 필수)
        (부위별 폴더 내 엑셀 파일을 참조하여 어떤 폴더에 넣을지 정할 수 있음)
- angle이 0이 아닌 경우 문제로 감지 및 보고
"""

import os
import argparse
from PIL import Image, ImageDraw, ImageOps
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


def corners_to_xywha(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    변환된 포맷(xyxyxyxy, 정규화된 좌표)을 원본 포맷(xywha, 정규화된 좌표)으로 역변환
    4개 모서리 좌표에서 중심점, 너비, 높이, 각도를 계산
    
    입력: 정규화된 모서리 좌표 (0~1 범위)
    출력: 정규화된 중심점, 너비, 높이, 각도 (라디안)
    """
    # 중심점 계산 (정규화된 좌표)
    cx = (x1 + x2 + x3 + x4) / 4.0
    cy = (y1 + y2 + y3 + y4) / 4.0
    
    # 첫 번째 모서리와 두 번째 모서리 사이의 벡터
    vx = x2 - x1
    vy = y2 - y1
    
    # 너비 계산 (정규화된 거리)
    w = math.sqrt(vx**2 + vy**2)
    
    # 두 번째 모서리와 세 번째 모서리 사이의 벡터
    vx2 = x3 - x2
    vy2 = y3 - y2
    
    # 높이 계산 (정규화된 거리)
    h = math.sqrt(vx2**2 + vy2**2)
    
    # 각도 계산 (라디안)
    angle = math.atan2(vy, vx)
    
    # 정규화된 좌표 반환
    return cx, cy, w, h, angle


def parse_obb_label(line):
    """
    OBB 라벨 파싱: 두 가지 포맷 지원
    1. 원본 포맷: class x y w h angle (6개 값)
    2. 변환된 포맷: class x1 y1 x2 y2 x3 y3 x4 y4 (9개 값)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    
    cls = int(float(parts[0]))
    
    # 포맷 자동 감지
    if len(parts) == 6:
        # 원본 포맷: class x y w h angle
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        angle = float(parts[5])
        return cls, x, y, w, h, angle
    elif len(parts) == 9:
        # 변환된 포맷: class x1 y1 x2 y2 x3 y3 x4 y4
        x1, y1 = float(parts[1]), float(parts[2])
        x2, y2 = float(parts[3]), float(parts[4])
        x3, y3 = float(parts[5]), float(parts[6])
        x4, y4 = float(parts[7]), float(parts[8])
        
        # 원본 포맷으로 역변환
        cx, cy, w, h, angle = corners_to_xywha(x1, y1, x2, y2, x3, y3, x4, y4)
        
        # 정규화된 좌표 반환
        return cls, cx, cy, w, h, angle
    elif len(parts) == 5:
        # angle이 없는 경우 (기본값 0.0)
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        angle = 0.0
        return cls, x, y, w, h, angle
    else:
        return None


def compute_rotated_rect_corners(cx, cy, w, h, angle, img_width, img_height):
    """회전된 사각형의 네 꼭짓점 계산 (PIL 좌표계용)"""
    dx = w / 2.0
    dy = h / 2.0
    local_pts = [
        (-dx, -dy),
        ( dx, -dy),
        ( dx,  dy),
        (-dx,  dy),
    ]
    c = math.cos(angle)
    s = math.sin(angle)
    corners = []
    for lx, ly in local_pts:
        rx = c * lx - s * ly + cx
        ry = s * lx + c * ly + cy
        corners.append((rx, ry))
    return corners


def draw_obb_bbox(draw, x_center, y_center, width, height, angle, img_width, img_height, 
                  outline='red', width_line=3, cls=None):
    """OBB 바운딩 박스를 그리는 함수 (PIL ImageDraw용)"""
    cx = x_center * img_width
    cy = y_center * img_height
    w_box = width * img_width
    h_box = height * img_height
    
    if abs(angle) > 0.001:  # 각도가 0이 아니면
        corners = compute_rotated_rect_corners(cx, cy, w_box, h_box, angle, img_width, img_height)
        draw.polygon(corners, outline=outline, width=width_line)
        if cls is not None:
            text = f'C{cls}'
            draw.text((int(cx) - 10, int(cy) - 10), text, fill=outline)
        angle_text = f'A{angle:.2f}'
        draw.text((int(corners[2][0]), int(corners[2][1]) + 10), angle_text, fill=outline)
    else:
        x = int(cx - w_box / 2)
        y = int(cy - h_box / 2)
        draw.rectangle([x, y, x + int(w_box), y + int(h_box)], outline=outline, width=width_line)
        if cls is not None:
            text = f'C{cls}'
            draw.text((x, max(0, y - 15)), text, fill=outline)


def yolo_to_bbox(x, y, w, h, img_width, img_height):
    """YOLO OBB를 BB로 변환 (angle 무시)"""
    x_center = x * img_width
    y_center = y * img_height
    w_box = w * img_width
    h_box = h * img_height
    x1 = int(round(x_center - w_box / 2))
    y1 = int(round(y_center - h_box / 2))
    x2 = int(round(x_center + w_box / 2))
    y2 = int(round(y_center + h_box / 2))
    # 이미지 경계로 clipping
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    return x1, y1, x2, y2


def clean_directory(target_dir):
    """
    지정된 날짜 폴더 내의 모든 파트(frontfender, hood, trunklid)에서
    crop_bolt, crop_bolt_aug, debug_crop 폴더를 삭제합니다.
    """
    parts = ['frontfender', 'hood', 'trunklid']
    sub_dirs = ['bad', 'good']
    
    print(f"🧹 [{os.path.basename(target_dir)}] 청소(삭제) 시작...")
    
    cleaned_count = 0
    
    for part in parts:
        for sub in sub_dirs:
            base_path = os.path.join(target_dir, part, sub)
            if not os.path.exists(base_path):
                continue
            
            # debug_crop 삭제
            debug_path = os.path.join(base_path, 'debug_crop')
            if os.path.exists(debug_path):
                try:
                    shutil.rmtree(debug_path)
                    print(f"  - 삭제됨: {debug_path}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"  ! 삭제 실패: {debug_path} ({e})")

            # crop_bolt, crop_bolt_aug 삭제
            for crop_folder in ['crop_bolt', 'crop_bolt_aug']:
                crop_path = os.path.join(base_path, crop_folder)
                if os.path.exists(crop_path):
                    try:
                        shutil.rmtree(crop_path)
                        print(f"  - 삭제됨: {crop_path}")
                        cleaned_count += 1
                    except Exception as e:
                        print(f"  ! 삭제 실패: {crop_path} ({e})")
                        
    if cleaned_count == 0:
        print("  (삭제할 폴더가 없습니다)")
    print("--------------------------------------------------")


def process_bolt_mode(base_dir):
    """
    볼트 모드: 클래스 0, 1만 처리, OBB를 BB처럼 크롭
    반환: 문제 파일 목록 [(label_path, img_name, line_num, angle), ...]
    """
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    debug_dir = os.path.join(base_dir, 'debug_crop')
    
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        return []
    
    os.makedirs(debug_dir, exist_ok=True)
    crop_bolt_dir = os.path.join(base_dir, 'crop_bolt')
    for i in range(2):
        os.makedirs(os.path.join(crop_bolt_dir, str(i)), exist_ok=True)
    
    problem_files = []
    cls_counters = {0: 0, 1: 0}
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        # .bak 파일이 있으면 우선 사용 (원본 포맷), 없으면 현재 txt 파일 사용
        # YOLO 학습 중에는 txt 파일을 절대 수정하지 않음
        bak_path = label_path + '.bak'
        actual_label_path = bak_path if os.path.exists(bak_path) else label_path
        
        if not os.path.exists(actual_label_path):
            continue
        
        try:
            # PIL로 이미지 로드 (EXIF 보정 포함)
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            img_width, img_height = img.size
            debug_img = img.copy()
            draw = ImageDraw.Draw(debug_img)
            
            labels_data = []
            
            with open(actual_label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    parsed = parse_obb_label(line)
                    if parsed is None:
                        continue
                    cls, x, y, w, h, angle = parsed
                    # parse_obb_label에서 이미 정규화된 좌표를 반환하므로 그대로 사용
                    if cls not in [0, 1]:
                        continue
                    
                    # 클래스가 0,1인데 angle이 0이 아니면 문제 파일로 기록하고 크롭하지 않음
                    if abs(angle) > 1e-6:
                        problem_files.append((actual_label_path, img_name, line_num, angle))
                        print(f"[문제] {actual_label_path} - {img_name} (라인 {line_num}): 클래스 {cls}인데 angle={angle} (0이 아님, 크롭 무시)")
                        # 디버그 이미지에는 표시하되 크롭은 하지 않음
                        bbox_color = 'yellow'
                        draw_obb_bbox(draw, x, y, w, h, angle, img_width, img_height, 
                                     outline=bbox_color, width_line=3, cls=cls)
                        continue
                    
                    # angle이 0인 경우만 labels_data에 추가하여 크롭 진행
                    labels_data.append((cls, x, y, w, h, angle, line_num))
            
            for cls, x, y, w, h, angle, line_num in labels_data:
                # OBB 바운딩 박스 그리기 (디버그용) - angle이 0인 경우만 여기 도달
                bbox_color = 'red'
                draw_obb_bbox(draw, x, y, w, h, angle, img_width, img_height, 
                             outline=bbox_color, width_line=3, cls=cls)
                
                # BB로 변환하여 크롭 (angle 무시)
                x1, y1, x2, y2 = yolo_to_bbox(x, y, w, h, img_width, img_height)
                
                if x2 > x1 and y2 > y1:
                    crop = img.crop((x1, y1, x2, y2))
                    
                    idx = cls_counters[cls]
                    cls_counters[cls] += 1
                    
                    crop_filename = f"{os.path.splitext(img_name)[0]}_{cls}_{idx}.jpg"
                    crop_path = os.path.join(crop_bolt_dir, str(cls), crop_filename)
                    crop.save(crop_path)
                    
                    # 실제 크롭된 영역도 파란색으로 표시
                    draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)
            
            # 디버그 이미지 저장
            debug_save_path = os.path.join(debug_dir, img_name)
            debug_img.save(debug_save_path)

        except Exception as e:
            print(f"오류: {img_path}, {e}")
            continue
    
    return problem_files


def main():
    parser = argparse.ArgumentParser(description='YOLO OBB 라벨을 BB처럼 처리하여 볼트를 크롭합니다.')
    parser.add_argument('--target_dir', nargs='*',
                        help='일반 폴더 날짜들 (예: 0616 0718 0721)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                        help='일반 폴더 날짜 구간 (MMDD 또는 YYYYMMDD)')
    parser.add_argument('--obb-folders', nargs='*',
                        help='OBB 폴더 날짜들 (예: 0718 0806)')
    parser.add_argument('--obb-date-range', nargs=2, metavar=('START', 'END'),
                        help='OBB 폴더 날짜 구간 (MMDD 또는 YYYYMMDD)')
    parser.add_argument('--clean', action='store_true',
                        help='실행 전 기존 crop_bolt, crop_bolt_aug, debug_crop 폴더 삭제')
    args = parser.parse_args()
    
    base_path = "/home/ciw/work/datasets"
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

    # 일반 폴더와 OBB 폴더 구분하여 출력
    normal_dirs = [d for d in target_dirs if not d.startswith(obb_base_path)]
    obb_target_dirs = [d for d in target_dirs if d.startswith(obb_base_path)]
    
    print(f"대상 폴더:")
    if normal_dirs:
        print(f"  일반 폴더: {[os.path.basename(p) for p in normal_dirs]}")
    if obb_target_dirs:
        print(f"  OBB 폴더: {[os.path.basename(p) for p in obb_target_dirs]}")
    
    # Clean 옵션 처리
    if args.clean:
        print("\n=== [CLEAN MODE] 기존 결과물 삭제 ===")
        for target_dir in target_dirs:
            clean_directory(target_dir)
        print("=== 청소 완료, 데이터 처리를 시작합니다 ===\n")
    
    all_problem_files = []
    
    for target_dir in target_dirs:
        print(f"\n폴더: {target_dir}")
        for part in ['frontfender', 'hood', 'trunklid']:
            p_dir = os.path.join(target_dir, part)
            if os.path.exists(os.path.join(p_dir, 'bad')):
                problem_files = process_bolt_mode(os.path.join(p_dir, 'bad'))
                for label_path, img_name, line_num, angle in problem_files:
                    all_problem_files.append({
                        'target_dir': target_dir,
                        'subfolder': part,
                        'set_type': 'bad',
                        'label_path': label_path,
                        'image_name': img_name,
                        'line_num': line_num,
                        'angle': angle
                    })
            if os.path.exists(os.path.join(p_dir, 'good')):
                problem_files = process_bolt_mode(os.path.join(p_dir, 'good'))
                for label_path, img_name, line_num, angle in problem_files:
                    all_problem_files.append({
                        'target_dir': target_dir,
                        'subfolder': part,
                        'set_type': 'good',
                        'label_path': label_path,
                        'image_name': img_name,
                        'line_num': line_num,
                        'angle': angle
                    })

    # 크롭이 전부 끝난 후 문제 파일 목록 출력
    if all_problem_files:
        print("\n" + "="*80)
        print("=== 문제가 있는 라벨링 파일 목록 ===")
        print("="*80)
        print(f"총 {len(all_problem_files)}개의 문제가 발견되었습니다.\n")
        
        for problem in all_problem_files:
            print(f"경로: {problem['target_dir']}/{problem['subfolder']}/{problem['set_type']}")
            print(f"  라벨 파일: {problem['label_path']}")
            print(f"  이미지 파일: {problem['image_name']}")
            print(f"  라인 번호: {problem['line_num']}")
            print(f"  Angle 값: {problem['angle']}")
            print()
        
        print("="*80)
        print("=== 요약 ===")
        print("="*80)
        path_counts = {}
        for problem in all_problem_files:
            key = f"{problem['target_dir']}/{problem['subfolder']}/{problem['set_type']}"
            path_counts[key] = path_counts.get(key, 0) + 1
        
        for path, count in sorted(path_counts.items()):
            print(f"  {path}: {count}개 문제")
    else:
        print("\n" + "="*80)
        print("=== 문제가 있는 라벨링 파일 없음 ===")
        print("="*80)
        print("모든 라벨링 파일이 정상입니다 (클래스 0,1의 angle이 모두 0입니다).")
    
    print("\n처리 완료!")


if __name__ == "__main__":
    main()
