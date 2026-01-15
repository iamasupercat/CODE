#!/usr/bin/env python3
"""
CropforBB.py의 크롭 결과를 시각화하는 디버그 스크립트
원본 이미지에 OBB 바운딩 박스와 실제 크롭된 영역(BB)을 함께 표시합니다.


# 이 밖의 자세한 사용법은 USAGE.md 파일을 참조하세요.
사용법:
    python debug_crop.py <원본_이미지_경로>
    
결과:
    - 화면에 원본 이미지 + OBB 박스 + 크롭 영역 표시
    - {원본폴더}/debug_crop_visual/{이미지명}_debug.png 파일로 저장
    - 바운딩 박스 색상:
      - 초록색: OBB (angle=0, 정상)
      - 노란색: OBB (angle≠0, 문제)
      - 파란색: 실제 크롭된 영역 (BB)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import math


def load_yolo_label(label_path):
    """YOLO 라벨 파일 로드 (OBB 포맷)"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        angle = float(parts[5]) if len(parts) > 5 else 0.0
                        labels.append([class_id, x_center, y_center, width, height, angle])
    return labels


def compute_rotated_rect_corners(cx, cy, w, h, angle):
    """회전된 사각형의 네 꼭짓점 계산"""
    dx = w / 2.0
    dy = h / 2.0
    # 네 꼭짓점 (중심 기준)
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
        corners.append((int(rx), int(ry)))
    return corners


def yolo_to_bbox(yolo_line, img_width, img_height):
    """YOLO OBB를 BB로 변환 (크롭 영역 계산용)"""
    parts = yolo_line.strip().split()
    cls = int(parts[0])
    vals = list(map(float, parts[1:]))
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


def draw_bbox(img, labels, img_width, img_height, show_crop_boxes=True):
    """이미지에 바운딩 박스 그리기 (OBB + 크롭 영역)"""
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    for label in labels:
        class_id, x_center, y_center, width, height, angle = label
        
        # 클래스 0, 1만 처리 (볼트)
        if class_id not in [0, 1]:
            continue
        
        # YOLO 형식을 절대 좌표로 변환
        cx = x_center * w
        cy = y_center * h
        w_box = width * w
        h_box = height * h
        
        # OBB 바운딩 박스 그리기
        if abs(angle) > 0.001:  # 각도가 0이 아니면 (문제)
            corners = compute_rotated_rect_corners(cx, cy, w_box, h_box, angle)
            # 다각형으로 그리기 (노란색)
            pts = np.array(corners, np.int32)
            cv2.polylines(img_copy, [pts], True, (0, 255, 255), 3)  # 노란색
            
            # 클래스 ID 표시 (중심점에)
            cv2.putText(img_copy, f'C{class_id}', (int(cx) - 10, int(cy) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 각도 표시
            cv2.putText(img_copy, f'A{angle:.2f}', (corners[2][0], corners[2][1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            # 각도가 없으면 일반 사각형 그리기 (초록색)
            x = int(cx - w_box / 2)
            y = int(cy - h_box / 2)
            cv2.rectangle(img_copy, (x, y), (x + int(w_box), y + int(h_box)), (0, 255, 0), 3)  # 초록색
            
            # 클래스 ID 표시
            cv2.putText(img_copy, f'C{class_id}', (x, max(10, y - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 실제 크롭된 영역 표시 (파란색)
        if show_crop_boxes:
            yolo_line = f"{class_id} {x_center} {y_center} {width} {height} {angle}"
            _, x1, y1, x2, y2 = yolo_to_bbox(yolo_line, img_width, img_height)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란색
    
    return img_copy


def visualize_crop(original_img_path):
    """원본 이미지와 크롭 영역을 시각화"""
    original_img_path = Path(original_img_path)
    
    if not original_img_path.exists():
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {original_img_path}")
        return
    
    # 경로 설정
    images_dir = original_img_path.parent  # .../bad/images 또는 .../good/images
    base_dir = images_dir.parent  # .../bad 또는 .../good
    subfolder_dir = base_dir.parent  # .../frontfender 등
    date_dir = subfolder_dir.parent  # .../1010 등
    
    category = original_img_path.parent.name  # 'images'
    
    if category != 'images':
        print("오류: images 폴더의 이미지 파일을 지정해주세요.")
        return
    
    labels_dir = base_dir / 'labels'
    base_name = original_img_path.stem
    label_path = labels_dir / f"{base_name}.txt"
    
    # 원본 이미지 로드
    original_img = cv2.imread(str(original_img_path))
    if original_img is None:
        print(f"오류: 이미지를 로드할 수 없습니다: {original_img_path}")
        return
    
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_labels = load_yolo_label(label_path)
    
    # 클래스 0, 1만 필터링
    bolt_labels = [label for label in original_labels if label[0] in [0, 1]]
    
    if not bolt_labels:
        print(f"경고: 클래스 0 또는 1이 없습니다. (라벨 파일: {label_path})")
        return
    
    # debug_crop 이미지 로드 (있는 경우)
    debug_crop_path = base_dir / 'debug_crop' / original_img_path.name
    debug_crop_img = None
    if debug_crop_path.exists():
        debug_crop_img = cv2.imread(str(debug_crop_path))
        if debug_crop_img is not None:
            debug_crop_img = cv2.cvtColor(debug_crop_img, cv2.COLOR_BGR2RGB)
    
    # 바운딩 박스 그리기
    img_width, img_height = original_img_rgb.shape[1], original_img_rgb.shape[0]
    img_with_bbox = draw_bbox(original_img_rgb, bolt_labels, img_width, img_height, show_crop_boxes=True)
    
    # 시각화
    n_images = 2 if debug_crop_img is not None else 1
    fig, axes = plt.subplots(1, n_images, figsize=(10 * n_images, 10))
    
    if n_images == 1:
        axes = [axes]
    
    # 원본 이미지 + 바운딩 박스
    axes[0].imshow(img_with_bbox)
    axes[0].set_title(f'Original with BBoxes\n{base_name}', fontsize=12)
    axes[0].axis('off')
    
    # 범례 추가
    legend_text = "초록: OBB (angle=0, 정상)\n노랑: OBB (angle≠0, 문제)\n파랑: 크롭 영역 (BB)"
    axes[0].text(0.02, 0.98, legend_text, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # debug_crop 이미지 (있는 경우)
    if debug_crop_img is not None:
        axes[1].imshow(debug_crop_img)
        axes[1].set_title(f'Debug Crop (from CropforBB.py)\n{base_name}', fontsize=12)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    # 이미지 저장
    output_dir = base_dir / 'debug_crop_visual'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{base_name}_debug.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n시각화 이미지 저장: {output_path}")
    
    plt.show()
    
    # 라벨 정보 출력
    print("\n=== 라벨 정보 (클래스 0, 1만) ===")
    print(f"라벨 파일: {label_path}")
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            bolt_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and int(parts[0]) in [0, 1]:
                    bolt_lines.append(line.rstrip())
                    print(line.rstrip())
            print(f"  (볼트 라벨 {len(bolt_lines)}개)")
            
            # angle이 0이 아닌 경우 확인
            problem_count = 0
            for line in bolt_lines:
                parts = line.strip().split()
                if len(parts) >= 6:
                    angle = float(parts[5])
                    if abs(angle) > 1e-6:
                        problem_count += 1
            if problem_count > 0:
                print(f"\n⚠️  경고: angle이 0이 아닌 라벨이 {problem_count}개 있습니다!")
    else:
        print("  (파일 없음)")


def main():
    if len(sys.argv) < 2:
        print("사용법: python debug_crop.py <원본_이미지_경로>")
        print("\n예시:")
        print("  # 절대 경로")
        print("  python debug_crop.py /workspace/datasets/1010/frontfender/good/images/good_8128_1_13_62c4b142-e159-4c33-9368-98c3502cc696.jpg")
        print("\n  # 상대 경로 (datasets 폴더에서 실행)")
        print("  cd /workspace/datasets")
        print("  python CODE/debug_crop.py 1010/frontfender/good/images/good_8128_1_13_62c4b142-e159-4c33-9368-98c3502cc696.jpg")
        sys.exit(1)
    
    original_img_path = sys.argv[1]
    visualize_crop(original_img_path)


if __name__ == '__main__':
    main()

