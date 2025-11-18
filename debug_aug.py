#!/usr/bin/env python3
"""
Augmentation 결과를 시각화하는 디버그 스크립트

원본 이미지와 증강된 이미지들(noise, invert, flip)을 바운딩 박스와 함께 시각화합니다.

사용법:
    python debug_aug.py <원본_이미지_경로>
    
예시:
    # 0718 폴더의 이미지
    python debug_aug.py /home/work/datasets/0718/frontfender/bad/images/bad_0216_1_1_5a269cdf-35fe-4259-a086-7ccab92112ae.jpg
    
    # 1103 폴더의 이미지
    python debug_aug.py /home/work/datasets/1103/frontfender/bad/images/bad_6921_1_7_7576ea8b-45c4-4df1-8b49-6eb41704e7fa.jpg
    python debug_aug.py /home/work/datasets/1103/hood/bad/images/bad_0688_1_12_0a258553-3a33-41e9-9393-107f5b275f75.jpg

    # 상대 경로 사용
    cd /home/work/datasets
    python CODE/debug_aug.py 0718/frontfender/bad/images/bad_0216_1_1_5a269cdf-35fe-4259-a086-7ccab92112ae.jpg

결과:
    - 화면에 원본 + 증강 이미지들 표시
    - {원본폴더}/debug_aug_output/{이미지명}_debug.png 파일로 저장
    - 바운딩 박스 색상: 원본(초록), noise(빨강), invert(파랑), flip(노랑)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
import os
import math


def load_yolo_label(label_path):
    """YOLO 라벨 파일 로드"""
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


def draw_bbox(img, labels, color=(0, 255, 0), thickness=2):
    """이미지에 바운딩 박스 그리기 (OBB 지원)"""
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    for label in labels:
        class_id, x_center, y_center, width, height, angle = label
        
        # YOLO 형식을 절대 좌표로 변환
        cx = x_center * w
        cy = y_center * h
        w_box = width * w
        h_box = height * h
        
        # 각도가 있으면 회전된 사각형 그리기
        if abs(angle) > 0.001:  # 각도가 0이 아니면
            corners = compute_rotated_rect_corners(cx, cy, w_box, h_box, angle)
            # 다각형으로 그리기
            pts = np.array(corners, np.int32)
            cv2.polylines(img_copy, [pts], True, color, thickness)
            
            # 클래스 ID 표시 (중심점에)
            cv2.putText(img_copy, f'C{class_id}', (int(cx) - 10, int(cy) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 각도 표시
            cv2.putText(img_copy, f'A{angle:.2f}', (corners[2][0], corners[2][1] + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            # 각도가 없으면 일반 사각형 그리기
            x = int(cx - w_box / 2)
            y = int(cy - h_box / 2)
            cv2.rectangle(img_copy, (x, y), (x + int(w_box), y + int(h_box)), color, thickness)
            
            # 클래스 ID 표시
            cv2.putText(img_copy, f'C{class_id}', (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_copy


def visualize_augmentation(original_img_path):
    """원본 이미지와 증강된 이미지들을 시각화"""
    original_img_path = Path(original_img_path)
    
    if not original_img_path.exists():
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {original_img_path}")
        return
    
    # 경로 설정
    images_dir = original_img_path.parent  # .../bad/images 또는 .../good/images
    base_dir = images_dir.parent  # .../bad 또는 .../good
    subfolder_dir = base_dir.parent  # .../frontfender 등
    date_dir = subfolder_dir.parent  # .../1103 등
    
    category = original_img_path.parent.name  # 'images' 또는 'labels'
    
    if category != 'images':
        print("오류: images 폴더의 이미지 파일을 지정해주세요.")
        return
    
    labels_dir = base_dir / 'labels'
    base_name = original_img_path.stem
    label_path = labels_dir / f"{base_name}.txt"
    
    # aug 폴더 찾기: subfolder_aug/bad 또는 subfolder_aug/good
    subfolder_name = subfolder_dir.name  # frontfender 등
    aug_subfolder_dir = date_dir / f"{subfolder_name}_aug"  # .../1103/frontfender_aug
    aug_images_dir = aug_subfolder_dir / base_dir.name / 'images'  # .../1103/frontfender_aug/bad/images
    aug_labels_dir = aug_subfolder_dir / base_dir.name / 'labels'  # .../1103/frontfender_aug/bad/labels
    
    # 원본 이미지 로드
    original_img = cv2.imread(str(original_img_path))
    if original_img is None:
        print(f"오류: 이미지를 로드할 수 없습니다: {original_img_path}")
        return
    
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_labels = load_yolo_label(label_path)
    
    # 증강 이미지들 로드 (3가지 모두 필수)
    aug_images = []
    aug_labels_list = []
    aug_names = []
    
    aug_types = ['noise', 'invert', 'flip']
    missing_aug = []
    
    for aug_type in aug_types:
        aug_img_path = aug_images_dir / f"{base_name}_{aug_type}.jpg"
        aug_label_path = aug_labels_dir / f"{base_name}_{aug_type}.txt"
        
        if aug_img_path.exists():
            aug_img = cv2.imread(str(aug_img_path))
            if aug_img is not None:
                aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                aug_labels = load_yolo_label(aug_label_path)
                aug_images.append(aug_img_rgb)
                aug_labels_list.append(aug_labels)
                aug_names.append(aug_type)
            else:
                missing_aug.append(aug_type)
        else:
            missing_aug.append(aug_type)
    
    # 3가지 aug가 모두 있는지 확인
    if missing_aug:
        print(f"\n⚠️  경고: 다음 증강 이미지가 없습니다: {', '.join(missing_aug)}")
        print(f"   aug 폴더 경로: {aug_images_dir}")
        print(f"   augmentation.py를 먼저 실행하여 증강 이미지를 생성하세요.\n")
    
    if len(aug_images) == 0:
        print("⚠️  증강 이미지가 하나도 없습니다. 원본만 표시합니다.\n")
    
    # 시각화
    n_images = 1 + len(aug_images)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    # 원본 이미지
    img_with_bbox = draw_bbox(original_img_rgb, original_labels, color=(0, 255, 0))
    axes[0].imshow(img_with_bbox)
    axes[0].set_title(f'Original\n{base_name}', fontsize=10)
    axes[0].axis('off')
    
    # 증강 이미지들
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0)]  # 빨강, 파랑, 노랑
    for i, (aug_img, aug_labels, aug_name) in enumerate(zip(aug_images, aug_labels_list, aug_names)):
        color = colors[i % len(colors)]
        img_with_bbox = draw_bbox(aug_img, aug_labels, color=color)
        axes[i + 1].imshow(img_with_bbox)
        axes[i + 1].set_title(f'{aug_name.capitalize()}\n{base_name}_{aug_name}', fontsize=10)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    # 이미지 저장
    output_dir = base_dir / 'debug_aug_output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{base_name}_debug.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n시각화 이미지 저장: {output_path}")
    
    plt.show()
    
    # 라벨 정보 출력 (3가지 aug 모두 확인)
    print("\n=== 라벨 정보 ===")
    print(f"원본: {label_path}")
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                print(line.rstrip())
            print(f"  ({len([l for l in lines if l.strip()])}줄)")
    else:
        print("  (파일 없음)")
    
    # 3가지 aug 라벨 모두 확인
    for aug_type in ['noise', 'invert', 'flip']:
        aug_label_path = aug_labels_dir / f"{base_name}_{aug_type}.txt"
        print(f"\n{aug_type}: {aug_label_path}")
        if aug_label_path.exists():
            with open(aug_label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    print(line.rstrip())
                print(f"  ({len([l for l in lines if l.strip()])}줄)")
        else:
            print("  (파일 없음)")


def main():
    if len(sys.argv) < 2:
        print("사용법: python debug_aug.py <원본_이미지_경로>")
        print("\n예시:")
        print("  # 절대 경로")
        print("  python debug_aug.py /home/work/datasets/0718/frontfender/bad/images/bad_0216_1_1_5a269cdf-35fe-4259-a086-7ccab92112ae.jpg")
        print("  python debug_aug.py /home/work/datasets/1103/frontfender/bad/images/bad_6921_1_7_7576ea8b-45c4-4df1-8b49-6eb41704e7fa.jpg")
        print("\n  # 상대 경로 (datasets 폴더에서 실행)")
        print("  cd /home/work/datasets")
        print("  python CODE/debug_aug.py 0718/frontfender/bad/images/bad_0216_1_1_5a269cdf-35fe-4259-a086-7ccab92112ae.jpg")
        sys.exit(1)
    
    original_img_path = sys.argv[1]
    visualize_augmentation(original_img_path)


if __name__ == '__main__':
    main()

