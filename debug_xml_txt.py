#!/usr/bin/env python3
"""
XML → TXT 변환 결과를 시각화하는 디버그 스크립트
XML 파일과 변환된 TXT 파일의 바운딩 박스를 비교하여 변환이 올바른지 확인합니다.


# 이 밖의 자세한 사용법은 USAGE.md 파일을 참조하세요.
사용법:
    python debug_xml_txt.py <이미지_경로>

결과:
    - 화면에 원본 이미지 + XML 바운딩 박스 + TXT 바운딩 박스 표시
    - {labels폴더}/debug_xml_txt_output/{이미지명}_debug.png 파일로 저장
    - 바운딩 박스 색상: XML(빨강), TXT(초록)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import xml.etree.ElementTree as ET
import math


def parse_xml_label(xml_path):
    """XML 파일을 파싱하여 바운딩 박스 정보 반환"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"XML 파싱 오류: {xml_path} - {e}")
        return None, []
    
    # 이미지 크기 가져오기
    size = root.find('size')
    if size is None:
        return None, []
    
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    boxes = []
    class_mapping = {}
    
    # 모든 object 처리
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is None:
            continue
        
        class_name = name_elem.text.strip()
        
        # 클래스 ID 할당
        if class_name not in class_mapping:
            class_mapping[class_name] = len(class_mapping)
        
        class_id = class_mapping[class_name]
        
        # robndbox 정보 가져오기
        robndbox = obj.find('robndbox')
        if robndbox is None:
            continue
        
        try:
            cx = float(robndbox.find('cx').text)
            cy = float(robndbox.find('cy').text)
            w = float(robndbox.find('w').text)
            h = float(robndbox.find('h').text)
            angle = float(robndbox.find('angle').text)
        except (ValueError, AttributeError) as e:
            print(f"경고: {xml_path}의 robndbox 파싱 오류 - {e}")
            continue
        
        boxes.append([class_id, cx, cy, w, h, angle, class_name])
    
    return (img_width, img_height), boxes


def load_txt_label(txt_path):
    """TXT 파일을 로드하여 바운딩 박스 정보 반환"""
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
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
                        boxes.append([class_id, x_center, y_center, width, height, angle])
    return boxes


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


def draw_xml_bbox(img, boxes, img_width, img_height, color=(0, 0, 255), thickness=2):
    """XML 바운딩 박스 그리기 (절대 좌표)"""
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    # XML의 이미지 크기와 실제 이미지 크기가 다를 수 있으므로 스케일 조정
    scale_x = w / img_width
    scale_y = h / img_height
    
    for box in boxes:
        class_id, cx, cy, w_box, h_box, angle, class_name = box
        
        # 스케일 적용
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y
        w_scaled = w_box * scale_x
        h_scaled = h_box * scale_y
        
        # 각도가 있으면 회전된 사각형 그리기
        if abs(angle) > 0.001:
            corners = compute_rotated_rect_corners(cx_scaled, cy_scaled, w_scaled, h_scaled, angle)
            pts = np.array(corners, np.int32)
            cv2.polylines(img_copy, [pts], True, color, thickness)
            
            # 클래스 이름 표시
            cv2.putText(img_copy, f'{class_name}', (int(cx_scaled) - 20, int(cy_scaled) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 각도 표시
            cv2.putText(img_copy, f'A{angle:.2f}', (corners[2][0], corners[2][1] + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            # 각도가 없으면 일반 사각형 그리기
            x = int(cx_scaled - w_scaled / 2)
            y = int(cy_scaled - h_scaled / 2)
            cv2.rectangle(img_copy, (x, y), (x + int(w_scaled), y + int(h_scaled)), color, thickness)
            
            # 클래스 이름 표시
            cv2.putText(img_copy, f'{class_name}', (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_copy


def draw_txt_bbox(img, boxes, color=(0, 255, 0), thickness=2):
    """TXT 바운딩 박스 그리기 (정규화된 좌표)"""
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    for box in boxes:
        class_id, x_center, y_center, width, height, angle = box
        
        # YOLO 형식을 절대 좌표로 변환
        cx = x_center * w
        cy = y_center * h
        w_box = width * w
        h_box = height * h
        
        # 각도가 있으면 회전된 사각형 그리기
        if abs(angle) > 0.001:
            corners = compute_rotated_rect_corners(cx, cy, w_box, h_box, angle)
            pts = np.array(corners, np.int32)
            cv2.polylines(img_copy, [pts], True, color, thickness)
            
            # 클래스 ID 표시
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


def visualize_xml_txt(image_path):
    """XML과 TXT 바운딩 박스를 비교하여 시각화"""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    # 경로 설정
    images_dir = image_path.parent  # .../good/images 또는 .../bad/images
    base_dir = images_dir.parent  # .../good 또는 .../bad
    labels_dir = base_dir / 'labels'
    base_name = image_path.stem
    
    xml_path = labels_dir / f"{base_name}.xml"
    txt_path = labels_dir / f"{base_name}.txt"
    
    # 원본 이미지 로드
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"오류: 이미지를 로드할 수 없습니다: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # XML 파싱
    xml_size, xml_boxes = None, []
    if xml_path.exists():
        xml_size, xml_boxes = parse_xml_label(xml_path)
        if xml_size is None:
            print(f"경고: XML 파일을 파싱할 수 없습니다: {xml_path}")
    else:
        print(f"경고: XML 파일이 없습니다: {xml_path}")
    
    # TXT 로드
    txt_boxes = []
    if txt_path.exists():
        txt_boxes = load_txt_label(txt_path)
    else:
        print(f"경고: TXT 파일이 없습니다: {txt_path}")
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 원본 이미지
    axes[0].imshow(img_rgb)
    axes[0].set_title(f'Original\n{base_name}', fontsize=12)
    axes[0].axis('off')
    
    # 2. XML 바운딩 박스
    if xml_boxes and xml_size:
        img_xml = draw_xml_bbox(img_rgb, xml_boxes, xml_size[0], xml_size[1], color=(255, 0, 0))
        axes[1].imshow(img_xml)
        axes[1].set_title(f'XML (Red)\n{len(xml_boxes)} boxes', fontsize=12)
    else:
        axes[1].imshow(img_rgb)
        axes[1].set_title('XML (Red)\nNo boxes', fontsize=12)
    axes[1].axis('off')
    
    # 3. TXT 바운딩 박스
    if txt_boxes:
        img_txt = draw_txt_bbox(img_rgb, txt_boxes, color=(0, 255, 0))
        axes[2].imshow(img_txt)
        axes[2].set_title(f'TXT (Green)\n{len(txt_boxes)} boxes', fontsize=12)
    else:
        axes[2].imshow(img_rgb)
        axes[2].set_title('TXT (Green)\nNo boxes', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 이미지 저장
    output_dir = labels_dir / 'debug_xml_txt_output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{base_name}_debug.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n시각화 이미지 저장: {output_path}")
    
    plt.show()
    
    # 상세 정보 출력
    print("\n=== 변환 정보 ===")
    print(f"이미지 크기: {w} x {h}")
    if xml_size:
        print(f"XML 이미지 크기: {xml_size[0]} x {xml_size[1]}")
    print(f"\nXML 바운딩 박스: {len(xml_boxes)}개")
    for i, box in enumerate(xml_boxes):
        class_id, cx, cy, w_box, h_box, angle, class_name = box
        print(f"  {i+1}. {class_name} (ID:{class_id}) - cx:{cx:.1f}, cy:{cy:.1f}, w:{w_box:.1f}, h:{h_box:.1f}, angle:{angle:.3f}")
    
    print(f"\nTXT 바운딩 박스: {len(txt_boxes)}개")
    for i, box in enumerate(txt_boxes):
        class_id, x_center, y_center, width, height, angle = box
        print(f"  {i+1}. Class {class_id} - x:{x_center:.6f}, y:{y_center:.6f}, w:{width:.6f}, h:{height:.6f}, angle:{angle:.6f}")
    
    # 비교
    if len(xml_boxes) != len(txt_boxes):
        print(f"\n⚠️  경고: 바운딩 박스 개수가 다릅니다! (XML: {len(xml_boxes)}, TXT: {len(txt_boxes)})")
    else:
        print(f"\n✓ 바운딩 박스 개수 일치: {len(xml_boxes)}개")


def main():
    if len(sys.argv) < 2:
        print("사용법: python debug_xml_txt.py <이미지_경로>")
        print("\n예시:")
        print("  # 절대 경로")
        print("  python debug_xml_txt.py /home/ciw/work/datasets/OBB/0718/hood/good/images/good_9362_1_e97d8f94-e26c-455d-9c96-40a63f6a2c83.jpg")
        print("\n  # 상대 경로 (datasets 폴더에서 실행)")
        print("  cd /home/ciw/work/datasets")
        print("  python CODE/debug_xml_txt.py OBB/0718/hood/good/images/good_9362_1_e97d8f94-e26c-455d-9c96-40a63f6a2c83.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    visualize_xml_txt(image_path)


if __name__ == '__main__':
    main()

