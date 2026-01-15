#!/usr/bin/env python3
"""
앞도어의 상단, 중간, 하단 각각의 불량 클래스별 개수와 퍼센테이지를 출력하고,
good 폴더의 이미지 총 개수를 출력하는 스크립트


# 이 밖의 자세한 사용법은 USAGE.md 파일을 참조하세요.
사용법:
    python count_frontdoor_classes.py 0807 1109
"""

import os
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re


def is_date_in_range(date_str, start_date, end_date):
    """날짜가 범위 내에 있는지 확인"""
    try:
        date_num = int(date_str)
        start_num = int(start_date)
        end_num = int(end_date)
        return start_num <= date_num <= end_num
    except:
        return False


def count_images_in_good_folder(good_folder_path):
    """good 폴더 안에 있는 이미지 수를 세기 (images 하위 폴더 또는 직접)"""
    count = 0
    
    if not os.path.exists(good_folder_path):
        return 0
    
    # images 하위 폴더가 있는지 확인
    images_folder = os.path.join(good_folder_path, 'images')
    if os.path.exists(images_folder) and os.path.isdir(images_folder):
        # images 폴더 안의 이미지 파일들
        for file in os.listdir(images_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                count += 1
    else:
        # images 폴더가 없으면 good 폴더 안의 이미지 파일들
        for file in os.listdir(good_folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                count += 1
    
    return count


def process_frontdoor_excel(excel_path):
    """frontdoor.xlsx 파일을 읽어서 상단, 중간, 하단의 클래스 값들을 반환"""
    if not os.path.exists(excel_path):
        return None
    
    try:
        df = pd.read_excel(excel_path)
        
        # 상단, 중간, 하단 열이 있는지 확인
        required_cols = ['상단', '중간', '하단']
        if not all(col in df.columns for col in required_cols):
            # 열이 없으면 None 반환 (무시)
            return None
        
        results = {
            '상단': [],
            '중간': [],
            '하단': []
        }
        
        for _, row in df.iterrows():
            # 상단 처리
            if pd.notna(row['상단']):
                try:
                    value = int(row['상단'])
                    if value in [1, 2, 3, 4]:
                        results['상단'].append(value)
                except (ValueError, TypeError):
                    pass
            
            # 중간 처리 (null이면 무시)
            if pd.notna(row['중간']):
                try:
                    value = int(row['중간'])
                    if value in [1, 2, 3, 4]:
                        results['중간'].append(value)
                except (ValueError, TypeError):
                    pass
            
            # 하단 처리
            if pd.notna(row['하단']):
                try:
                    value = int(row['하단'])
                    if value in [1, 2, 3, 4]:
                        results['하단'].append(value)
                except (ValueError, TypeError):
                    pass
        
        return results
    
    except Exception as e:
        print(f"  오류: {excel_path} 읽기 실패 - {e}")
        return None


def process_date_range(start_date, end_date):
    """지정된 날짜 범위의 frontdoor 데이터를 처리"""
    base_dir = "/workspace/datasets"
    
    # 전체 통계
    total_counts = {
        '상단': defaultdict(int),
        '중간': defaultdict(int),
        '하단': defaultdict(int)
    }
    
    total_good_images = 0
    
    # 날짜 폴더 찾기
    date_folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and is_date_in_range(item, start_date, end_date):
            date_folders.append(item)
    
    date_folders.sort()
    
    if not date_folders:
        print(f"❌ {start_date}~{end_date} 날짜 범위에 해당하는 폴더가 없습니다.")
        return
    
    print(f"=== {start_date}~{end_date} 날짜 범위의 frontdoor 분석 ===\n")
    
    processed_dates = []
    
    for date_folder in date_folders:
        frontdoor_dir = os.path.join(base_dir, date_folder, 'frontdoor')
        excel_path = os.path.join(frontdoor_dir, 'frontdoor.xlsx')
        good_folder = os.path.join(frontdoor_dir, 'good')
        
        if not os.path.exists(frontdoor_dir):
            continue
        
        # Excel 파일 처리
        excel_results = process_frontdoor_excel(excel_path)
        
        if excel_results is not None:
            processed_dates.append(date_folder)
            for region in ['상단', '중간', '하단']:
                for value in excel_results[region]:
                    total_counts[region][value] += 1
        
        # good 폴더 이미지 수 세기
        good_count = count_images_in_good_folder(good_folder)
        if good_count > 0:
            total_good_images += good_count
            if date_folder not in processed_dates:
                processed_dates.append(date_folder)
    
    print(f"처리된 날짜: {len(processed_dates)}개")
    if len(processed_dates) > 0:
        print(f"  {', '.join(processed_dates[:10])}{'...' if len(processed_dates) > 10 else ''}\n")
    
    # 각 영역별 통계 출력
    regions = ['상단', '중간', '하단']
    
    for region in regions:
        print(f"=== {region} ===")
        region_total = sum(total_counts[region].values())
        
        if region_total == 0:
            print("  데이터 없음\n")
            continue
        
        for value in [1, 2, 3, 4]:
            count = total_counts[region][value]
            percentage = (count / region_total) * 100 if region_total > 0 else 0
            print(f"  클래스 {value}: {count}개 ({percentage:.2f}%)")
        
        print(f"  총계: {region_total}개\n")
    
        # good 이미지를 클래스 1에 추가한 통계
        print(f"=== {region} (good 포함) ===")
        region_with_good = defaultdict(int)
        for value in [1, 2, 3, 4]:
            region_with_good[value] = total_counts[region][value]
    
    # good 이미지를 클래스 1에 추가
        region_with_good[1] += total_good_images
        region_total_with_good = region_total + total_good_images
    
        if region_total_with_good == 0:
        print("  데이터 없음\n")
    else:
        for value in [1, 2, 3, 4]:
                count = region_with_good[value]
                percentage = (count / region_total_with_good) * 100 if region_total_with_good > 0 else 0
            print(f"  클래스 {value}: {count}개 ({percentage:.2f}%)")
        
            print(f"  총계: {region_total_with_good}개 (기본 {region_total}개 + good {total_good_images}개)\n")


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법:")
        print("  단일 날짜: python count_frontdoor_classes.py <날짜>")
        print("  기간별: python count_frontdoor_classes.py <시작날짜> <끝날짜>")
        print("\n예시:")
        print("  python count_frontdoor_classes.py 1031")
        print("  python count_frontdoor_classes.py 0807 1103")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # 단일 날짜
        date = sys.argv[1]
        process_date_range(date, date)
    elif len(sys.argv) == 3:
        # 기간별
        start_date = sys.argv[1]
        end_date = sys.argv[2]
        process_date_range(start_date, end_date)
    else:
        print("❌ 잘못된 인수 개수입니다.")
        print("사용법:")
        print("  단일 날짜: python count_frontdoor_classes.py <날짜>")
        print("  기간별: python count_frontdoor_classes.py <시작날짜> <끝날짜>")
        sys.exit(1)


if __name__ == "__main__":
    main()
