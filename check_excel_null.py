#!/usr/bin/env python3
"""
frontdoor.xlsx 파일에서 상단, 중간, 하단 열의 null 값을 확인하는 스크립트



# 이 밖의 자세한 사용법은 USAGE.md 파일을 참조하세요.
사용법:
    python check_excel_null.py --date-range 0616 1109
"""

import os
import pandas as pd
import argparse
from collections import defaultdict


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


def load_excel_data(excel_path):
    """엑셀 파일 로드"""
    if not os.path.exists(excel_path):
        return None
    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        print(f"엑셀 파일 로드 실패: {excel_path}, 오류: {e}")
        return None


def check_null_values(df):
    """
    엑셀 데이터프레임에서 quality가 null인 행만 체크하여 상단, 중간, 하단 열의 null 값을 확인
    
    반환:
        {
            'mid_only': [],  # 중간만 null
            'top_or_bottom': [],  # 상단 또는 하단만 null (둘 중 하나만)
            'two_missing': [],  # 2개 누락
            'three_missing': []  # 3개 누락
        }
    """
    if df is None:
        return None
    
    # 필요한 열이 있는지 확인
    required_cols = ['상단', '중간', '하단']
    if not all(col in df.columns for col in required_cols):
        return None
    
    # quality 열이 있는지 확인
    if 'quality' not in df.columns:
        return None
    
    result = {
        'mid_only': [],
        'top_or_bottom': [],
        'two_missing': [],
        'three_missing': []
    }
    
    for idx, row in df.iterrows():
        # quality가 null인 행만 체크
        quality_val = row['quality']
        if not pd.isna(quality_val):
            continue
        
        high_val = row['상단']
        mid_val = row['중간']
        low_val = row['하단']
        
        # null 여부 확인
        high_null = pd.isna(high_val)
        mid_null = pd.isna(mid_val)
        low_null = pd.isna(low_val)
        
        null_count = sum([high_null, mid_null, low_null])
        
        if null_count == 0:
            # 모두 있으면 건너뜀
            continue
        elif null_count == 1:
            # 1개만 누락
            if mid_null and not high_null and not low_null:
                # 중간만 누락
                result['mid_only'].append(idx)
            elif (high_null or low_null) and not mid_null:
                # 상단 또는 하단만 누락 (둘 중 하나만)
                result['top_or_bottom'].append(idx)
        elif null_count == 2:
            # 2개 누락
            result['two_missing'].append(idx)
        elif null_count == 3:
            # 3개 누락
            result['three_missing'].append(idx)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='frontdoor.xlsx 파일에서 null 값을 확인합니다.')
    parser.add_argument('--target_dir', nargs='*',
                        help='일반 폴더 날짜들 (예: 0616 0718 0721)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                        help='일반 폴더 날짜 구간 (MMDD 또는 YYYYMMDD)')
    parser.add_argument('--obb-folders', nargs='*',
                        help='OBB 폴더 날짜들 (예: 0718 0806)')
    parser.add_argument('--obb-date-range', nargs=2, metavar=('START', 'END'),
                        help='OBB 폴더 날짜 구간 (MMDD 또는 YYYYMMDD)')
    
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

    print(f"대상 폴더: {[os.path.basename(p) for p in target_dirs]}\n")
    
    # 날짜별로 결과 수집
    date_results = defaultdict(lambda: {
        'mid_only': False,
        'top_or_bottom': False,
        'two_missing': False,
        'three_missing': False
    })
    
    for target_dir in target_dirs:
        frontdoor = os.path.join(target_dir, 'frontdoor')
        excel_path = os.path.join(frontdoor, 'frontdoor.xlsx')
        
        if not os.path.exists(excel_path):
            continue
        
        date_name = os.path.basename(target_dir)
        df = load_excel_data(excel_path)
        
        if df is None:
            continue
        
        null_result = check_null_values(df)
        
        if null_result is None:
            continue
        
        # 날짜별로 결과 기록
        if len(null_result['mid_only']) > 0:
            date_results[date_name]['mid_only'] = True
        if len(null_result['top_or_bottom']) > 0:
            date_results[date_name]['top_or_bottom'] = True
        if len(null_result['two_missing']) > 0:
            date_results[date_name]['two_missing'] = True
        if len(null_result['three_missing']) > 0:
            date_results[date_name]['three_missing'] = True
    
    # 결과 출력
    print("=" * 80)
    print("=== 엑셀 null 값 확인 결과 ===")
    print("=" * 80)
    
    # 1. 중간만 누락된 경우
    mid_only_dates = [date for date, result in date_results.items() if result['mid_only']]
    if mid_only_dates:
        print(f"\n[1개 누락 - 중간만 누락] ({len(mid_only_dates)}개 날짜)")
        for date in sorted(mid_only_dates):
            print(f"  {date}")
    
    # 2. 상단 또는 하단만 누락된 경우
    top_or_bottom_dates = [date for date, result in date_results.items() if result['top_or_bottom']]
    if top_or_bottom_dates:
        print(f"\n[1개 누락 - 상단 또는 하단만 누락] ({len(top_or_bottom_dates)}개 날짜)")
        for date in sorted(top_or_bottom_dates):
            print(f"  {date}")
    
    # 3. 2개 누락된 경우
    two_missing_dates = [date for date, result in date_results.items() if result['two_missing']]
    if two_missing_dates:
        print(f"\n[2개 누락] ({len(two_missing_dates)}개 날짜)")
        for date in sorted(two_missing_dates):
            print(f"  {date}")
    
    # 4. 3개 누락된 경우
    three_missing_dates = [date for date, result in date_results.items() if result['three_missing']]
    if three_missing_dates:
        print(f"\n[3개 누락] ({len(three_missing_dates)}개 날짜)")
        for date in sorted(three_missing_dates):
            print(f"  {date}")
    
    # 요약
    total_dates_with_null = len([d for d, r in date_results.items() if any(r.values())])
    print(f"\n{'=' * 80}")
    print(f"총 {total_dates_with_null}개 날짜 폴더에서 null 값이 발견되었습니다.")
    print("=" * 80)


if __name__ == "__main__":
    main()

