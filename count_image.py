'''
전체 데이터 수를 출력하는 스크립트




# 이 밖의 자세한 사용법은 USAGE.md 파일을 참조하세요.
사용법:
    python count_image.py 0718 0725 hood trunklid frontfender
'''

import os
import sys
from pathlib import Path

def count_images_in_folder(folder_path):
    """폴더 내의 이미지 파일 개수를 세는 함수"""
    if not os.path.exists(folder_path):
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    count = 0
    
    # images 폴더가 있는지 확인
    images_folder = os.path.join(folder_path, "images")
    if os.path.exists(images_folder):
        # images 폴더 내의 이미지 개수 세기
        for file in os.listdir(images_folder):
            if os.path.isfile(os.path.join(images_folder, file)):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    count += 1
    else:
        # images 폴더가 없으면 직접 폴더 내의 이미지 개수 세기
        for file in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, file)):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    count += 1
    
    return count

def count_data_by_date_and_part(date, part):
    """지정된 날짜와 부위의 데이터 개수를 세는 함수"""
    base_dir = "/workspace/datasets"
    target_dir = os.path.join(base_dir, date, part)
    
    if not os.path.exists(target_dir):
        print(f"❌ 경로가 존재하지 않습니다: {target_dir}")
        return
    
    print(f"📅 날짜: {date}")
    print(f"🔧 부위: {part}")
    print(f"📁 경로: {target_dir}")
    print("-" * 50)
    
    # bad 폴더 처리
    bad_dir = os.path.join(target_dir, "bad")
    bad_count = 0
    if os.path.exists(bad_dir):
        bad_count = count_images_in_folder(bad_dir)
        print(f"🔴 bad 폴더: {bad_count}개 이미지")
    else:
        print(f"🔴 bad 폴더: 존재하지 않음")
    
    # good 폴더 처리
    good_dir = os.path.join(target_dir, "good")
    good_count = 0
    if os.path.exists(good_dir):
        good_count = count_images_in_folder(good_dir)
        print(f"🟢 good 폴더: {good_count}개 이미지")
    else:
        print(f"🟢 good 폴더: 존재하지 않음")
    
    # 총합
    total_count = bad_count + good_count
    print("-" * 50)
    print(f"📊 총 이미지 개수: {total_count}개")
    print(f"   - bad: {bad_count}개")
    print(f"   - good: {good_count}개")

def count_data_by_period(start_date, end_date, parts):
    """지정된 기간과 부위들의 모든 데이터 개수를 합산하는 함수"""
    base_dir = "/workspace/datasets"
    
    # parts가 문자열이면 리스트로 변환
    if isinstance(parts, str):
        parts = [parts]
    
    # 모든 날짜 폴더 가져오기
    all_dates = [d for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d)) and d not in ['TXT', '코드']]
    all_dates.sort()
    
    # 기간 내 날짜 필터링
    target_dates = []
    for date in all_dates:
        if start_date <= date <= end_date:
            target_dates.append(date)
    
    if not target_dates:
        print(f"❌ {start_date} ~ {end_date} 기간에 해당하는 날짜가 없습니다.")
        return
    
    print(f"📅 기간: {start_date} ~ {end_date}")
    print(f"🔧 부위: {', '.join(parts)}")
    print(f"📊 대상 날짜: {len(target_dates)}개")
    print("=" * 60)
    
    total_bad = 0
    total_good = 0
    valid_combinations = 0
    
    for date in target_dates:
        date_total_bad = 0
        date_total_good = 0
        date_valid_parts = 0
        
        for part in parts:
            target_dir = os.path.join(base_dir, date, part)
            
            if not os.path.exists(target_dir):
                continue
            
            # bad 폴더 처리
            bad_dir = os.path.join(target_dir, "bad")
            bad_count = 0
            if os.path.exists(bad_dir):
                bad_count = count_images_in_folder(bad_dir)
            
            # good 폴더 처리
            good_dir = os.path.join(target_dir, "good")
            good_count = 0
            if os.path.exists(good_dir):
                good_count = count_images_in_folder(good_dir)
            
            if bad_count > 0 or good_count > 0:
                date_total_bad += bad_count
                date_total_good += good_count
                date_valid_parts += 1
        
        if date_total_bad > 0 or date_total_good > 0:
            print(f"📅 {date}: bad {date_total_bad}개, good {date_total_good}개 (총 {date_total_bad + date_total_good}개)")
            total_bad += date_total_bad
            total_good += date_total_good
            valid_combinations += 1
        else:
            print(f"📅 {date}: 데이터 없음")
    
    print("=" * 60)
    print(f"📊 기간별 총합:")
    print(f"   - 처리된 날짜: {valid_combinations}개")
    print(f"   - bad 총합: {total_bad}개")
    print(f"   - good 총합: {total_good}개")
    print(f"   - 전체 총합: {total_bad + total_good}개")

def main():
    """메인 함수"""
    if len(sys.argv) < 3:
        print("최소 하나의 날짜와 하나의 부위를 입력하시오.")
        return
    
    if len(sys.argv) == 3:
        # 특정 날짜, 단일 부위
        date = sys.argv[1]
        part = sys.argv[2]
        count_data_by_date_and_part(date, part)
    elif len(sys.argv) == 4:
        # 기간별 합산, 단일 부위 또는 특정 날짜, 여러 부위
        # 두 번째 인수가 숫자 4자리면 기간별 합산으로 판단
        if len(sys.argv[2]) == 4 and sys.argv[2].isdigit():
            # 기간별 합산, 단일 부위
            start_date = sys.argv[1]
            end_date = sys.argv[2]
            part = sys.argv[3]
            count_data_by_period(start_date, end_date, part)
        else:
            # 특정 날짜, 여러 부위
            date = sys.argv[1]
            parts = sys.argv[2:]
            print(f"📅 날짜: {date}")
            print(f"🔧 부위: {', '.join(parts)}")
            print("=" * 60)
            
            total_bad = 0
            total_good = 0
            
            for part in parts:
                target_dir = os.path.join("/Users/csj/Downloads/CLMS/Backup", date, part)
                
                if not os.path.exists(target_dir):
                    print(f"⚠️  {part}: 폴더가 존재하지 않음")
                    continue
                
                # bad 폴더 처리
                bad_dir = os.path.join(target_dir, "bad")
                bad_count = 0
                if os.path.exists(bad_dir):
                    bad_count = count_images_in_folder(bad_dir)
                
                # good 폴더 처리
                good_dir = os.path.join(target_dir, "good")
                good_count = 0
                if os.path.exists(good_dir):
                    good_count = count_images_in_folder(good_dir)
                
                print(f"🔧 {part}: bad {bad_count}개, good {good_count}개 (총 {bad_count + good_count}개)")
                total_bad += bad_count
                total_good += good_count
            
            print("=" * 60)
            print(f"📊 부위별 총합:")
            print(f"   - bad 총합: {total_bad}개")
            print(f"   - good 총합: {total_good}개")
            print(f"   - 전체 총합: {total_bad + total_good}개")
    elif len(sys.argv) > 4:
        # 기간별 합산, 여러 부위
        start_date = sys.argv[1]
        end_date = sys.argv[2]
        parts = sys.argv[3:]
        count_data_by_period(start_date, end_date, parts)
    else:
        print("❌ 잘못된 인수 개수입니다.")
        print("사용법:")
        print("  1. 특정 날짜: python count.py <날짜> <부위1> [부위2] [부위3] ...")
        print("  2. 기간별 합산: python count.py <시작날짜> <끝날짜> <부위1> [부위2] [부위3] ...")

if __name__ == "__main__":
    main()
