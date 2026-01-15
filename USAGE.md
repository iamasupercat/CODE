# 데이터셋 처리 스크립트 사용 가이드

이 문서는 `CODE` 디렉토리에 있는 모든 스크립트의 사용법을 정리한 가이드입니다.

---

## 📦 환경 설정

### 1. 가상환경 생성 및 활성화

```bash
cd /workspace/datasets/CODE

# # 가상환경 생성 (한 번만) <- 가상환경은 이미 생성해둠
# python3 -m venv venv

# 가상환경 활성화 (매번 실행 전에)
source venv/bin/activate
```

<!-- (가상환경에 이미 설치되어있음)
### 2. 패키지 설치

```bash
# requirements.txt 기반으로 모든 패키지 설치
pip install -r requirements.txt
```

**필요한 패키지:**
- `ultralytics` - YOLOv11 학습용
- `torch`, `torchvision` - PyTorch
- `Pillow`, `opencv-python` - 이미지 처리
- `pyyaml`, `tqdm`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `pandas`, `scikit-learn` - 유틸리티 -->

---

## 🚀 스크립트 사용법
## 목차

1. [이미지 크롭 스크립트](#이미지-크롭-스크립트)
2. [이미지 증강 스크립트](#이미지-증강-스크립트)
3. [데이터 분할 스크립트](#데이터-분할-스크립트)


4. [라벨 변환 스크립트](#라벨-변환-스크립트)
5. [통계 및 카운트 스크립트](#통계-및-카운트-스크립트)
6. [디버깅을 위한 시각화 스크립트](#디버깅을-위한-시각화-스크립트)
7. [유틸리티 스크립트](#유틸리티-스크립트)


8. [공통 사항](#공통-사항)
9. [작업 흐름 예시](#작업-흐름-예시)
10. [주의사항](#주의사항)
11. [업데이트 이력](#업데이트-이력)

---

## 이미지 크롭 스크립트

### 1. CropforBB.py - 볼트 이미지 크롭 (BB 방식)

YOLO OBB 라벨을 BB처럼 처리하여 볼트를 크롭하는 스크립트입니다.
볼트 크롭 시 사용합니다. (볼트 라벨링에는 회전을 적용하지 않음)

**기능:**
- 클래스 0(정측면), 1(측면)만 처리
- OBB 포맷이지만 angle을 무시하고 BB처럼 크롭
- angle이 0이 아닌 경우 문제로 감지 및 보고

**사용법:**
```bash
# 단일 날짜 지정
python CropforBB.py --target_dir 0616 --clean

# 날짜 범위 지정 (일반 폴더)
python CropforBB.py --date-range 0807 1109 --clean

# 일반 폴더 + OBB 폴더 (별도 날짜 범위)
python CropforBB.py \
    --date-range 0807 1109 \
    --obb-date-range 0616 0806 \
    --clean
```

**옵션:**
- `--target_dir`: 일반 폴더 날짜들 (예: 0616 0718 0721)
- `--date-range START END`: 일반 폴더 날짜 구간 (MMDD)
- `--obb-folders`: OBB 폴더 날짜들 (예: 0718 0806)
- `--obb-date-range START END`: OBB 폴더 날짜 구간 (MMDD)
- `--clean`: 실행 전 기존 crop_bolt, crop_bolt_aug, debug_crop 폴더 삭제

**출력:**
- `{날짜}/{subfolder}/{bad|good}/crop_bolt/{0|1}/`: 크롭된 이미지
- `{날짜}/{subfolder}/{bad|good}/debug_crop/`: 디버그 이미지

---

### 2. CropforOBB.py - 도어 이미지 크롭 (OBB 방식)

YOLO OBB 라벨을 이용하여 회전된 객체를 크롭하는 스크립트입니다.
앞도어 크롭 시 사용합니다. (앞ㄷ어 라벨링에는 회전을 적용함)

**기능:**
- Door 모드: 앞도어 이미지를 크롭하여 high/mid/low 폴더의 하위폴더(0,1,2,3,4)에 저장
- 엑셀 파일(`frontdoor.xlsx`)을 읽어서 각 영역별 클래스 번호 결정
- YOLO 학습 중 `.bak` 파일 우선 읽기 지원
    * 우리의 obb txt 형식과 yolo 학습을 위해 필요한 obb 형식이 다름. yolo 학습 시, yolo 학습을 위한 형식으로 일시적으로 변환하기 위해 원본파일 내용을 .bak에 복사해넣고 원본파일을 수정하여 yolo 학습을 수행함.
    * yolo 학습 중에 원본 txt 파일에 접근하면 수정되어있는 내용에 접근하게 되기 때문에 .bak 파일을 읽어야 함.

**사용법:**
```bash
# 단일 날짜 지정
python CropforOBB.py --target_dir 0805 --mode door --clean

# 날짜 범위 지정 (일반 폴더)
python CropforOBB.py --date-range 0807 1103 --mode door

# 일반 폴더 + OBB 폴더 (별도 날짜 범위)
python CropforOBB.py \
    --date-range 0807 1109 \
    --obb-date-range 0616 0806 \
    --mode door \
    --clean
```

**옵션:**
- `--target_dir`: 일반 폴더 날짜들 (예: 0616 0718 0721)
- `--date-range START END`: 일반 폴더 날짜 구간 (MMDD)
- `--obb-folders`: OBB 폴더 날짜들 (예: 0718 0806)
- `--obb-date-range START END`: OBB 폴더 날짜 구간 (MMDD)
- `--mode`: `door` 또는 `bolt` 선택
- `--clean`: 실행 전 기존 crop_*, debug_crop 폴더 삭제

**출력:**
- Door 모드: `{날짜}/frontdoor/{bad|good}/crop_high/{0|1|2|3|4}/`, `crop_mid/`, `crop_low/`
- Bolt 모드: `{날짜}/{subfolder}/{bad|good}/crop_bolt/{0|1}/`
- `{날짜}/{subfolder}/{bad|good}/debug_crop/`: 디버그 이미지

---

## 이미지 증강 스크립트

### 3. Augmentation.py - 원본 이미지 증강

원본 이미지에 3가지 증강 기법을 적용하는 스크립트입니다.

**기능:**
- 노이즈 추가 (Gaussian noise)
- 색상 반전 (Color inversion)
- 좌우 반전 (Horizontal flip) - OBB 각도 보정 포함
- OBB 폴더 자동 지원

**사용법:**
```bash
# 단일 날짜
python Augmentation.py 1103 --subfolder hood

# 날짜 구간 (기본 경로 + OBB 폴더 모두 검색)
python Augmentation.py --date-range 0616 1109 --subfolder frontdoor

# OBB 폴더만 처리
python Augmentation.py --date-range 0616 0718 --subfolder trunklid hood --obb-only
```

**옵션:**
- `folders`: 입력 폴더 이름들 (예: 0718 0721)
- `--date-range START END`: 날짜 구간 선택 (MMDD 또는 YYYYMMDD)
- `--subfolder`: 처리할 서브폴더들 (예: frontdoor hood)
- `--obb-only`: OBB 폴더만 처리 (기본 경로 제외)

**출력:**
- `{날짜}/{subfolder}_aug/{bad|good}/images/`: 증강 이미지
- `{날짜}/{subfolder}_aug/{bad|good}/labels/`: 증강 라벨

**증강 기법:**
- `_noise.jpg`: 가우시안 노이즈 추가
- `_invert.jpg`: 색상 반전
- `_flip.jpg`: 좌우 반전 (OBB 각도 보정: `angle' = π - angle`)

---

### 4. AugforBolt_crop.py - 볼트 크롭 이미지 증강

크롭된 볼트 이미지에 6가지 증강 기법을 적용하는 스크립트입니다.

**기능:**
- 각 이미지당 6개의 증강된 이미지 생성
- 증강 기법: rot, flip, noise, gray, bright, contrast

**사용법:**
```bash
# 개별 날짜 지정
python AugforBolt_crop.py \
    --target_dir 0616 0718 0721 0725 0728 0729 0731 0801 0804 0805 0806 \
    --subfolders frontfender hood trunklid \
    --set_types bad good

# 날짜 범위 지정
python AugforBolt_crop.py \
    --date-range 0807 1013 \
    --subfolders frontfender hood trunklid \
    --set_types bad

# 일반 폴더 + OBB 폴더 (별도 날짜 범위)
python AugforBolt_crop.py \
    --obb-date-range 0616 0806 \
    --subfolders frontfender hood trunklid \
    --set_types good
```

**옵션:**
- `--target_dir`: 대상 폴더 날짜들 (예: 0616 0718 0721, 일반 폴더)
- `--date-range START END`: 일반 폴더 날짜 구간 (MMDD)
- `--obb-folders`: OBB 폴더 날짜들 (예: 0718 0806)
- `--obb-date-range START END`: OBB 폴더 날짜 구간 (MMDD)
- `--subfolders`: 처리할 서브폴더들 (기본값: hood_revised trunk_revised front_revised)
- `--set_types`: 처리할 set 타입들 (기본값: bad good)

**출력:**
- `{날짜}/{subfolder}/{bad|good}/crop_bolt_aug/{0|1}/`: 증강된 크롭 이미지

**증강 기법:**
- `_rot.jpg`: 랜덤 회전 (-180° ~ +180°)
- `_flip.jpg`: 수평 뒤집기
- `_noise.jpg`: 가우시안 노이즈 (mean 0, std 10)
- `_gray.png`: 그레이스케일 변환
- `_bright.jpg`: 밝기 증가 (+40, HSV의 V채널)
- `_contrast.jpg`: 대비 증가 (1.5배, HSV의 S채널)

---

### 5. AugforDoor_crop.py - 도어 크롭 이미지 증강

크롭된 도어 이미지에 증강 기법을 적용하는 스크립트입니다.

**기능:**
- 각 이미지당 증강된 이미지 생성
- 증강 기법: rot, flip, noise, gray, bright, contrast
- 볼트와의 차이점: rot 범위가 -10° ~ +10° (볼트는 -180° ~ +180°)

**사용법:**
```bash
# 개별 날짜 지정
python AugforDoor_crop.py \
    --target_dir 0616 0721 0728 0729 0731 0801 0804 0805 0806 \
    --subfolders frontdoor \
    --set_types good bad

# 날짜 범위 지정
python AugforDoor_crop.py \
    --date-range 0807 1103 \
    --subfolders frontdoor \
    --set_types bad

# 일반 폴더 + OBB 폴더
python AugforDoor_crop.py \
    --date-range 0807 1109 \
    --obb-date-range 0616 0806 \
    --subfolders frontdoor \
    --set_types good bad
```

**옵션:**
- `--target_dir`: 대상 폴더 날짜들 (예: 0616 0721 0728)
- `--date-range START END`: 날짜 구간 선택 (MMDD)
- `--obb-folders`: OBB 폴더 날짜들 (예: 0718 0806)
- `--obb-date-range START END`: OBB 폴더 날짜 구간 (MMDD)
- `--subfolders`: 처리할 서브폴더들 (기본값: frontdoor)
- `--set_types`: 처리할 set 타입들 (기본값: bad good)

**출력:**
- `{날짜}/{subfolder}/{bad|good}/crop_high_aug/{0|1|2|3}/`: 증강된 크롭 이미지
- `{날짜}/{subfolder}/{bad|good}/crop_mid_aug/{0|1|2|3}/`
- `{날짜}/{subfolder}/{bad|good}/crop_low_aug/{0|1|2|3}/`

**증강 기법:**
- `_rot.jpg`: 랜덤 회전 (-10° ~ +10°)
- `_flip.jpg`: 수평 뒤집기
- `_noise.jpg`: 가우시안 노이즈 추가
- `_gray.png`: 그레이스케일 변환
- `_bright.jpg`: 밝기 증가 (+40)
- `_contrast.jpg`: 대비 증가 (1.5배)

---

## 데이터 분할 스크립트

### 6. UnifiedSplit.py - YOLO/DINO 통합 split 생성 (최신 통합 버전)

YOLO와 DINO 훈련용 split을 **동일한 기준(원본 이미지 ID)** 으로 한 번에 생성하는 통합 스크립트입니다.  
기존 `YOLOsplit.py` + `DINOsplit.py`(bolt/door_area 모드)의 역할을 합친 최신 버전입니다.

**모드:**
1. **Bolt 모드** (`--mode bolt`):
   - YOLO: 원본 + 증강 이미지 (Augmentation, AugforBolt_crop 등) 포함
   - DINO: `crop_bolt`, `crop_bolt_aug` 기반 2-class 또는 4-class 라벨링
2. **Door 모드** (`--mode door`):
   - YOLO: 원본(앞도어) + 증강 이미지 포함
   - DINO: `crop_high[_aug]`, `crop_mid[_aug]`, `crop_low[_aug]` 를 모두 모아 frontdoor 전체를 하나의 분류 문제로 사용

**사용법 (예시):**
```bash
cd /workspace/datasets/CODE

# Bolt 모드 (4클래스, YOLO+DINO 통합 split)
python UnifiedSplit.py \
    --mode bolt \
    --bolt-4class \
    --bad-date-range 0807 1013 \
    --good-date-range 0616 1103 \
    --subfolders frontfender hood trunklid \
    --name Bolt_4class

# Door 모드 (앞도어 전체, 0~4 클래스 그대로 사용)
python UnifiedSplit.py \
    --mode door \
    --date-range 0807 1109 \
    --obb-date-range 0616 0806 \
    --subfolders frontdoor \
    --name Door
```

**주요 옵션:**
- `--mode {bolt,door}`: 모드 선택 (필수)
- `--folders`: 분석할 기본 폴더 날짜들 (예: 0718 0721)
- `--date-range START END`: 일반 폴더 날짜 구간 (MMDD)
- `--obb-folders`: OBB 폴더 날짜들
- `--obb-date-range START END`: OBB 폴더 날짜 구간
- `--subfolders`: 찾을 하위폴더들 (예: frontfender hood trunklid, 또는 frontdoor)
- `--name`: 출력 파일명에 사용할 이름 (예: Bolt_4class, Door)
- `--bolt-4class`: Bolt 모드에서 4클래스 라벨 사용 (0:정측면 양품, 1:정측면 불량, 2:측면 양품, 3:측면 불량)
- `--merge-classes`: Door 모드에서 DINO 라벨 1,2,3을 1로 합쳐 2-class(0:good, 1:defect)로 학습

**출력:**
- YOLO용:
  - `TXT/train_{name}.txt`
  - `TXT/val_{name}.txt`
  - `TXT/test_{name}.txt`
- DINO용:
  - `TXT/train_dino_{name}.txt`
  - `TXT/val_dino_{name}.txt`
  - `TXT/test_dino_{name}.txt`

**분할 특징:**
- 원본 이미지 기준으로 **분할 키(image_id)** 를 만든 뒤, 해당 키에 속한 모든 증강/크롭 이미지를 같은 split에 배치
- good/bad 비율을 고려한 stratified split (기존 `[수정]bolt_split.py`, `[수정]sealing_split.py`의 로직 계승)
- DINO 증강(`*_aug`) 이미지는 train split에만 추가

---

## 라벨 변환 스크립트

### 7. xml_to_txt_obb.py - XML을 OBB TXT로 변환

LabelImg XML 파일을 YOLO OBB 형식의 TXT 파일로 변환하는 스크립트입니다.

**사용법:**
```bash
# 단일 날짜
python xml_to_txt_obb.py 0718

# 날짜 범위
python xml_to_txt_obb.py --date-range 0718 0806
```

**옵션:**
- `folders`: 변환할 날짜 폴더들 (예: 0718 0806)
- `--date-range START END`: 날짜 구간 선택 (MMDD)

**출력:**
- `{날짜}/{subfolder}/{bad|good}/labels/*.txt`: YOLO OBB 형식 라벨 파일

**형식:**
XML (robndbox 형식):
    <size>
        <width>3000</width>
        <height>4000</height>
    </size>
    <object>
        <name>bolt_frontside</name>
        <robndbox>
            <cx>1786.0</cx>
            <cy>1669.0</cy>
            <w>138.0</w>
            <h>120.0</h>
            <angle>0.0</angle>
        </robndbox>
    </object>

YOLO OBB 형식:
    class_id x_center y_center width height angle
    (모든 값은 0~1 범위로 정규화)

---

### 8. bb_to_obb.py - BB/OBB 형식 변환

YOLO 라벨 파일을 BB(5개 값)와 OBB(6개 값) 모드 간 전환하는 스크립트입니다.

**사용법:**
```bash
# 전체를 BB 형식(5개 값)으로 변환
python bb_to_obb.py --to-bb

# 전체를 OBB 형식(6개 값)으로 변환
python bb_to_obb.py --to-obb
```

**옵션:**
- `--to-bb`: 모든 라인을 BB 형식(5개 값)으로 변환 (6개 값이면 마지막 값 제거)
- `--to-obb`: 모든 라인을 OBB 형식(6개 값)으로 변환 (5개 값이면 0 추가)

**처리 범위:** 0718부터 0806까지의 폴더 자동 처리

---

## 통계 및 카운트 스크립트

### 9. count_bolt.py - 볼트 개수 세기

labels 폴더의 txt 파일에서 클래스 0 또는 1인 경우를 볼트로 카운팅합니다.

**사용법:**
```bash
# 단일 날짜, 단일 부위
python count_bolt.py 0718 hood

# 단일 날짜, 여러 부위
python count_bolt.py 0718 hood trunklid frontfender

# 기간별, 단일 부위
python count_bolt.py 0718 0725 hood

# 기간별, 여러 부위
python count_bolt.py 0718 0725 hood trunklid frontfender
```

**인수:**
- 첫 번째: 날짜 또는 시작 날짜
- 두 번째: 끝 날짜 (기간별인 경우) 또는 부위
- 나머지: 부위들

---

### 10. count_door.py - 도어 클래스 개수 세기

frontdoor 폴더에서 frontdoor.xlsx 파일을 읽어서 상단, 중간, 하단 각각의 클래스 개수와 퍼센테이지를 출력합니다.

**사용법:**
```bash
# 단일 날짜
python count_door.py 1031

# 기간별
python count_door.py 0807 1109
```

**인수:**
- 첫 번째: 날짜 또는 시작 날짜
- 두 번째: 끝 날짜 (기간별인 경우)

**출력:**
- 각 영역(상단, 중간, 하단)별 클래스(1,2,3,4) 개수 및 퍼센테이지
- good 폴더의 이미지 총 개수
- good 포함 통계

---

### 11. count_image.py - 이미지 개수 세기

특정 기간의 데이터 개수를 세는 스크립트입니다.

**사용법:**
```bash
# 단일 날짜, 단일 부위
python count_image.py 0718 hood

# 단일 날짜, 여러 부위
python count_image.py 0718 hood trunklid frontfender

# 기간별, 단일 부위
python count_image.py 0807 1016 frontdoor

# 기간별, 여러 부위
python count_image.py 0718 0725 hood trunklid frontfender
```

**인수:**
- 첫 번째: 날짜 또는 시작 날짜
- 두 번째: 끝 날짜 (기간별인 경우) 또는 부위
- 나머지: 부위들

---

### 12. check_excel_null.py - 엑셀 null 값 확인

frontdoor.xlsx 파일에서 quality가 null인 행에 대해 상단, 중간, 하단 열의 null 값을 확인하는 스크립트입니다.

**기능:**
- quality가 null인 행만 체크
- 상단, 중간, 하단 열의 null 패턴을 분류:
  - 1개 누락: 중간만 누락 / 상단 또는 하단만 누락
  - 2개 누락: 어떤 조합이든 모두 묶음
  - 3개 누락: 모두 누락
- 날짜 폴더만 출력 (이미지 파일명은 출력하지 않음)

**사용법:**
```bash
# 날짜 범위 지정
python check_excel_null.py --date-range 0616 1109

# 일반 폴더 + OBB 폴더
python check_excel_null.py --date-range 0807 1109 --obb-date-range 0616 0806

# 개별 날짜 지정
python check_excel_null.py --folders 0807 0808 0809
```

**옵션:**
- `--target_dir`: 일반 폴더 날짜들 (예: 0616 0718 0721)
- `--date-range START END`: 일반 폴더 날짜 구간 (MMDD)
- `--obb-folders`: OBB 폴더 날짜들 (예: 0718 0806)
- `--obb-date-range START END`: OBB 폴더 날짜 구간 (MMDD)

**출력:**
- `[1개 누락 - 중간만 누락]`: 중간만 null인 경우
- `[1개 누락 - 상단 또는 하단만 누락]`: 상단 또는 하단 중 하나만 null인 경우
- `[2개 누락]`: 2개가 null인 경우
- `[3개 누락]`: 3개 모두 null인 경우

각 카테고리별로 해당하는 날짜 폴더 목록이 출력됩니다.

---

## 디버깅을 위한 시각화 스크립트

### 13. debug_aug.py - 증강 결과 시각화

Augmentation 결과를 시각화하는 디버그 스크립트입니다.

**사용법:**
```bash
# 절대 경로
python debug_aug.py /workspace/datasets/0718/frontfender/bad/images/bad_0216_1_1_5a269cdf-35fe-4259-a086-7ccab92112ae.jpg

# 상대 경로 (datasets 폴더에서 실행)
cd /workspace/datasets
python CODE/debug_aug.py 0718/frontfender/bad/images/bad_0216_1_1_5a269cdf-35fe-4259-a086-7ccab92112ae.jpg
```

**인수:**
- 원본 이미지 경로

**출력:**
- 화면에 원본 + 증강 이미지들 표시
- `{원본폴더}/debug_aug_output/{이미지명}_debug.png` 파일로 저장
- 바운딩 박스 색상: 원본(초록), noise(빨강), invert(파랑), flip(노랑)

---

### 14. debug_crop.py - 크롭 결과 시각화

CropforBB.py의 크롭 결과를 시각화하는 디버그 스크립트입니다.

**사용법:**
```bash
# 절대 경로
python debug_crop.py /workspace/datasets/1010/frontfender/good/images/good_8128_1_13_62c4b142-e159-4c33-9368-98c3502cc696.jpg

# 상대 경로
cd /workspace/datasets
python CODE/debug_crop.py 1010/trunklid/good/images/good_8128_1_13_62c4b142-e159-4c33-9368-98c3502cc696.jpg
```

**인수:**
- 원본 이미지 경로

**출력:**
- 화면에 원본 이미지 + OBB 박스 + 크롭 영역 표시
- `{원본폴더}/debug_crop_visual/{이미지명}_debug.png` 파일로 저장
- 바운딩 박스 색상:
  - 초록색: OBB (angle=0, 정상)
  - 노란색: OBB (angle≠0, 문제)
  - 파란색: 실제 크롭된 영역 (BB)

---

### 15. debug_xml_txt.py - XML/TXT 변환 결과 시각화

XML → TXT 변환 결과를 시각화하는 디버그 스크립트입니다.

**사용법:**
```bash
# 절대 경로
python debug_xml_txt.py /workspace/datasets/OBB/0718/hood/good/images/good_9362_1_e97d8f94-e26c-455d-9c96-40a63f6a2c83.jpg

# 상대 경로
cd /workspace/datasets
python CODE/debug_xml_txt.py OBB/0718/hood/good/images/good_9362_1_e97d8f94-e26c-455d-9c96-40a63f6a2c83.jpg
```

**인수:**
- 이미지 경로

**출력:**
- 화면에 원본 이미지 + XML 바운딩 박스 + TXT 바운딩 박스 표시
- `{labels폴더}/debug_xml_txt_output/{이미지명}_debug.png` 파일로 저장
- 바운딩 박스 색상: XML(빨강), TXT(초록)

---

## 유틸리티 스크립트

### 16. find_missing_labels.py - 라벨 파일 누락 검색

매칭되는 txt 파일이 없는 이미지들의 리스트를 추출하는 스크립트입니다.

**사용법:**
```bash
# 날짜 범위 지정
python find_missing_labels.py \
    --date-range 0807 1109 \
    --subfolders frontfender hood trunklid \
    --output missing_labels.txt
```

**옵션:**
- `--date-range START END`: 일반 폴더 날짜 구간 (MMDD)
- `--obb-date-range START END`: OBB 폴더 날짜 구간 (MMDD)
- `--subfolders`: 찾을 하위폴더들 (필수)
- `--output`: 출력 파일 경로 (기본값: missing_labels.txt)

**출력:**
- 이미지 경로와 예상 라벨 경로를 `|`로 구분하여 저장

---

### 17. find_missing_bolt_classes.py - Bolt 클래스 누락 검색

Bolt 모드에서 txt 파일 내에 클래스 0 또는 1이 아예 없는 경우를 찾는 스크립트입니다.

**사용법:**
```bash
# 날짜 범위 지정
python find_missing_bolt_classes.py \
    --date-range 0807 1109 \
    --subfolders frontfender hood trunklid \
    --output missing_bolt_classes.txt
```

**옵션:**
- `--date-range START END`: 일반 폴더 날짜 구간 (MMDD)
- `--obb-date-range START END`: OBB 폴더 날짜 구간 (MMDD)
- `--subfolders`: 찾을 하위폴더들 (필수, frontfender, hood, trunklid)
- `--output`: 출력 파일 경로 (기본값: missing_bolt_classes.txt)

**출력:**
- 이미지 경로, 라벨 경로, 발견된 클래스들을 저장
- 클래스별 통계 제공

---

### 18. verify_unified_split.py - Split 검증

UnifiedSplit.py로 생성된 YOLO와 DINO용 TXT 파일들의 고유 ID 일치 여부를 검증하는 스크립트입니다.

**사용법:**
```bash
# Bolt 모드
python verify_unified_split.py --name Bolt

# Door 모드 (high 영역만)
python verify_unified_split.py --name Door --area high

# Door 모드 (모든 영역)
python verify_unified_split.py --name Door --all-areas
```

**옵션:**
- `--name`: 검증할 split 이름
- `--area`: Door 모드에서 검증할 영역 (high, mid, low)
- `--all-areas`: Door 모드에서 모든 영역 검증

**출력:**
- YOLO와 DINO split 간 ID 일치 여부 확인
- 불일치하는 ID 목록 출력

---

## Archive 폴더

`archive/` 폴더는 더 이상 사용하지 않는 코드 파일들을 백업용으로 보관하고 있는 폴더입니다.  
현재는 `UnifiedSplit.py`가 기존의 `YOLOsplit.py`, `DINOsplit.py`, `bolt_split.py`, `sealing_split.py` 등의 기능을 통합하여 대체하고 있습니다.

---

## 공통 사항

### 날짜 형식
- **4자리 (MMDD)**: 예: `0616`, `1109`

### 기본 경로
- 일반 폴더: `/workspace/datasets/{날짜}/`
- OBB 폴더: `/workspace/datasets/OBB/{날짜}/`

### 출력 디렉토리
- split 스크립트는 `TXT/` 디렉토리에 결과 파일을 저장합니다.
- 디렉토리가 없으면 자동으로 생성됩니다.

### 데이터 누수 방지
- 모든 split 스크립트는 `original_image_id`를 기준으로 분할하여 같은 원본 이미지(및 그 변형)가 여러 split에 들어가지 않도록 합니다.

### 클래스 비율 유지
- 모든 split 스크립트는 stratified split을 사용하여 클래스 비율을 유지합니다.

---

## 작업 흐름 예시

### 볼트 데이터 처리 흐름
1. **라벨 변환**: `xml_to_txt_obb.py` (필요시)
2. **크롭**: `CropforBB.py`
3. **크롭 증강**: `AugforBolt_crop.py` 과 `Augmentation.py`
5. **Split 생성**: `DINOsplit_bolt.py` 과 `YOLOsplit.py`

### 도어 데이터 처리 흐름
1. **라벨 변환**: `xml_to_txt_obb.py` (필요시)
2. **크롭**: `CropforOBB.py`
3. **크롭, 원본 증강**: `AugforDoor_crop.py` 과 `Augmentation.py`
5. **Split 생성**: `DINOsplit_door.py` 과 `YOLOsplit.py`

---

## 주의사항

1. **YOLO 학습 중**: YOLO 학습 중에는 `.txt` 파일이 임시로 `xyxyxyxy` 형식으로 변환될 수 있습니다. 이 경우 `.bak` 파일을 우선 읽습니다.

2. **데이터 삭제**: `--clean` 옵션을 사용하면 기존 결과물이 삭제됩니다. 주의하세요.

3. **엑셀 파일**: 도어 크롭 시 `frontdoor.xlsx` 파일이 필요합니다. 파일이 없으면 크롭이 건너뛰어질 수 있습니다.

4. **폴더 구조**: 각 스크립트는 특정 폴더 구조를 가정합니다. 폴더 구조가 맞지 않으면 오류가 발생할 수 있습니다.

---

## 문제 해결

### 스크립트 실행 권한
```bash
chmod +x /workspace/datasets/CODE/*.py
```

### Python 경로 확인
```bash
which python3
python3 --version
```

### 의존성 확인
필요한 패키지:
- `opencv-python` (cv2)
- `numpy`
- `pandas`
- `Pillow` (PIL)
- `matplotlib` (디버그 스크립트용)

---

## 업데이트 이력

- 2024-12-22: 초기 문서 작성