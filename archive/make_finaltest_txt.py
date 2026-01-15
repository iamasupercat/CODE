#!/usr/bin/env python3
"""
finaltest_Bolt.txt / finaltest_Door.txt 생성 스크립트

형식:
    /절대/경로/to/image.jpg <label>
에서 label은 0/1 (0: good, 1: bad)
quality는 폴더명(bad / good)을 기준으로 판단합니다.

- 대상 날짜: 1105, 1106, 1128
- Bolt: frontfender, hood, trunklid 에서 bad / good 이미지 모두 사용
- Door: frontdoor 에서 bad / good 이미지 모두 사용
"""

import os
from pathlib import Path

BASE_PATH = Path("/workspace/datasets")
DATES = ["1105", "1106", "1128"]

BOLT_PARTS = ["frontfender", "hood", "trunklid"]
DOOR_PARTS = ["frontdoor"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(date: str, part: str):
    """주어진 날짜/부품에서 bad/good 이미지를 수집하고 (경로, 라벨) 리스트 반환."""
    images = []
    date_dir = BASE_PATH / date / part
    if not date_dir.is_dir():
        return images

    for quality, label in (("bad", 1), ("good", 0)):
        q_dir = date_dir / quality
        if not q_dir.is_dir():
            continue

        for fname in os.listdir(q_dir):
            fpath = q_dir / fname
            if not fpath.is_file():
                continue
            if fpath.suffix.lower() not in IMG_EXTS:
                continue
            images.append((str(fpath.resolve()), label))

    return images


def main():
    txt_dir = Path("/workspace/datasets/CODE/TXT")
    txt_dir.mkdir(parents=True, exist_ok=True)

    bolt_out = txt_dir / "finaltest_Bolt.txt"
    door_out = txt_dir / "finaltest_Door.txt"

    # Bolt 수집 (경로만 사용, 라벨은 나중에 별도 처리)
    bolt_entries = []
    for d in DATES:
        for part in BOLT_PARTS:
            bolt_entries.extend(collect_images(d, part))

    # Door 수집 (경로만 사용, 라벨은 나중에 별도 처리)
    door_entries = []
    for d in DATES:
        for part in DOOR_PARTS:
            door_entries.extend(collect_images(d, part))

    # 정렬(안정적인 결과를 위해 경로 기준 정렬)
    bolt_entries.sort(key=lambda x: x[0])
    door_entries.sort(key=lambda x: x[0])

    # 파일에는 경로만 기록 (라벨 컬럼 제거)
    with open(bolt_out, "w") as f:
        for path, _label in bolt_entries:
            f.write(f"{path}\n")

    with open(door_out, "w") as f:
        for path, _label in door_entries:
            f.write(f"{path}\n")

    print(f"finaltest_Bolt.txt: {len(bolt_entries)}개 -> {bolt_out}")
    print(f"finaltest_Door.txt: {len(door_entries)}개 -> {door_out}")


if __name__ == "__main__":
    main()


