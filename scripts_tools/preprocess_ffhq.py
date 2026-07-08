import os
import random
import shutil
from PIL import Image

random.seed(42)

SRC_DIR = r'E:\FFHQ'
DST_DIR = r'E:\FFHQ_off'
TARGET_SIZE = 256

TRAIN_COUNT = 15000
VAL_COUNT = 4200
TEST_COUNT = 2100

SPLITS = {
    'train': TRAIN_COUNT,
    'val': VAL_COUNT,
    'test': TEST_COUNT,
}

def main():
    all_files = sorted([
        f for f in os.listdir(SRC_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))
    ])
    print(f'Found {len(all_files)} images in {SRC_DIR}')

    assert len(all_files) >= TRAIN_COUNT + VAL_COUNT + TEST_COUNT, \
        f'Not enough images: need {TRAIN_COUNT + VAL_COUNT + TEST_COUNT}, got {len(all_files)}'

    random.shuffle(all_files)

    split_map = {}
    idx = 0
    for split_name, count in SPLITS.items():
        split_map[split_name] = all_files[idx:idx + count]
        idx += count

    for split_name, files in split_map.items():
        split_dir = os.path.join(DST_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        print(f'\nProcessing {split_name}: {len(files)} images -> {split_dir}')

        for i, fname in enumerate(files):
            src_path = os.path.join(SRC_DIR, fname)
            dst_path = os.path.join(split_dir, fname)

            img = Image.open(src_path).convert('RGB')
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            img.save(dst_path)

            if (i + 1) % 500 == 0:
                print(f'  {split_name}: {i + 1}/{len(files)} done')

        print(f'  {split_name}: all {len(files)} done')

        txt_path = os.path.join(DST_DIR, f'split_{split_name}_real.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for fname in files:
                f.write(fname + '\n')
        print(f'  Saved: {txt_path}')

    print(f'\nAll done! Images saved to {DST_DIR}/')


if __name__ == '__main__':
    main()