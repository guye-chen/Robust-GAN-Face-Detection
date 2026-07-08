import os
import random
import shutil
from PIL import Image

random.seed(42)

SRC_BASE = r'E:\archive\GAN_Generated_Fake_Images'
DST_BASE = r'E:\archive\GAN_Generated_Fake_Images\fake'

GAN_SOURCES = ['ProGAN_256x256', 'StyleGAN2_256x256', 'StyleGAN3_256x256']
SPLITS = {
    'train': 5000,
    'val': 1400,
    'test': 700,
}
TOTAL_PER_SOURCE = 7100


def main():
    os.makedirs(DST_BASE, exist_ok=True)

    all_records = {split: [] for split in SPLITS}
    source_counts = {}
    split_counts = {split: 0 for split in SPLITS}

    for src_folder in GAN_SOURCES:
        src_path = os.path.join(SRC_BASE, src_folder)
        if not os.path.isdir(src_path):
            print(f'WARNING: {src_path} not found, skipping...')
            continue

        all_files = sorted([
            f for f in os.listdir(src_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))
        ])
        source_counts[src_folder] = len(all_files)

        assert len(all_files) >= TOTAL_PER_SOURCE, \
            f'{src_folder}: need {TOTAL_PER_SOURCE} images, found {len(all_files)}'

        random.shuffle(all_files)

        gan_label = src_folder.replace('_256x256', '')

        idx = 0
        for split_name, count in SPLITS.items():
            split_files = all_files[idx:idx + count]
            idx += count

            dst_dir = os.path.join(DST_BASE, split_name, gan_label)
            os.makedirs(dst_dir, exist_ok=True)

            for fname in split_files:
                src_file = os.path.join(src_path, fname)
                dst_file = os.path.join(dst_dir, fname)

                img = Image.open(src_file).convert('RGB')
                img.save(dst_file, 'PNG')

                all_records[split_name].append(f'{src_file},{dst_file},{gan_label}')
                split_counts[split_name] += 1

            print(f'  {src_folder} -> {split_name}/{gan_label}: {count} images')

    print('\n' + '=' * 60)
    print('Split Records')
    print('=' * 60)
    for split_name in ['train', 'val', 'test']:
        txt_path = os.path.join(DST_BASE, f'split_{split_name}_fake.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for record in all_records[split_name]:
                f.write(record + '\n')
        print(f'Saved: {txt_path}  ({len(all_records[split_name])} lines)')

    print('\n' + '=' * 60)
    print('Statistics')
    print('=' * 60)
    for src, count in source_counts.items():
        print(f'{src}: {count} images (original)')
    print()
    total_fake = 0
    for split_name in ['train', 'val', 'test']:
        c = split_counts[split_name]
        total_fake += c
        print(f'{split_name}/fake: {c}')
    print(f'\nTotal fake: {total_fake}')
    print(f'Output: {DST_BASE}')
    print(f'Random seed: 42')
    print('=' * 60)


if __name__ == '__main__':
    main()