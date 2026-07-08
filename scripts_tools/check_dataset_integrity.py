import argparse
import hashlib
import os
from collections import Counter, defaultdict

try:
    from PIL import Image
except Exception:
    Image = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

REAL_EXPECTED = {
    "train": 15000,
    "val": 4200,
    "test": 2100,
}

FAKE_SOURCES = ["ProGAN", "StyleGAN2", "StyleGAN3"]
FAKE_EXPECTED_PER_SOURCE = {
    "train": 5000,
    "val": 1400,
    "test": 700,
}


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def list_image_files(root):
    if not os.path.isdir(root):
        return []
    files = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isfile(path) and is_image_file(path):
            files.append(path)
    return files


def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_image(path):
    if Image is None:
        return True, "PIL not installed; skipped image verification"
    try:
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except Exception as exc:
        return False, repr(exc)


def find_duplicate_values(items):
    counts = Counter(items)
    return sorted([item for item, count in counts.items() if count > 1])


def format_status(ok):
    return "OK" if ok else "FAIL"


def add_line(lines, text=""):
    print(text, flush=True)
    lines.append(text)


def check_group(name, files, expected_count, args, lines):
    basenames = [os.path.basename(p) for p in files]
    duplicate_names = find_duplicate_values(basenames)
    unreadable = []
    hashes = {}
    duplicate_hashes = []

    add_line(lines, "{}: checking {} files...".format(name, len(files)))
    for index, path in enumerate(files, 1):
        ok, message = verify_image(path)
        if not ok:
            unreadable.append((path, message))
        if not args.skip_hash:
            digest = sha256_file(path)
            hashes[path] = digest
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(files)):
            mode = "verify+hash" if not args.skip_hash else "verify"
            add_line(lines, "  progress: {}/{} ({})".format(index, len(files), mode))

    if not args.skip_hash:
        hash_to_paths = defaultdict(list)
        for path, digest in hashes.items():
            hash_to_paths[digest].append(path)
        duplicate_hashes = [
            paths for paths in hash_to_paths.values()
            if len(paths) > 1
        ]

    count_ok = len(files) == expected_count
    name_ok = len(duplicate_names) == 0
    read_ok = len(unreadable) == 0
    hash_ok = args.skip_hash or len(duplicate_hashes) == 0

    add_line(lines, "{}:".format(name))
    add_line(lines, "  count = {} (expected {}) [{}]".format(
        len(files), expected_count, format_status(count_ok)
    ))
    add_line(lines, "  duplicate filenames = {} [{}]".format(
        len(duplicate_names), format_status(name_ok)
    ))
    add_line(lines, "  unreadable images = {} [{}]".format(
        len(unreadable), format_status(read_ok)
    ))
    if args.skip_hash:
        add_line(lines, "  duplicate image content = skipped")
    else:
        add_line(lines, "  duplicate image content = {} [{}]".format(
            len(duplicate_hashes), format_status(hash_ok)
        ))

    for value in duplicate_names[:args.max_examples]:
        add_line(lines, "    duplicate filename example: {}".format(value))
    for path, message in unreadable[:args.max_examples]:
        add_line(lines, "    unreadable example: {} | {}".format(path, message))
    for paths in duplicate_hashes[:args.max_examples]:
        add_line(lines, "    duplicate content example:")
        for path in paths[:5]:
            add_line(lines, "      {}".format(path))

    return {
        "count_ok": count_ok,
        "name_ok": name_ok,
        "read_ok": read_ok,
        "hash_ok": hash_ok,
        "basenames": set(basenames),
        "hashes": set(hashes.values()),
        "files": files,
    }


def check_overlap(label, a_name, a_values, b_name, b_values, lines, max_examples):
    overlap = sorted(a_values & b_values)
    ok = len(overlap) == 0
    add_line(lines, "{} overlap {} vs {} = {} [{}]".format(
        label, a_name, b_name, len(overlap), format_status(ok)
    ))
    for item in overlap[:max_examples]:
        add_line(lines, "  overlap example: {}".format(item))
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Check FFHQ real and GAN fake dataset counts, duplicates, split overlap, and unreadable images."
    )
    parser.add_argument("--real_root", default="FFHQ_off",
                        help="Path to FFHQ_off directory containing train/val/test.")
    parser.add_argument("--fake_root", default="fake",
                        help="Path to fake directory containing train/val/test/ProGAN etc.")
    parser.add_argument("--skip_hash", action="store_true",
                        help="Skip SHA256 duplicate-content checking.")
    parser.add_argument("--report", default="dataset_integrity_report.txt",
                        help="Path to save the text report.")
    parser.add_argument("--max_examples", type=int, default=10,
                        help="Maximum number of examples to print for each problem.")
    parser.add_argument("--progress_every", type=int, default=500,
                        help="Print progress every N images. Set 0 to disable progress.")
    args = parser.parse_args()

    lines = []
    all_ok = True

    add_line(lines, "=" * 80)
    add_line(lines, "DATASET INTEGRITY CHECK")
    add_line(lines, "=" * 80)
    add_line(lines, "real_root = {}".format(os.path.abspath(args.real_root)))
    add_line(lines, "fake_root = {}".format(os.path.abspath(args.fake_root)))
    add_line(lines, "skip_hash = {}".format(args.skip_hash))
    add_line(lines, "PIL available = {}".format(Image is not None))
    add_line(lines)

    real_results = {}
    add_line(lines, "[REAL / FFHQ_off]")
    for split, expected in REAL_EXPECTED.items():
        split_dir = os.path.join(args.real_root, split)
        files = list_image_files(split_dir)
        result = check_group("real {}".format(split), files, expected, args, lines)
        real_results[split] = result
        all_ok = all_ok and all(result[k] for k in ["count_ok", "name_ok", "read_ok", "hash_ok"])
        add_line(lines)

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        all_ok = check_overlap(
            "real filename", a, real_results[a]["basenames"],
            b, real_results[b]["basenames"], lines, args.max_examples
        ) and all_ok
        if not args.skip_hash:
            all_ok = check_overlap(
                "real image-content", a, real_results[a]["hashes"],
                b, real_results[b]["hashes"], lines, args.max_examples
            ) and all_ok
    add_line(lines)

    fake_results = {}
    add_line(lines, "[FAKE / GAN]")
    for split, expected_each in FAKE_EXPECTED_PER_SOURCE.items():
        for source in FAKE_SOURCES:
            group_name = "fake {}/{}".format(split, source)
            group_dir = os.path.join(args.fake_root, split, source)
            files = list_image_files(group_dir)
            result = check_group(group_name, files, expected_each, args, lines)
            fake_results[(split, source)] = result
            all_ok = all_ok and all(result[k] for k in ["count_ok", "name_ok", "read_ok", "hash_ok"])
            add_line(lines)

    for source in FAKE_SOURCES:
        add_line(lines, "[FAKE SPLIT OVERLAP / {}]".format(source))
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            all_ok = check_overlap(
                "fake filename", a, fake_results[(a, source)]["basenames"],
                b, fake_results[(b, source)]["basenames"], lines, args.max_examples
            ) and all_ok
            if not args.skip_hash:
                all_ok = check_overlap(
                    "fake image-content", a, fake_results[(a, source)]["hashes"],
                    b, fake_results[(b, source)]["hashes"], lines, args.max_examples
                ) and all_ok
        add_line(lines)

    total_real = sum(len(real_results[split]["files"]) for split in REAL_EXPECTED)
    total_fake = sum(len(result["files"]) for result in fake_results.values())

    add_line(lines, "=" * 80)
    add_line(lines, "SUMMARY")
    add_line(lines, "=" * 80)
    add_line(lines, "total real images = {} (expected {})".format(
        total_real, sum(REAL_EXPECTED.values())
    ))
    add_line(lines, "total fake images = {} (expected {})".format(
        total_fake, sum(FAKE_EXPECTED_PER_SOURCE.values()) * len(FAKE_SOURCES)
    ))
    add_line(lines, "total images = {} (expected {})".format(
        total_real + total_fake,
        sum(REAL_EXPECTED.values()) + sum(FAKE_EXPECTED_PER_SOURCE.values()) * len(FAKE_SOURCES)
    ))
    add_line(lines, "final result = {}".format("PASS" if all_ok else "FAIL"))

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    add_line([], "Report saved to: {}".format(os.path.abspath(args.report)))

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
