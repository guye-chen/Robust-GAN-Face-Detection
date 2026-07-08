import os
import sys
import argparse
import time
import warnings
import numpy as np
from io import BytesIO
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from tqdm import tqdm

REAL_ROOT = "/请我手动填写/ffhq"
FAKE_ROOT = "/请我手动填写/fake"
WEIGHTS_PATH = "/请我手动填写/weights/inception_resnet_v2-940b1cd6.pth"
CHECKPOINT_DIR = "/请我手动填写/checkpoint/inceptionresnetv2"

GAN_NAMES = ["ProGAN", "StyleGAN2", "StyleGAN3"]
NUM_CLASSES = 2

# -------------------- timm check --------------------

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# -------------------- dataset --------------------

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(BytesIO(f.read())).convert('RGB')
    return img


class ServerFFHQGANDataset(Dataset):
    def __init__(self, real_root, fake_root, split, transform=None):
        self.real_root = real_root
        self.fake_root = fake_root
        self.split = split
        self.transform = transform
        self.samples = []
        self.real_count = 0

        real_dir = os.path.join(real_root, split)
        if os.path.isdir(real_dir):
            for fname in sorted(os.listdir(real_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    self.samples.append((os.path.join(real_dir, fname), 0, 'real'))
                    self.real_count += 1

        self.fake_count = 0
        for gan_name in GAN_NAMES:
            gan_dir = os.path.join(fake_root, split, gan_name)
            if os.path.isdir(gan_dir):
                for fname in sorted(os.listdir(gan_dir)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                        self.samples.append((os.path.join(gan_dir, fname), 1, gan_name))
                        self.fake_count += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, _ = self.samples[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def get_gan_source_counts(self):
        counts = {}
        for gan_name in GAN_NAMES:
            count = sum(1 for _, _, src in self.samples if src == gan_name)
            if count > 0:
                counts[gan_name] = count
        return counts

    def get_fake_by_source(self):
        result = {}
        for path, label, source in self.samples:
            if label == 1:
                result.setdefault(source, []).append((path, label, source))
        return result


class FakeSubsetDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, source = self.samples[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, source


# -------------------- dataset checks --------------------

def print_dataset_info(name, dataset, expected_real, expected_fake, expected_sources):
    print('  {}:'.format(name))
    print('    real = {}  (expected {})'.format(dataset.real_count, expected_real))
    print('    fake = {}  (expected {})'.format(dataset.fake_count, expected_fake))
    print('    total = {}  (expected {})'.format(len(dataset), expected_real + expected_fake))
    src_counts = dataset.get_gan_source_counts()
    for gan in GAN_NAMES:
        actual = src_counts.get(gan, 0)
        expected = expected_sources[gan]
        print('    {} = {}  (expected {})'.format(gan, actual, expected))

    assert dataset.real_count == expected_real, \
        '{} real count mismatch: {} != {}'.format(name, dataset.real_count, expected_real)
    assert dataset.fake_count == expected_fake, \
        '{} fake count mismatch: {} != {}'.format(name, dataset.fake_count, expected_fake)
    for gan in GAN_NAMES:
        actual = src_counts.get(gan, 0)
        expected = expected_sources[gan]
        assert actual == expected, \
            '{} {} count mismatch: {} != {}'.format(name, gan, actual, expected)


# -------------------- metrics --------------------

def compute_metrics(all_labels, all_preds, all_probs, num_classes=2):
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    auc = 0.0
    if num_classes == 2 and len(np.unique(all_labels)) > 1:
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            auc = 0.0
    cm = confusion_matrix(all_labels, all_preds)
    tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
    fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
    tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'auc': auc, 'confusion_matrix': cm, 'real_accuracy': real_acc,
        'fake_recall': fake_recall,
    }


# -------------------- AMP --------------------

try:
    from torch.amp import autocast, GradScaler
    _AMP = 'new'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _AMP = 'legacy'


def _amp_context(use_amp, device_type='cuda'):
    if _AMP == 'new':
        return autocast(device_type, enabled=use_amp)
    else:
        return autocast(enabled=use_amp)


def _make_scaler(use_amp, device_type='cuda'):
    if not use_amp:
        return None
    if _AMP == 'new':
        return GradScaler(device_type)
    else:
        return GradScaler()


def get_gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


# -------------------- training / validation --------------------

def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device,
                    accumulation_steps, use_amp, epoch, epochs, lr):
    model.train()
    all_preds, all_labels = [], []
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc='Train Epoch {}/{}'.format(epoch, epochs),
                dynamic_ncols=True, leave=True)

    for i, (inputs, labels_batch) in enumerate(pbar):
        inputs = inputs.to(device)
        labels_batch = labels_batch.to(device)

        with _amp_context(use_amp):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels_batch) / accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accumulation_steps
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss = total_loss / (i + 1)
        running_acc = accuracy_score(all_labels, all_preds)
        gpu_mem = get_gpu_mem()

        pbar.set_postfix({
            'loss': '{:.4f}'.format(running_loss),
            'acc': '{:.4f}'.format(running_acc),
            'lr': '{:.1e}'.format(lr),
            'gpu': '{:.1f}G'.format(gpu_mem),
        })

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    pbar.close()
    return avg_loss, acc


def validate(model, dataloader, loss_fn, device, desc='Val', epochs=None, epoch=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    label = desc
    if epochs is not None and epoch is not None:
        label = '{} Epoch {}/{}'.format(desc, epoch, epochs)

    pbar = tqdm(dataloader, desc=label, dynamic_ncols=True, leave=True)
    with torch.no_grad():
        for inputs, labels_batch in pbar:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels_batch)
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            running_acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({
                'loss': '{:.4f}'.format(loss.item()),
                'acc': '{:.4f}'.format(running_acc),
            })

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds, all_probs, NUM_CLASSES)
    metrics['loss'] = avg_loss
    return metrics


# -------------------- test --------------------

def test_model(model, test_dataset, val_transform, loss_fn, device):
    print('\n' + '=' * 70)
    print('TESTING ON HELD-OUT TEST SET')
    print('=' * 70)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    metrics = validate(model, test_loader, loss_fn, device, desc='Test')

    print('\nOverall Test Results:')
    print('  Accuracy:       {:.4f}'.format(metrics['accuracy']))
    print('  Precision:      {:.4f}'.format(metrics['precision']))
    print('  Recall:         {:.4f}'.format(metrics['recall']))
    print('  F1-score:       {:.4f}'.format(metrics['f1']))
    print('  AUC:            {:.4f}'.format(metrics['auc']))
    print('  Real Accuracy:  {:.4f}'.format(metrics['real_accuracy']))
    print('  Fake Recall:    {:.4f}'.format(metrics['fake_recall']))
    print('  Confusion Matrix:')
    print('    {}'.format(metrics['confusion_matrix'].tolist()))

    print('\nPer-GAN Fake Recall:')
    fake_by_source = test_dataset.get_fake_by_source()
    for gan_name in sorted(fake_by_source.keys()):
        gan_samples = fake_by_source[gan_name]
        gan_dataset = FakeSubsetDataset(gan_samples, transform=val_transform)
        gan_loader = DataLoader(gan_dataset, batch_size=8, shuffle=False, num_workers=4)
        correct = 0
        total = 0
        model.eval()
        pbar = tqdm(gan_loader, desc='  {}'.format(gan_name), dynamic_ncols=True, leave=True)
        with torch.no_grad():
            for inputs, labels_batch, _ in pbar:
                inputs = inputs.to(device)
                labels_batch = labels_batch.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == 1).sum().item()
                total += labels_batch.size(0)
                pbar.set_postfix({'rec': '{:.4f}'.format(correct / total if total > 0 else 0)})
        pbar.close()
        fake_recall_gan = correct / total if total > 0 else 0.0
        print('  {} Fake Recall: {:.4f} ({}/{})'.format(gan_name, fake_recall_gan, correct, total))
    print('=' * 70)
    return metrics


# -------------------- build model --------------------

def build_inceptionresnetv2():
    if not _TIMM_AVAILABLE:
        print('=' * 70)
        print('ERROR: timm is not installed.')
        print('Please run: pip install timm')
        print('=' * 70)
        sys.exit(1)

    MODEL_NAME = 'inception_resnet_v2'
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    except Exception as e:
        print('=' * 70)
        print('ERROR: timm model "{}" not found.'.format(MODEL_NAME))
        print('  {}'.format(e))
        print('  Please check available model names:')
        print('  python -c "import timm; print(timm.list_models(\'*inception*\'))"')
        print('=' * 70)
        sys.exit(1)

    if not os.path.exists(WEIGHTS_PATH):
        print('=' * 70)
        print('ERROR: InceptionResNetV2 pretrained weights not found.')
        print('Expected: {}'.format(os.path.abspath(WEIGHTS_PATH)))
        print('Please download and place it at the expected path.')
        print('=' * 70)
        sys.exit(1)

    print('Loading backbone from: {}'.format(WEIGHTS_PATH))
    pretrained = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)
    if isinstance(pretrained, dict) and 'state_dict' in pretrained:
        pretrained = pretrained['state_dict']
    model_state = model.state_dict()
    loaded, skipped = 0, []
    for k, v in pretrained.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(model_state)
    print('  Loaded {} params, skipped {}'.format(loaded, len(skipped)))

    if loaded == 0:
        print('ERROR: 0 backbone params loaded! Check WEIGHTS_PATH.')
        sys.exit(1)

    return model


# -------------------- main --------------------

def parse_args():
    parser = argparse.ArgumentParser(description='InceptionResNetV2 Server Training')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--accumulation_steps', default=4, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    for var_name, var_val in [('REAL_ROOT', REAL_ROOT), ('FAKE_ROOT', FAKE_ROOT),
                                ('WEIGHTS_PATH', WEIGHTS_PATH), ('CHECKPOINT_DIR', CHECKPOINT_DIR)]:
        if not os.path.exists(var_val):
            print('=' * 70)
            print('ERROR: {} does not exist: {}'.format(var_name, var_val))
            print('Please update the path at the top of this script.')
            print('=' * 70)
            sys.exit(1)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()

    effective_bs = args.batch_size * args.accumulation_steps

    print('=' * 70)
    print('Model: InceptionResNetV2 (timm)')
    print('REAL_ROOT:      {}'.format(REAL_ROOT))
    print('FAKE_ROOT:      {}'.format(FAKE_ROOT))
    print('WEIGHTS_PATH:   {}'.format(WEIGHTS_PATH))
    print('CHECKPOINT_DIR: {}'.format(CHECKPOINT_DIR))
    print('optimizer: Adam')
    print('learning_rate: {}'.format(args.lr))
    print('epochs: {}'.format(args.epochs))
    print('batch_size: {}'.format(args.batch_size))
    print('accumulation_steps: {}'.format(args.accumulation_steps))
    print('effective batch size: {}'.format(effective_bs))
    print('AMP: {}  (api: {})'.format(use_amp, _AMP))
    print('=' * 70)

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print('\nLoading datasets...')
    train_dataset = ServerFFHQGANDataset(REAL_ROOT, FAKE_ROOT, 'train', transform=train_transform)
    val_dataset = ServerFFHQGANDataset(REAL_ROOT, FAKE_ROOT, 'val', transform=val_transform)
    test_dataset = ServerFFHQGANDataset(REAL_ROOT, FAKE_ROOT, 'test', transform=val_transform)

    print('\nDataset counts:')
    print_dataset_info('train', train_dataset, 15000, 15000,
                       {'ProGAN': 5000, 'StyleGAN2': 5000, 'StyleGAN3': 5000})
    print_dataset_info('val', val_dataset, 4200, 4200,
                       {'ProGAN': 1400, 'StyleGAN2': 1400, 'StyleGAN3': 1400})
    print_dataset_info('test', test_dataset, 2100, 2100,
                       {'ProGAN': 700, 'StyleGAN2': 700, 'StyleGAN3': 700})

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_inceptionresnetv2()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: {:,}'.format(total_params))
    print('Trainable params: {:,}'.format(trainable_params))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = _make_scaler(use_amp)

    best_val_acc = 0.0
    best_epoch = 0
    no_improve_count = 0
    results_log = []

    print('\n' + '=' * 70)
    print('TRAINING START')
    print('=' * 70)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device,
            args.accumulation_steps, use_amp, epoch, args.epochs, args.lr)

        val_metrics = validate(model, val_loader, loss_fn, device,
                               desc='Val', epochs=args.epochs, epoch=epoch)

        elapsed = time.time() - start_time

        log_line = (
            'Epoch [{:2d}/{:2d}] | train_loss={:.4f} | train_acc={:.4f} | '
            'val_loss={:.4f} | val_acc={:.4f} | val_prec={:.4f} | '
            'val_rec={:.4f} | val_f1={:.4f} | val_auc={:.4f} | '
            'time={:.1f}s'
        ).format(epoch, args.epochs, train_loss, train_acc,
                 val_metrics['loss'], val_metrics['accuracy'],
                 val_metrics['precision'], val_metrics['recall'],
                 val_metrics['f1'], val_metrics['auc'], elapsed)
        print(log_line)
        results_log.append(log_line)

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print('  *** Saved best model (val_acc={:.4f}) ***'.format(best_val_acc))
        else:
            no_improve_count += 1
            print('  No improvement ({}/{})'.format(no_improve_count, args.patience))

        if no_improve_count >= args.patience:
            print('\nEarly stopping triggered after epoch {}'.format(epoch))
            break

    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }, os.path.join(CHECKPOINT_DIR, 'last_model.pth'))

    with open(os.path.join(CHECKPOINT_DIR, 'training_log.txt'), 'w', encoding='utf-8') as f:
        for line in results_log:
            f.write(line + '\n')

    print('\n' + '=' * 70)
    print('TRAINING FINISHED')
    print('Best epoch: {}  |  best val_acc: {:.4f}'.format(best_epoch, best_val_acc))
    print('Best model: {}/best_model.pth'.format(CHECKPOINT_DIR))
    print('Last model: {}/last_model.pth'.format(CHECKPOINT_DIR))
    print('Training log: {}/training_log.txt'.format(CHECKPOINT_DIR))
    print('=' * 70)

    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    test_model(model, test_dataset, val_transform, loss_fn, device)


if __name__ == '__main__':
    main()