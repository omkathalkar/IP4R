"""
Phase-2 DTS Model — Finetune from dts_best.pth
Dataset: frames_p2/good/ (PASS=1)  frames_p2/not_good/ (FAIL=0)
Strategy:
  - Load existing dts_best.pth weights (already knows DTS patterns)
  - Freeze backbone for first 3 epochs, unfreeze + lower LR for remaining
  - Strong brightness/contrast augmentation (real-world lighting variation)
  - Save best checkpoint as models/dts_p2_best.pth
"""
import torch, csv, warnings, time
warnings.filterwarnings('ignore')
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from pathlib import Path
from PIL import Image

FRAMES_ROOT   = Path('/home/om/src/ip4r_v2/frames_p2')
PRETRAIN_PATH = Path('/home/om/src/ip4r_v2/models/dts_best.pth')
SAVE_DIR      = Path('/home/om/src/ip4r_v2/models')

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 10
LR_HEAD    = 3e-4   # classifier head LR (backbone frozen)
LR_FULL    = 5e-5   # full model LR after unfreeze
UNFREEZE_EPOCH = 3  # unfreeze backbone after this epoch
VAL_SPLIT  = 0.15


class P2Dataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.samples = []
        for lbl, folder in [(1, 'good'), (0, 'not_good')]:
            p = root / folder
            for img_path in sorted(p.glob('*.jpg')):
                self.samples.append((str(img_path), lbl))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.1),
    transforms.RandomAffine(degrees=3, translate=(0.03, 0.03)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_ds = P2Dataset(FRAMES_ROOT)
n_total = len(full_ds)
n_val   = max(int(n_total * VAL_SPLIT), 50)
n_train = n_total - n_val
train_sub, val_sub = random_split(full_ds, [n_train, n_val],
                                  generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(SubsetWithTransform(train_sub, train_tf),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(SubsetWithTransform(val_sub, val_tf),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

n_good     = sum(1 for _, l in full_ds.samples if l == 1)
n_not_good = sum(1 for _, l in full_ds.samples if l == 0)
print(f'Dataset: {n_total} total  (good={n_good}, not_good={n_not_good})')
print(f'Train: {n_train}  Val: {n_val}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load model and finetune from existing checkpoint
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

if PRETRAIN_PATH.exists():
    ckpt = torch.load(PRETRAIN_PATH, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    print(f'Loaded pretrain from {PRETRAIN_PATH}  (prev val_acc={ckpt.get("val_acc", "?"):.4f})')
else:
    print('WARNING: pretrain not found, starting from random init')

model = model.to(device)

# Phase 1: freeze backbone, train head only
for p in model.features.parameters():
    p.requires_grad = False

optimizer  = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion  = nn.BCEWithLogitsLoss()

best_val_acc = 0.0
log_rows = []

for epoch in range(1, EPOCHS + 1):
    # Unfreeze backbone after UNFREEZE_EPOCH
    if epoch == UNFREEZE_EPOCH + 1:
        print(f'\n--- Epoch {epoch}: unfreezing backbone, LR → {LR_FULL} ---')
        for p in model.features.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - UNFREEZE_EPOCH)

    t0 = time.time()
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        preds = (logits.sigmoid() >= 0.5).float()
        tr_correct += (preds == labels).sum().item()
        tr_total   += labels.size(0)
        tr_loss    += loss.item() * labels.size(0)
    scheduler.step()

    model.eval()
    va_loss, va_correct, va_total = 0.0, 0, 0
    va_tp = va_fp = va_tn = va_fn = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)
            preds  = (logits.sigmoid() >= 0.5).float()
            va_correct += (preds == labels).sum().item()
            va_total   += labels.size(0)
            va_loss    += loss.item() * labels.size(0)
            va_tp += ((preds == 1) & (labels == 1)).sum().item()
            va_fp += ((preds == 1) & (labels == 0)).sum().item()
            va_tn += ((preds == 0) & (labels == 0)).sum().item()
            va_fn += ((preds == 0) & (labels == 1)).sum().item()

    tr_acc = tr_correct / tr_total
    va_acc = va_correct / va_total
    precision = va_tp / (va_tp + va_fp + 1e-9)
    recall    = va_tp / (va_tp + va_fn + 1e-9)
    dt = time.time() - t0

    print(f'Epoch {epoch:02d}/{EPOCHS}  '
          f'tr_loss={tr_loss/tr_total:.4f}  tr_acc={tr_acc:.4f}  '
          f'va_loss={va_loss/va_total:.4f}  va_acc={va_acc:.4f}  '
          f'prec={precision:.3f}  rec={recall:.3f}  ({dt:.1f}s)')
    log_rows.append([epoch, tr_loss/tr_total, tr_acc,
                     va_loss/va_total, va_acc, precision, recall])

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save({'state_dict': model.state_dict(), 'val_acc': va_acc,
                    'epoch': epoch}, SAVE_DIR / 'dts_p2_best.pth')
        print(f'  → Saved dts_p2_best.pth  (val_acc={va_acc:.4f})')

print(f'\nBest val_acc: {best_val_acc:.4f}')
with open(SAVE_DIR / 'dts_p2_train_log.csv', 'w', newline='') as f:
    import csv
    w = csv.writer(f)
    w.writerow(['epoch','tr_loss','tr_acc','va_loss','va_acc','precision','recall'])
    w.writerows(log_rows)
print('Done.')
