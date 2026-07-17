"""
Train EfficientNet-B0 on DTS video crops.
Labels: good/ folder = PASS (1), not_good/ folder = FAIL (0)

Roughly balanced: ~3042 good vs ~3465 not_good
No excel labels needed — folder structure IS the label.
"""
import torch, csv, warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from pathlib import Path
from PIL import Image
import time

FRAMES_ROOT = Path('/home/om/src/ip4r_v2/frames')
SAVE_DIR    = Path('/home/om/src/ip4r_v2/models')
SAVE_DIR.mkdir(exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 3e-4


class DTSDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.samples = []
        for lbl, folder in [(1, 'good'), (0, 'not_good')]:
            for p in sorted((root / folder).glob('*.jpg')):
                self.samples.append((str(p), lbl))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.0),   # LCD is not horizontally symmetric
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_ds = DTSDataset(FRAMES_ROOT)
n_total = len(full_ds)
n_val   = max(int(n_total * 0.15), 100)
n_train = n_total - n_val

train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))

train_ds.dataset.transform = train_tf   # hack: same transform object for all
# More correct: use Subset wrappers
# Let's just set transforms after split
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32)

train_loader = DataLoader(SubsetWithTransform(train_ds, train_tf),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(SubsetWithTransform(val_ds, val_tf),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f'Dataset: {n_total} total  (train={n_train}, val={n_val})')
print(f'  good={sum(1 for _,l in full_ds.samples if l==1)}  '
      f'not_good={sum(1 for _,l in full_ds.samples if l==0)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCEWithLogitsLoss()

best_val_acc = 0.0
log_rows = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        preds = (logits.sigmoid() >= 0.5).float()
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
        train_loss += loss.item() * labels.size(0)
    scheduler.step()

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            preds = (logits.sigmoid() >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_loss += loss.item() * labels.size(0)

    train_acc = train_correct / train_total
    val_acc   = val_correct / val_total
    dt = time.time() - t0
    print(f'Epoch {epoch:02d}/{EPOCHS}  '
          f'train_loss={train_loss/train_total:.4f}  train_acc={train_acc:.4f}  '
          f'val_loss={val_loss/val_total:.4f}  val_acc={val_acc:.4f}  '
          f'({dt:.1f}s)')
    log_rows.append([epoch, train_loss/train_total, train_acc,
                     val_loss/val_total, val_acc])

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'state_dict': model.state_dict(), 'val_acc': val_acc,
                    'epoch': epoch}, SAVE_DIR / 'dts_best.pth')
        print(f'  → New best saved (val_acc={val_acc:.4f})')

print(f'\nBest val_acc: {best_val_acc:.4f}')

with open(SAVE_DIR / 'dts_train_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch','train_loss','train_acc','val_loss','val_acc'])
    writer.writerows(log_rows)
print('Done.')
