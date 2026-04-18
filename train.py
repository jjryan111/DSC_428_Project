import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

class AuthorshipDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.texts  = df["text"].tolist()
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    @classmethod
    def from_lists(cls, texts: list[str], labels: list):
        df = pd.DataFrame({"text": texts, "label": labels})
        return cls(df)


def train(
    model,
    train_texts, train_labels,
    val_texts,   val_labels,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(AuthorshipDataset.from_lists(train_texts, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AuthorshipDataset.from_lists(val_texts, val_labels), batch_size=batch_size)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs
    )

    # Warmup: freeze transformer for first 2 epochs if the model has one
    has_backbone = hasattr(model, "freeze_backbone")

    for epoch in range(epochs):
        if has_backbone:
            if epoch < 2:
                model.freeze_backbone()
            else:
                model.unfreeze_backbone()

        model.train()
        for texts, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(list(texts))
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation F1 + threshold sweep
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for texts, labels in val_loader:
                all_logits.append(model(list(texts)).cpu())
                all_labels.append(labels)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        best_f1, best_thresh = 0.0, 0.5
        for t in torch.arange(0.2, 0.8, 0.05):
            preds = (torch.sigmoid(logits) > t).float()
            tp = ((preds == 1) & (labels == 1)).sum().item()
            fp = ((preds == 1) & (labels == 0)).sum().item()
            fn = ((preds == 0) & (labels == 1)).sum().item()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t.item()

        print(f"Epoch {epoch+1:02d} | val F1: {best_f1:.3f} @ threshold {best_thresh:.2f}")

    return model, best_thresh

def load_splits(
    conn,
    transcript_type,
    val_size= 0.15,
    test_size= 0.15,
    random_state= 42,
    min_tokens= 20,
) -> tuple[AuthorshipDataset, AuthorshipDataset, AuthorshipDataset]:

    df = load_dataset_from_db(conn, transcript_type)
    df = df[df["text"].str.split().str.len() >= min_tokens].reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=val_size + test_size, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)

    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    print(f"Train label balance:  {train_df['label'].mean():.3f} trump ratio")
    print(f"Val   label balance:  {val_df['label'].mean():.3f}")
    print(f"Test  label balance:  {test_df['label'].mean():.3f}")

    return (
        AuthorshipDataset(train_df.reset_index(drop=True)),
        AuthorshipDataset(val_df.reset_index(drop=True)),
        AuthorshipDataset(test_df.reset_index(drop=True)),
    )
def load_dataset_from_db(conn, transcript_type: str) -> pd.DataFrame:

    trump = pd.DataFrame(conn.execute(f"SELECT text FROM trump_{transcript_type}").fetchall(), columns=["text"])
    coca  = pd.DataFrame(conn.execute(f"SELECT text FROM congress_tweets").fetchall(),  columns=["text"])

    trump["label"] = 1
    coca["label"]  = 0

    df = pd.concat([trump, coca], ignore_index=True).dropna(subset=["text"])
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""]
    return df