wandb.login(key=secret)
print("Done")

# Hyperparameters for transformer
embed_dim=16
d_model=16
nhead=2
num_layers=1
dim_feedforward=16
wandb.init(
    project="DLgenAI",
    name=f"CT {embed_dim},{d_model},{nhead},{num_layers},{dim_feedforward},1e-4"
)

# Seed for reproducible results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(42)
# Text preprocessing
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
VOCAB_SIZE = tokenizer.vocab_size

# Encoder function
def encode_bert(text, tokenizer, max_len):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors=None
    )

    return encoded["input_ids"], encoded["attention_mask"]

# Dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids, mask = encode_bert(self.texts[idx], self.tokenizer, self.max_len)

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )
    
# Transformer Architecture
class EmotionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=embed_dim, d_model=d_model,
                 nhead=nhead, num_layers=num_layers, output_dim=5, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(embed_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        pad_mask = (attention_mask == 0)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        cls_rep = self.norm(x[:, 0, :])
        return self.fc(cls_rep)

train = pd.read_csv("/kaggle/input/2025-sep-dl-gen-ai-project/train.csv")
train["clean"] = train["text"].apply(clean_text)

target_cols = ["anger", "fear", "joy", "sadness", "surprise"]

X_train, X_val, y_train, y_val = train_test_split(
    train["clean"].values,
    train[target_cols].values,
    test_size=0.1,
    random_state=42,
    stratify = train["anger"]
)

MAX_LEN = 64
BATCH_SIZE = 256

train_dataset = EmotionDataset(X_train, y_train, tokenizer, MAX_LEN)
val_dataset   = EmotionDataset(X_val, y_val, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = EmotionTransformer(
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN,
    output_dim=len(target_cols)
).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 40
THRESHOLD = 0.5

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Training]")
    for input_ids, attention_mask, labels in loop:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * input_ids.size(0)
        loop.set_postfix(loss=loss.item())
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            preds = (torch.sigmoid(logits).cpu().numpy() >= THRESHOLD).astype(int)
            preds_all.append(preds)
            labels_all.append(labels.cpu().numpy())

    preds_all = np.vstack(preds_all)
    labels_all = np.vstack(labels_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "val_f1": macro_f1})

    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss:.4f}, Val Macro-F1: {macro_f1:.4f}")
wandb.finish()
