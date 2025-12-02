# Download stopwords and define which is essential for emotion classification
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

emotion_keep = {
    "not", "no", "nor", "never",
    "very", "too", "so",
    "i", "you", "he", "she", "we", "they",
    "am", "is", "are", "was", "were",
    "my", "me", "your", "yours",
    "dont", "didnt", "doesnt",
    "cant", "couldnt", "wouldnt", "shouldnt"
}

custom_stopwords = stop_words - emotion_keep
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return text.split()
    
def remove_stopwords(tokens):
    return [t for t in tokens if t not in custom_stopwords]

def lemma_then_stem(tokens):
    return [stemmer.stem(lemmatizer.lemmatize(t, pos='v')) for t in tokens]

def build_vocab_all(texts):
    all_tokens = []
    for t in texts:
        toks = preprocess_text(t)
        all_tokens.extend(toks)

    unique_words = sorted(set(all_tokens))

    word2idx = {"<PAD>": 0, "<OOV>": 1}
    for idx, word in enumerate(unique_words, start=2):
        word2idx[word] = idx
    return word2idx

    return word2idx
def preprocess_text(text):
    text = clean_text(text)
    toks = tokenize(text)
    toks = remove_stopwords(toks)
    toks = lemma_then_stem(toks)
    return toks


train["clean"] = train["text"].apply(preprocess_text)
vocab = build_vocab_all(train["clean"].values)

def text_to_sequence(text, vocab):
    toks = preprocess_text(text)

    if len(toks) == 0:
        toks = ["<OOV>"]

    return [vocab.get(t, vocab["<OOV>"]) for t in toks]
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=64):
        self.sequences = [text_to_sequence(t, vocab)[:max_len] for t in texts]
        self.labels = np.array(labels, dtype=np.float32)
        self.lengths = [len(seq) for seq in self.sequences]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
            self.lengths[idx]
        )

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)

    labels = torch.stack(labels)

    pad_idx = vocab["<PAD>"]
    max_len = max(lengths)

    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        if pad_len > 0:
            seq = torch.cat([seq, torch.full((pad_len,), pad_idx, dtype=torch.long)])
        padded_sequences.append(seq)

    sequences = torch.stack(padded_sequences)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return sequences, labels, lengths
wandb.login(key=secret)
print("Done")
embed_dim = 2000
hidden_dim = 2000
wandb.init(
    project="DLgenAI",
    name=f"LSTM {embed_dim},{hidden_dim}, 0.05 "
)
class EmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                 output_dim=5, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), 
            batch_first=True,
            enforce_sorted=False    
        )

        out, (h, c) = self.lstm(packed)
        h_forward  = h[-2]
        h_backward = h[-1]
        final_rep = torch.cat((h_forward, h_backward), dim=1)
        final_rep = self.dropout(final_rep)
        return self.fc(final_rep)

       
train = pd.read_csv("/kaggle/input/2025-sep-dl-gen-ai-project/train.csv")
train["clean"] = train["text"].apply(clean_text)
vocab = build_vocab_all(train["clean"].values)
target_cols = ["anger", "fear", "joy", "sadness", "surprise"]

X_train, X_val, y_train, y_val = train_test_split(
    train["clean"].values,
    train[target_cols].values,
    test_size=0.1,
    random_state=42,
    stratify = train["anger"]
)

MAX_LEN = 64
BATCH_SIZE = 64

train_dataset = EmotionDataset(X_train, y_train, vocab, MAX_LEN)
val_dataset   = EmotionDataset(X_val, y_val, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = EmotionBiLSTM(len(vocab), output_dim=len(target_cols)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
model = nn.DataParallel(model)

EPOCHS = 30
THRESHOLD = 0.5

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

EPOCHS = 30
THRESHOLD = 0.5

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
print(len(vocab))
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Training]")

    for input_ids, labels, lengths in loop:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        logits = model(input_ids, lengths) 
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item() * input_ids.size(0)
        loop.set_postfix(loss=loss.item())

    train_loss /= len(train_loader.dataset)

    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for input_ids, labels, lengths in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(input_ids, lengths)
            probs  = torch.sigmoid(logits)           
            preds  = (probs >= THRESHOLD).int().cpu().numpy()

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
