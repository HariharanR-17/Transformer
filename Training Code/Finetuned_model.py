# Set seed for reproducible results
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
set_seed(42)

wandb.login(key=secret)
print("Done")
wandb.init(project="DLgenAI", name="Roberta Large")

# cuda for faster computing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, label

# Data loaders
def get_loaders(tokenizer, batch_size=100):
    train_dataset = EmotionDataset(X_train.tolist(), y_train, tokenizer)
    val_dataset = EmotionDataset(X_val.tolist(), y_val, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Transformer architecture
class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_classes=5):
        super().__init__()
        self.model = AutoModel.from_pretrained("roberta-large")
        hidden = self.model.config.hidden_size
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Sequential(        
            nn.Linear(hidden, 10),
            nn.Linear(10,5)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :] 
        # cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

# Evaluation function
def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(loader, desc="Validating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    bin_preds = (all_preds >= threshold).astype(int)
    f1_macro = f1_score(all_labels, bin_preds, average="macro")
    num_classes = all_preds.shape[1]
    best_thresholds = []
    for c in range(num_classes):
        best_f1 = 0
        best_t = 0.5
        for t in np.linspace(0.1, 0.9, 81): 
            preds_c = (all_preds[:, c] >= t).astype(int)
            f1 = f1_score(all_labels[:, c], preds_c)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds.append(best_t)
    thresholds_array = np.array(best_thresholds)
    bin_preds_opt = (all_preds >= thresholds_array).astype(int)
    max_f1_macro = f1_score(all_labels, bin_preds_opt, average="macro")
    print("Optimal thresholds per class:", best_thresholds)
    print("Maximum achievable F1-macro:", max_f1_macro)
    return f1_macro
    
# Training function
def train_model(model_name, num_epochs=10, lr=5e-5):
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    train_loader, val_loader = get_loaders(tokenizer)

    model = TransformerClassifier(model_name).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    model.load_state_dict(torch.load("robertalargef_best.pth", map_location=device))
    model = nn.DataParallel(model)
    best_f1 = 0.0
    patience = 10
    drops = 0

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps) 
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        power=1
    )

    best_path = f"{model_name}_best.pth"

    for epoch in range(num_epochs):
        model.train()            
        running_loss = 0.0

        for input_ids, attention_mask, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            # max_grad_norm = 1.0 
            loss.backward()
            # utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * input_ids.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)    
        f1_macro = evaluate(model, val_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Val Macro-F1: {f1_macro:.4f}")
        
        wandb.log({
        "epoch": epoch+1,
        "train_loss": epoch_loss,
        "val_f1": f1_macro})
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.module.state_dict(), f"{model_name}_best.pth")
            print("Saved best model")
        else:
            drops += 1
            print(f"F1 drop count: {drops}/{patience}")

        if drops >= patience:
            print("Early stopping due to F1 drops")
            break

    final_path = f"{model_name}_final.pth"
    torch.save(model.module.state_dict(), final_path)
    print(f"Saved FINAL model {final_path}")
    return model

X_train, X_val, y_train, y_val = train_test_split(
    train['text'], 
    train[['anger','fear','joy','sadness','surprise']].values, 
    test_size=0.2, 
    random_state=42,
    shuffle=True,
    stratify=train['anger']
)
# pos weight for class imbalance handling
# pos = y_train.sum(axis=0)          
# neg = len(y_train) - pos          
# pos_weight = torch.tensor(neg / pos, dtype=torch.float)

for model_name in ["robertalargeuf"]:
    print(f"Training {model_name.upper()}")
    train_model(model_name, num_epochs=15)
wandb.finish()
