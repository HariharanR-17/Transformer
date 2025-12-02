import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"

class TransformerClassifier(nn.Module):
    def __init__(self, model_name="roberta-large", num_classes=5):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 10),
            nn.Linear(10, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


class TestEmotionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0), idx


MODEL_PATH = hf_hub_download(
    repo_id="Hariharan1703/emotionclassifier",
    filename="robertalargeuf_best.pth"
)

BASE_MODEL = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = TransformerClassifier(BASE_MODEL, num_classes=5)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']
thresholds = np.array([0.24, 0.89, 0.5, 0.55, 0.36])

def predict(text):
    dataset = TestEmotionDataset([text], tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for input_ids, attention_mask, _ in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = (probs >= thresholds).astype(int)

    result = {labels[i]: float(probs[i]) for i in range(5)}
    result["predicted_labels"] = [labels[i] for i in range(5) if preds[i] == 1]

    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter text"),
    outputs=gr.JSON(label="Emotion Output"),
    title="RoBERTa-Large Emotion Classifier",
    description="Multilabel emotion prediction"
)

if __name__ == "__main__":
    demo.launch()
