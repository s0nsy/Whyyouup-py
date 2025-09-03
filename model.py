from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# 1. 사전 학습된 FinBERT 로드
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 2. 데이터 불러오기
df = pd.read_csv("news_dataset.csv")  # 컬럼: code, text, sentiment
df = df[['text', 'sentiment']]  # code는 지금 학습에 불필요

# sentiment가 문자열이면 숫자로 매핑
label_map = {"positive": 2, "neutral": 1, "negative": 0}
df['sentiment'] = df['sentiment'].map(label_map)

# 3. Dataset 변환
dataset = Dataset.from_pandas(df)

# 4. 토큰화
def tokenize(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("sentiment", "labels")
dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

# 5. Trainer 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # 임시 (나중에 train/test split 추가)
    tokenizer=tokenizer
)

# 6. 학습
trainer.train()

# 7. 모델 저장
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
