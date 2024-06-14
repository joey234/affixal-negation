import datasets
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_function(examples):
    return tokenizer(examples["word"], truncation=True)


ds = datasets.load_dataset('joey234/affixal_negation', split = 'train')
ds = ds.class_encode_column('label')
ds = ds.train_test_split(test_size = 0.1, stratify_by_column="label")

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_ds = ds.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

id2label = {0: "Neg", 1: "Non-neg"}
label2id = {"Neg": 0, "Non-neg": 1}


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)



training_args = TrainingArguments(
    output_dir="finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()