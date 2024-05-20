#library
import os
import wandb
import yaml
import torch
import torchvision
import transformers
import datasets
import peft
import accelerate
import numpy as np
import evaluate

#initialization
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    print(cfg)

if cfg["use_wandb"] == True:
    wandb.login()
    run = wandb.init(
        project="Vision Transformer with LoRA",
        name=cfg["wandb_project"],
        config={
            "learning_rate" : cfg["learning_rate"],
            "architecture" : cfg["arc"],
            "dataset" : cfg["dataset"]
            },
    )

# function

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

# main
# print(cfg['path'], cfg['images'])
# print(os.path.exists(cfg['path']+cfg['images']))
dataset = datasets.load_dataset("dataset.py",name="ISIC-2018",data_dir=cfg['path']+cfg['images'], split='train')
labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
image_processor = transformers.AutoImageProcessor.from_pretrained(cfg["checkpoint"])
normalize = torchvision.transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(image_processor.size["height"]),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize,
    ]
)
val_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(image_processor.size["height"]),
        torchvision.transforms.CenterCrop(image_processor.size["height"]),
        torchvision.transforms.ToTensor(),
        normalize,
    ]
)
splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
model = transformers.AutoModelForImageClassification.from_pretrained(
    cfg["checkpoint"],
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
config = peft.LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = peft.get_peft_model(model, config)
print_trainable_parameters(lora_model)
model_name = cfg["checkpoint"].split("/")[-1]
args = transformers.TrainingArguments(
    f"{model_name}-finetuned-lora-ISIC-2018",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=cfg["learning_rate"],
    per_device_train_batch_size=cfg["batch_size"],
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=cfg["batch_size"],
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    label_names=["labels"],
) 
metric = evaluate.load("accuracy")
trainer = transformers.Trainer(
    lora_model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()
trainer.evaluate(val_ds)