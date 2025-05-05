#!pip install unsloth tqdm
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


# Configuration
BASE_MODEL = "unsloth/Qwen2.5-Coder-3B"
SAVED_MODEL_NAME = "nkasmanoff/jupyter-pilot" # HF username + model name
DATASET_PATH = "jupyter_fim.json"


# Model parameters
max_seq_length = 4096 
dtype = None 
load_in_4bit = False 


def initialize_model(max_seq_length=4096, dtype=None, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer

def format_train_example(example):
    prefix = example['prefix']
    suffix = example['suffix']
    middle = example['middle']
    return {'text': f'<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}<|endoftext|>'}

def format_test_example(example):
    prefix = example['prefix']
    suffix = example['suffix']
    return {'text': f'<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>'}

def load_and_prepare_dataset():
    dataset = load_dataset('json', data_files={'train': DATASET_PATH})
    dataset = dataset['train'].train_test_split(test_size=0.01, seed=42)
    train_valid_dataset = dataset['train'].train_test_split(test_size=0.01, seed=42)

    train_dataset = train_valid_dataset['train'].map(format_train_example)
    valid_dataset = train_valid_dataset['test'].map(format_train_example)
    test_dataset = dataset['test'].map(format_test_example)

    return train_dataset, valid_dataset, test_dataset

def get_peft_model(model):
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

def get_training_args():
    return TrainingArguments(
        run_name=f'{SAVED_MODEL_NAME}-sft',
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=1,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=10,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb",
    )

def main():
    model, tokenizer = initialize_model()
    model = get_peft_model(model)
    
    train_dataset, valid_dataset, test_dataset = load_and_prepare_dataset()
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=get_training_args(),
    )

    trainer_stats = trainer.train()
    model.push_to_hub(SAVED_MODEL_NAME)

if __name__ == "__main__":
    main()

# Convert adapter using https://huggingface.co/spaces/ggml-org/gguf-my-lora