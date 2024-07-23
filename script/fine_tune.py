import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ["WANDB_API_KEY"] = ""

import torch
import wandb
import json

from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

wandb.login()

wandb.init(project="llama3")


def load_model_tokenizer(ckpt_dir):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = 'left'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 在4bit上，进行量化
        bnb_4bit_use_double_quant=True,  # 嵌套量化，每个参数可以多节省0.4位
        bnb_4bit_quant_type="nf4",  # NF4（normalized float）或纯FP4量化 博客说推荐NF4
        bnb_4bit_compute_dtype=torch.float16
    )
    # assert model_args.vocab_size == tokenizer.n_words
    # if torch.cuda.is_bf16_supported():
    #     torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    # else:
    #     torch.set_default_tensor_type(torch.cuda.HalfTensor)

    device_map = {"": 0}  # 有多个gpu时，为：device_map = {"": [0,1,2,3……]}
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        quantization_config=bnb_config,  # 上面本地模型的配置
        device_map="auto",  # 使用GPU的编号
        torch_dtype=torch.float16
    )

    # model.config.use_cache = False
    # model.config.pretraining_tp = 1
    return tokenizer, model


def train(ckpt_dir, output_dir):
    training_args = TrainingArguments(
        report_to="wandb",
        per_device_train_batch_size = 2, #每个GPU的批处理数据量
        gradient_accumulation_steps = 4, #在执行反向传播/更新过程之前，要累积其梯度的更新步骤数
        warmup_steps = 5,
        max_steps = 60,  # 微调步数
        learning_rate = 2e-4, # 学习率
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1, #两个日志记录之间的更新步骤数
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
    )

    # 配置QLora
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    max_seq_length = 512
    tokenizer, model = load_model_tokenizer(ckpt_dir)

    dataset = load_dataset("json",data_files="data/Belle_open_source_0.5M_changed_test.json",split="train")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()

    adapter_path = os.path.join(output_dir, "belle_cn")
    trainer.model.save_pretrained(adapter_path)


if __name__ == "__main__":
    ckpt_dir = "/home/jim/nas/lilxiaochen/kdd_cup_v2/models/llama3/Meta-Llama-3-8B-Instruct"
    output_dir = "save"
    train(ckpt_dir, output_dir)