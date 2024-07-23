import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge(ckpt_dir, output_dir, save_path):
    adapter_path = os.path.join(output_dir, "belle_cn")
    # 设置合并后模型的导出地址
    # save_path = 'llama3/new_model'

    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    print("load model success")

    model = PeftModel.from_pretrained(model, adapter_path)
    print("load adapter success")
    model = model.merge_and_unload()
    print("merge success")

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print("save done.")


if __name__ == "__main__":
    ckpt_dir = "/home/jim/nas/lilxiaochen/kdd_cup_v2/models/llama3/Meta-Llama-3-8B-Instruct"
    output_dir = "save"
    merge(ckpt_dir, output_dir, "new_model")