import os
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29505'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'


class ShopBenchBaseModel:
    def __init__(self, ckpt_dir):
        max_batch_size = 20
        max_seq_len = 2048
        seed = AICROWD_RUN_SEED
        model_parallel_size = None
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # assert model_args.vocab_size == tokenizer.n_words
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.float16,
            device_map='auto',
            load_in_8bit=True
        )
        self.model = model
        self.tokenizer = tokenizer
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

    def predict_o(self, prompt: str, is_multiple_choice: bool, temperature: float = 0.6,
                  top_p: float = 0.9,
                  max_gen_len: Optional[int] = None,
                  logprobs: bool = False, ) -> str:

        if max_gen_len is None:
            max_gen_len = 512
        # 指定instruction和input
        if is_multiple_choice == True:  # task 11 #task15
            if "Which of the following statements best describes the relation from query" in prompt:  # task 11
                instruction = "You are a helpful shop assistant,you can understand the meaning of words very well.Please respond to the questions strictly in accordance with the user's requirements.Your response should only include the numbers preceding the correct answers, without any need for further explanation."
                input = prompt
            elif "Instructions: Evaluate the following product review on a scale of 1 to 5, with 1 being very negative and 5 being very positive." in prompt:  # task15
                first_line, rest_of_lines = prompt.split('\n', 1)
                a, instruction_2 = first_line.split(':', 1)
                b, input = rest_of_lines.split(':', 1)
                instruction_1 = "You are a helpful shop assistant.You are able to read user reviews of products very accurately and determine whether their evaluation of the product is postive or negative."
                instruction_3 = "Your answer should only include the numerical result of the evaluation; no further explanation is necessary."
                instruction = instruction_1 + instruction_2 + instruction_3
            else:  # 没见过的选择题任务
                instruction = "You are a helpful assistant,you can understand the meaning of words very well.Please respond to the questions strictly in accordance with the user's requirements.Your response should only include the numbers preceding the correct answers, without any need for further explanation."
                input = prompt
        else:  # task 12 #task 13 #task 14
            if " Each product and its number should appear only once in the output." in prompt:  # task 12
                instruction_1, input = prompt.split('\n', 1)
                instruction_2 = "You should output a permutation of 1 to 5. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations."
                instruction = instruction_1 + instruction_2
            elif "You make queries and click on products to eventually find the product you want and make your purchase" in prompt:  # task 13
                lines = prompt.split('\n')
                instruction = '\n'.join(lines[:3])
                input = '\n'.join(lines[3:])
            elif "You should output 3 numbers that correspond to the selected products. There should be a comma separating every two numbers. Only respond with the results. Do not say any word or explanations" in prompt:  # task 14
                instruction_1, input = prompt.split('\n', 1)
                instruction_2 = "You should output 3 numbers that correspond to the selected products. There should be a comma separating every two numbers. Only respond with the results. Do not say any word or explanations."
                instruction = instruction_1 + instruction_2
            else:  # 没见过的非选择题任务
                instruction = "You are a helpful assistant,you can understand the meaning of words very well.Please respond to the questions strictly in accordance with the user's requirements."
                input = prompt
        # 得到输出
        dialog = []
        text_1 = {"role": "system", "content": instruction}
        text_2 = {"role": "user", "content": input}
        dialog.append(text_1)
        dialog.append(text_2)
        prompt_token = self.tokenizer.apply_chat_template(
            dialog,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to('cuda')

        prompt_tokens = []
        prompt_tokens.append(prompt_token)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        generation_tokens = self.model.generate(
            prompt_token,
            max_new_tokens=max_gen_len,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        generation_token = generation_tokens[0][prompt_token.shape[-1]:]
        response = self.tokenizer.decode(generation_token, skip_special_tokens=True)
        # 规范化输出
        if is_multiple_choice == True:  # task 11 #task15
            response = find_first_digit(response)
        else:  # task 12 #task 13 #task 14
            if " Each product and its number should appear only once in the output." in prompt:  # task 12
                response = create_consecutive_digits_str(response)
            elif "You make queries and click on products to eventually find the product you want and make your purchase" in prompt:  # task 13
                response = process_string(response)
            elif "You should output 3 numbers that correspond to the selected products. There should be a comma separating every two numbers. Only respond with the results. Do not say any word or explanations" in prompt:  # task 14
                response = process_string(response)
            else:  # 没见过的非选择题任务
                response = response
        return response

    def predict(self, instruction: str, input: str, temperature: float = 0.6,
                top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.3,
                max_gen_len: Optional[int] = None,
                logprobs: bool = False, ) -> str:

        if max_gen_len is None:
            max_gen_len = 512

        # 得到输出
        dialog = []
        text_1 = {"role": "system", "content": instruction}
        text_2 = {"role": "user", "content": input}
        dialog.append(text_1)
        dialog.append(text_2)
        prompt_token = self.tokenizer.apply_chat_template(
            dialog,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to('cuda')

        prompt_tokens = []
        prompt_tokens.append(prompt_token)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        generation_tokens = self.model.generate(
            prompt_token,
            max_new_tokens=max_gen_len,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        generation_token = generation_tokens[0][prompt_token.shape[-1]:]
        response = self.tokenizer.decode(generation_token, skip_special_tokens=True)
        return response
