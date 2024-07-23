import transformers
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

model_id = "/home/jim/nas/lilxiaochen/kdd_cup_v2/models/llama3/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are an intelligent shopping assistant that can rank products based on their relevance to the query.The following numbered list contains 5 products.Please rank the products according to their relevance with the queryYou should output a permutation.  There should be a comma separating two numbers.  Each product and its number should appear only once in the output.  Only respond with the ranking results.  Do not say any word or explanations"},
    {"role": "user", "content": "\"\"query\": \"intex saltwater system for above ground pool\", \"product list\": [\"A: Main Access 444301 Power Ionizer Hybrid Complete Swimming Pool Care Sanitation System that Treats Up to 40,000 Gallons of Water\", \"B: Hayward W3AQ-TROL-RJ AquaTrol Salt Chlorination System for Above-Ground Pools up to 18,000 Gallons with Return Jet Fittings, Straight Blade Line Cord and Outlet\", \"C: Intex Krystal Clear Sand Filter Pump for Above Ground Pools, 16-inch, 110-120V with GFCI\", \"D: Intex 2100 GPH Above Ground Pool Sand Filter Pump w/ Deluxe Pool Maintenance Kit\", \"E: Pentair 520555 IntelliChlor IC40 Salt Chlorine Generator Cell (40,000 Gallon\"]\""},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])
