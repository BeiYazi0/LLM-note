## Ref

[大模型相关技术原理以及实战经验](https://github.com/liguodongiot/llm-action)

---

## llama3 本地部署

```python
conda create --name llama3 python=3.9.18

source activate llama3

git clone https://github.com/meta-llama/llama3

cd llama3

pip install -e .

bash ./download.sh
```

下载需要在 [Meta Llama website](https://llama.meta.com/llama-downloads/) 填写信息，收到的电子邮件中含有 URL，填入下载脚本中。

[403 forbidden error](https://github.com/Meta-Llama/llama/issues/373) 可能需要走代理。

官方给出了文本生成和对话的两个例子，两者加载 llama3 和 生成的代码相似

```python
generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
```

超参数的含义 [Top-k & Top-p, Temperature](https://zhuanlan.zhihu.com/p/613428710)。

官方给出对话的运行示例如下

```python
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```

单卡要求 16G，运行结果

```python
Loaded in 20.44 seconds                                                                                                                                                       
User: what is the recipe of mayonnaise?                                                                                                                                       
                                                                                                                                                                              
> Assistant: The classic condiment! Mayonnaise is a thick, creamy sauce made from a combination of oil, egg yolks, acid (such as vinegar or lemon juice), and seasonings. Here
's a basic recipe for homemade mayonnaise:                                                                                                                                    
                                                                                                                                                                              
**Ingredients:**                                                                                                                                                              
                                                                                                                                                                              
* 2 large egg yolks                                                                                                                                                           
* 1 tablespoon lemon juice or vinegar (such as apple cider vinegar or white wine vinegar)                                                                                     
* 1/2 teaspoon Dijon mustard (optional, but recommended for flavor)                                                                                                           
* 1/2 teaspoon salt                                                                                                                                                           
* 1 cup (240 ml) neutral-tasting oil, such as canola, grapeseed, or vegetable oil                                                                                             
                                                                                                                                                                              
**Instructions:**                                                                                                                                                             
                                                                                                                                                                              
1. **Start with room temperature ingredients**: This ensures that the egg yolks and oil mix smoothly.                                                                         
2. **In a medium bowl, whisk together the egg yolks, lemon juice or vinegar, Dijon mustard (if using), and salt**: Whisk until the mixture is smooth and well combined.       
3. **Slowly add the oil while whisking continuously**: Start with a very slow, thin stream of oil and gradually increase the flow as the mixture thickens. Whisk constantly to
 prevent the oil from separating from the egg yolks.                                                                                                                          
4. **Continue whisking until the mixture thickens and emulsifies**: This should take about 5-7 minutes, depending on the speed of your whisking. The mixture will start to thi
cken and turn pale yellow, indicating that it's emulsifying.                                                                                                                  
5. **Taste and adjust**: If the mayonnaise is too thick, add a little water. If it's too thin, add a little more oil.                                                         
6. **Transfer to an airtight container**: Store the mayonnaise in the refrigerator for up to 1 week.                                                                          
                                                                                                                                                                              
**Tips and Variations:**                                                                                                                                                      
                                                                                                                                                                              
* Use a neutral-tasting oil to prevent the mayonnaise from taking on a strong flavor.                                                                                         
* Add flavorings like minced garlic, grated ginger, or chopped fresh herbs (like parsley, dill, or chives) to the egg yolks before whisking in the oil.
* For a lighter mayonnaise, use only 1/4 cup (60 ml) of oil and whisk in a little more lemon juice or vinegar.
* For a spicy mayonnaise, add a pinch of cayenne pepper or red pepper flakes to the egg yolks before whisking in the oil.

Remember

==================================

User: I am going to Paris, what should I see?

Assistant: Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attracti
ons to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in
 the world.

User: What is so great about #1?

> Assistant: The Eiffel Tower is an iconic symbol of Paris and one of the most recognizable landmarks in the world. Here are some reasons why it's so great:

1. **Engineering marvel**: The Eiffel Tower was built for the 1889 World's Fair and was the tallest structure in the world at the time, standing at 324 meters (1,063 feet). Its intricate iron latticework design was a technological marvel for its time and has been an inspiration to engineers and architects ever since.
2. **Panoramic views**: The Eiffel Tower offers breathtaking views of the City of Light from its observation decks, which are located at 57 meters (187 feet) and 115 meters (377 feet) above ground. On a clear day, you can see for miles in every direction, taking in the stunning architecture, gardens, and landmarks of Paris.
3. **Romantic ambiance**: The Eiffel Tower is often associated with romance and is a popular spot for couples to propose, celebrate anniversaries, or simply enjoy a romantic dinner at one of the many restaurants on site.
4. **Historical significance**: The Eiffel Tower has played a significant role in French history, having been built for the World's Fair and later serving as a symbol of French culture and engineering prowess.
5. **Accessibility**: The Eiffel Tower is easily accessible by public transportation or on foot, making it a must-visit attraction for anyone

==================================

System: Always answer with Haiku

User: I am going to Paris, what should I see?

> Assistant: Eiffel's iron kiss
River Seine's gentle whispers
Art and love abide

==================================

System: Always answer with emojis

User: How to go from Beijing to NY?

> Assistant: ✈️️️️️️🛫️️️️️️🗺️️️🇺🇸️

==================================
```

---

## Token 分词

[Tokenizer](tokenizer.md)

---

## 微调

[fine tune](fine_tune.md)

---

## 向量检索增强

[rag](rag.md)

---

## 思维链

[cot](cot.md)

---

## 多任务推理

在实际对话时,通常会有三种角色
- System 给定一些上下文信息，比如“你是一个安全的AI助手”。
- User 实际用户，会提出一些问题，比如“世界第一高峰是？”。
- Assistant 根据User的输入，结合System 的上下文信息，做出回答。

[{"role": "system","content":instruction}, {"role": "user","content":input}]

在使用对话模型时，通常是不会感知到这三种角色的。

使用 llama3 对多种类型任务进行推理示例如下。对于不同类型的任务，指定不同的 instruction，对 User 输入的 prompt 进行预处理得到 input，
对 response 进行后处理得到推理结果。

task11: 描述关系，多选。
task12: Concept Normalization，将许多指向同一事物的概念规范化，每个产品及其编号只能在输出中出现一次。
task13: 用通俗易懂、简洁明了的语言向顾客阐述或解释产品。
task14: 产品描述通常很长，提取和总结来告诉客户产品是否符合他的特定需求。输出3个与所选产品相对应的数字。
task15: 分析评论中表达的情感的能力，1代表非常负面，5代表非常正面。

```python
class ShopBenchBaseModel:
    def __init__(self):
        ckpt_dir="./models/llama3/Meta-Llama-3-8B-Instruct"
        max_batch_size = 20
        max_seq_len = 2048
        seed=AICROWD_RUN_SEED
        model_parallel_size=None
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
        #assert model_args.vocab_size == tokenizer.n_words
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.float16,
            device_map='cuda:0',
            load_in_8bit=True
        )
        self.model=model
        self.tokenizer=tokenizer
        print(f"Loaded in {time.time() - start_time:.2f} seconds")


    def predict(self, prompt: str, is_multiple_choice: bool,temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,) -> str:

        if max_gen_len is None:
            max_gen_len = 512
        #指定instruction和input
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
            else: #没见过的选择题任务
                instruction="You are a helpful assistant,you can understand the meaning of words very well.Please respond to the questions strictly in accordance with the user's requirements.Your response should only include the numbers preceding the correct answers, without any need for further explanation."
                input=prompt
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
            else: #没见过的非选择题任务
                instruction="You are a helpful assistant,you can understand the meaning of words very well.Please respond to the questions strictly in accordance with the user's requirements."
                input=prompt
        #得到输出
        dialog=[]
        text_1={"role": "system","content":instruction}
        text_2={"role": "user","content":input}
        dialog.append(text_1)
        dialog.append(text_2)
        prompt_token = self.tokenizer.apply_chat_template(
            dialog,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to('cuda')

        prompt_tokens=[]
        prompt_tokens.append(prompt_token)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        generation_tokens= self.model.generate(
            prompt_token,
            max_new_tokens=max_gen_len,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        generation_token= generation_tokens[0][prompt_token.shape[-1]:]
        response=self.tokenizer.decode(generation_token, skip_special_tokens=True)
        #规范化输出
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
```

---

## Related

[Transformer](https://zhuanlan.zhihu.com/p/389183195)

[模型并行和分布式训练解析](https://zhuanlan.zhihu.com/p/343951042)

[大模型学习路线（10）——入门项目推荐](https://blog.csdn.net/qq_51175703/article/details/137229088)
