import os
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['cot_type'] = 'no'

import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import List

def no_cot(inputs):
    return inputs


def zero_shot(inputs):
    cot_inputs = "Let's work this out in a step by step way to be sure we have the right answer.\n" + inputs
    return cot_inputs


def few_shot(inputs):
    example = '''query:Which of the following statements best describes the relation from query \"Coffee Maker\" to query \"Yoga Mat\""\n0.   irrelevant\n1.   complement\n2.   substitute\n3.   narrowing".
answer:
A coffee machine is a device used to brew and dispense coffee.
Yoga mat, which is the mat you put under when you practice yoga.
They do not have a direct relationship or connection.
The answer is 0.
query:Which of the following statements best describes the relation from query \"Gardening Gloves\" to query \"Gardening Tools\"?,\n0.  irrelevant\n1.  complement\n2.  substitute\n3.  narrowing.
answer:
Gardening gloves are a type of protective handwear specifically designed for gardening activities. Gardening gloves help to protect the hands from cuts, blisters, thorns, and other potential injuries while working with gardening tools or handling plants.
Gardening tools refer to a variety of equipment and implements that are used for gardening activities. These tools can include items such as shovels, rakes, hoes, pruning shears, watering cans, and trowels, among others. 
They complement each other in the context of gardening activities.
The answer is 1.
query:Which of the following statements best describes the relation from query \"Crest Complete Whitening + Scope Toothpaste\" to query \"Colgate Total Whitening Toothpaste\"?,\n0. irrelevant\n1. complement\n2. substitute\n3. narrowing".
answer:
"Crest Complete Whitening + Scope Toothpaste" is a specific brand and variant of toothpaste. 
"Colgate Total Whitening Toothpaste" is also a specific brand and variant of toothpaste. 
They can be substituted for each other.
The answer is 2.
query:Which of the following statements best describes the relation from query \"The Legend of Zelda: Breath of the Wild\" to query \"Nintendo Switch\"?,\n0. irrelevant\n1. complement\n2. substitute\n3. narrowing".
answer:
"The Legend of Zelda: Breath of the Wild" is an open-world action-adventure game developed and published by Nintendo. It was released for the Nintendo Switch and Wii U consoles.
The "Nintendo Switch" is a hybrid gaming console developed by Nintendo. "The Legend of Zelda: Breath of the Wild" is a specific game available on the "Nintendo Switch".
Therefore, the game query narrows down the scope of the console query.
The answer is 3.
query:'''
    cot_inputs = example + inputs
    return cot_inputs


def calculate_per_sample_accuracy(prediction: int, truth: int) -> bool:
    return prediction == truth


def calculate_ndcg(predicted_relevance_scores: List[int], true_relevance_weights: List[float]) -> float:
    # Trim the predicted scores to match the true scores length if necessary
    if len(predicted_relevance_scores) > len(true_relevance_weights):
        predicted_relevance_scores = predicted_relevance_scores[:len(true_relevance_weights)]

    dcg, idcg = 0.0, 0.0

    # Calculate DCG for the predicted ranking
    for i, score_index in enumerate(predicted_relevance_scores, start=1):
        if score_index - 1 < len(true_relevance_weights):
            relevance = true_relevance_weights[score_index - 1]
        else:
            relevance = 0
        dcg += (np.power(2, relevance) - 1) / np.log2(i + 1)

    # Calculate IDCG using sorted true relevance weights
    for i, weight in enumerate(sorted(true_relevance_weights, reverse=True), start=1):
        idcg += (np.power(2, weight) - 1) / np.log2(i + 1)

    # Avoid division by zero
    return 0 if idcg == 0 else dcg / idcg


def pre_process(prompt, is_multiple_choice):
    if is_multiple_choice == True:  # task 11 #task15
        if "Which of the following statements best describes the relation from query" in prompt:  # task 11
            instruction = "You are a helpful shop assistant,you can understand the meaning of words very well.Please respond to the questions strictly in accordance with the user's requirements.Your response should include the numbers preceding the correct answers."
            inputs = cot_method(prompt)
        elif "Instructions: Evaluate the following product review on a scale of 1 to 5, with 1 being very negative and 5 being very positive." in prompt:  # task15
            first_line, rest_of_lines = prompt.split('\n', 1)
            a, instruction_2 = first_line.split(':', 1)
            b, inputs = rest_of_lines.split(':', 1)
            instruction_1 = "You are a helpful shop assistant.You are able to read user reviews of products very accurately and determine whether their evaluation of the product is postive or negative."
            instruction_3 = "Your answer should only include the numerical result of the evaluation; no further explanation is necessary."
            instruction = instruction_1 + instruction_2 + instruction_3
        else:  # 没见过的选择题任务
            instruction = "You are a helpful assistant,you can understand the meaning of words very well.Please respond to the questions strictly in accordance with the user's requirements.Your response should only include the numbers preceding the correct answers, without any need for further explanation."
            inputs = prompt
    else:  # task 12 #task 13 #task 14
        if " Each product and its number should appear only once in the output." in prompt:  # task 12
            instruction_1, inputs = prompt.split('\n', 1)
            instruction_2 = "You should output a permutation of 1 to 5. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations."
            instruction = instruction_1 + instruction_2
        elif "You make queries and click on products to eventually find the product you want and make your purchase" in prompt:  # task 13
            lines = prompt.split('\n')
            instruction = '\n'.join(lines[:3])
            inputs = '\n'.join(lines[3:])
        elif "You should output 3 numbers that correspond to the selected products. There should be a comma separating every two numbers. Only respond with the results. Do not say any word or explanations" in prompt:  # task 14
            instruction_1, inputs = prompt.split('\n', 1)
            instruction_2 = "You should output 3 numbers that correspond to the selected products. There should be a comma separating every two numbers. Only respond with the results. Do not say any word or explanations."
            instruction = instruction_1 + instruction_2
        else:  # 没见过的非选择题任务
            instruction = "You are a helpful assistant,you can understand the meaning of words very well.Please respond to the questions strictly in accordance with the user's requirements."
            inputs = prompt

    return instruction, inputs

def generate_model_outputs(data_df, model):
    outputs = []
    for _, row in tqdm(
        data_df.iterrows(), total=len(data_df), desc="Generating Responses"
    ):
        instruction, inputs = pre_process(row["input_field"], row["is_multiple_choice"])
        model_output = model.predict(instruction, inputs)
        outputs.append(model_output)
    return outputs


def parse_ranking(response: str) -> list:
    # res_list = ["A", "B", "C", "D", "E", ",", " "]
    # res_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    default_respomse = []
    # Keep only numeric characters and specific punctuation.
    cleaned_response = "".join(
        c for c in response if c.isnumeric() or c in [",", " "]
    )

    # Convert to list of integers
    ranked_items = []
    for item in cleaned_response.split(","):
        try:
            # Attempt to convert each item to an integer and add it to the list.
            int_item = int(item)
            if int_item <= 5:  # we know int_item can be at most 5
                ranked_items.append(int_item)
        except ValueError:
            pass  # Skip non-numeric items.

    # Consider only the first 5 unique elements.
    ranked_items = ranked_items[:5]

    # If there are duplicates, empty the list
    if len(ranked_items) != len(set(ranked_items)):
        ranked_items = default_respomse

    return ranked_items


def parse_multichoice(response: str) -> int:
    choice = 0
    response = response[::-1]
    for i in range(len(response)):
        if response[i].isdigit():
            choice = response[i]
            break
    return int(choice)


def evaluate_outputs(data_df, outputs, log_every_n_steps=1):
    per_task_metrics = {}
    task_parsers = {
        "multiple-choice": parse_multichoice,
        # "generation": parsers.ShoppingBenchTaskParsers("generation"),
        # "retrieval": parsers.ShoppingBenchTaskParsers("retrieval"),
        "ranking": parse_ranking,
        # "named_entity_recognition": parsers.ShoppingBenchTaskParsers(
        #     "named_entity_recognition"
        # ),
    }
    evaluation_methods = {
        "accuracy": calculate_per_sample_accuracy,
        # "hit rate@3": metrics.calculate_hit_rate_3,
        # "rougel": metrics.calculate_rougel,
        # "sent-transformer": lambda generated_text, reference_texts: metrics.calculate_cosine_similarity(
        #     generated_text=generated_text,
        #     reference_texts=reference_texts,
        #     model_name="all-MiniLM-L6-v2",
        # ),
        # "multilingual-sent-transformer": lambda generated_text, reference_texts: metrics.calculate_cosine_similarity(
        #     generated_text=generated_text,
        #     reference_texts=reference_texts,
        #     model_name="paraphrase-multilingual-MiniLM-L12-v2",
        # ),
        # "micro f1": metrics.calculate_true_positive_false_positives_false_negatives,
        "ndcg": calculate_ndcg,
        # "bleu": metrics.calculate_bleu_score,
        # "jp-bleu": lambda generated_text, reference_text: metrics.calculate_bleu_score(
        #     generated_text=generated_text,
        #     reference_text=reference_text,
        #     is_japanese=True,
        # ),
    }

    confusion_matrix = np.zeros((4, 4))

    for row_idx, row in tqdm(
        data_df.iterrows(), total=len(data_df), desc="Evaluating"
    ):
        task_name, task_type, metric, ground_truth = (
            row["task_name"],
            row["task_type"],
            row["metric"],
            row["output_field"],
        )

        model_output = task_parsers[task_type](outputs[row_idx])
        eval_fn = evaluation_methods[metric]
        metric_score = eval_fn(model_output, ground_truth)

        if task_name == "task11":
            try:
                confusion_matrix[ground_truth][model_output] += 1
            except:
                confusion_matrix[ground_truth][3] += 1

        if task_name not in per_task_metrics:
            per_task_metrics[task_name] = {
                "task_type": task_type,
                "metric": metric,
                "sample_score": [],
            }

        per_task_metrics[task_name]["sample_score"].append(metric_score)

        if (row_idx + 1) % log_every_n_steps == 0:
            print(
                row_idx + 1, model_output, ground_truth, metric, metric_score
            )
    print(confusion_matrix)
    res.append(confusion_matrix)
    return per_task_metrics


def aggregate_scores(per_task_metrics):
    overall_metrics = {
        "task_name": [],
        "task_type": [],
        "metric": [],
        "num_samples": [],
        "overall_score": [],
    }
    for task_name, values in per_task_metrics.items():
        task_type, metric, sample_scores = (
            values["task_type"],
            values["metric"],
            values["sample_score"],
        )
        overall_score = np.mean(sample_scores)

        overall_metrics["task_name"].append(task_name)
        overall_metrics["task_type"].append(task_type)
        overall_metrics["metric"].append(metric)
        overall_metrics["num_samples"].append(len(sample_scores))
        overall_metrics["overall_score"].append(overall_score)

    return pd.DataFrame(overall_metrics)


def main(model, idx):
    DATA_FILENAME = "./data/multiple-choice-task11_test.json"
    data_df = pd.read_json(DATA_FILENAME, lines=True)

    # Generate model outputs
    outputs = generate_model_outputs(data_df, model)
    answer[idx] = outputs[-1]
    data_df["outputs"] = outputs
    print(data_df.head())

    # Evaluate the generated outputs and calculate metrics
    per_task_metrics = evaluate_outputs(data_df, outputs)

    # Aggregate and display the evaluation scores
    overall_metrics = aggregate_scores(per_task_metrics)
    res.append(overall_metrics)
    print("=" * 100)
    print("Task specific metrics: ")
    print(overall_metrics)

    overall_score = overall_metrics["overall_score"].mean()
    print(f"Overall Score: {overall_score}")
    print(res)


if __name__ == "__main__":
    res = []
    answer = [0]*3
    ckpt_dir = "/home/jim/nas/lilxiaochen/kdd_cup_v2/models/llama3/Meta-Llama-3-8B-Instruct"
    from base_model import ShopBenchBaseModel
    model = ShopBenchBaseModel(ckpt_dir)

    # cot_method = no_cot
    # main(model, 0)
    #
    # cot_method = zero_shot
    # main(model, 1)

    cot_method = few_shot
    main(model, 2)

    for ans in answer:
        print(ans)
    for data in res:
        print(data)