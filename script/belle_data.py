import random
import json


def write_txt(file_path, datas):
    with open(file_path, "w", encoding="utf8") as f:
        for d in datas:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        f.close()


with open("data/Belle_open_source_0.5M.json", "r", encoding="utf8") as f:
    lines = f.readlines()
    # 拼接数据
    changed_data = []
    for l in lines:
        l = json.loads(l)
        changed_data.append({"text": "### Human: " + l["instruction"] + " ### Assistant: " + l["output"]})

    # 从拼好后的数据中，随机选出若干条，作为训练数据
    # r_changed_data = random.sample(changed_data, 1000)
    r_changed_data = changed_data

    # 写到json中
    write_txt("data/Belle_open_source_0.5M_changed_test.json", r_changed_data)