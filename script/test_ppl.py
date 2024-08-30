from datasets import load_dataset

from base_model import ShopBenchBaseModel


if __name__ == "__main__":
    # 准备PTB数据集
    # dataset = load_dataset("ptb_text_only", "penn_treebank")
    # test_data = dataset["test"]["sentence"]
    # print(test_data[0])

    ckpt_dir = "/home/jim/nas/lilxiaochen/kdd_cup_v2/models/llama3/Meta-Llama-3-8B-Instruct"
    model = ShopBenchBaseModel(ckpt_dir)

    # perplexity = model.calculate_perplexity(test_data)
    # print(f"Perplexity on PTB test set: {perplexity}")
    while True:
        ins = input(":")
        if ins == "q":
            break

        try:
            eval(ins)
        except:
            pass