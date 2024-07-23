import os
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from langchain.document_loaders import UnstructuredFileLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from base_model import ShopBenchBaseModel


def load_knowledge_base(filepath):
    # 加载外部知识库
    loader = UnstructuredFileLoader(filepath, separator="\n")  # 把带格式的文本，读取为无格式的纯文本
    # loader = JSONLoader(filepath, jq_schema='.[]')
    docs = loader.load()

    # 对读取的文档进行chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents(docs)

    return docs

def load_vector_store(model_name, filepath=None, from_documents=True):
    # 使用text2vec模型，对上面chunk后的doc进行embedding。然后使用FAISS存储到向量数据库
    embeddings = HuggingFaceEmbeddings(model_name=model_name, #"/root/autodl-tmp/text2vec-large-chinese",
                                       model_kwargs={'device': 'cuda'})

    # 注意：如果修改了知识库（knowledge.txt）里的内容，则需要把原来的 my_faiss_store.faiss 删除后，重新生成向量库。
    if from_documents:#"/root/autodl-tmp/knowledge.txt"
        docs = load_knowledge_base(filepath)
        vector_store = FAISS.from_documents(docs, embeddings)

    else:#"/root/autodl-tmp/my_faiss_store.faiss"
        vector_store = FAISS.load_local(filepath, embeddings=embeddings)

    return vector_store


def create_inputs(query, vector_store):
    # 向量检索：通过用户问句，到向量库中，匹配相似度高的文本
    docs = vector_store.similarity_search(query)  # 计算相似度，并把相似度高的chunk放在前面
    context = [doc.page_content for doc in docs]  # 提取chunk的文本内容

    # 4.2.7 构造prompt_template
    my_input = "\n".join(context)
    input = f"已知: \n{my_input}\n请回答: {query}"
    # print(input)
    return input

def json2txt(src_file, dst_file):
    with open(src_file, "r") as f:
        data_lst = json.load(f)

    res = []
    for data_dict in data_lst:
        data_res = []
        for key, val in data_dict.items():
            data_res.append(key + "是" + val)
        res.append("，".join(data_res))

    with open(dst_file, "w", encoding="utf8") as f:
        f.write("\n".join(res))


if __name__ == "__main__":
    ckpt_dir = "/home/jim/nas/lilxiaochen/kdd_cup_v2/models/llama3/Meta-Llama-3-8B-Instruct"
    text2vec_model = "/media/ssd/yzg/all-MiniLM-L6-v2"
    knowledge_file = "/media/ssd/yzg/data/dual_data.txt"
    json2txt("/media/ssd/yzg/data/dual_data.json", knowledge_file)
    vector_store = load_vector_store(text2vec_model, knowledge_file)
    model = ShopBenchBaseModel(ckpt_dir)

    # query = "公会是破晓之星的角色的名字"
    instruction = "根据已知的信息回答问题。如果信息不足，无法回答，则回复不知道。"
    while True:
        query = input("query: ")
        if query == "q":
            break
        inputs = create_inputs(query, vector_store)
        response = model.predict(instruction, inputs)
        print(response)
