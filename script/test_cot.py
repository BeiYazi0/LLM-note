import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from base_model import ShopBenchBaseModel


def no_cot(instruction, inputs, model):
    print(inputs)
    response = model.predict(instruction, inputs)
    return response


def zero_shot(instruction, inputs, model):
    cot_inputs = "Let's work this out in a step by step way to be sure we have the right answer.\n" + inputs
    print(cot_inputs)
    response = model.predict(instruction, cot_inputs)
    return response


def few_shot(instruction, inputs, model):
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
    print(cot_inputs)
    response = model.predict(instruction, cot_inputs)
    return response


def least2most(instruction, inputs, model):
    prompt = "对以下数学问题进行问题拆解，分成几个必须的中间解题步骤并给出对应问题, 问题："
    cot_inputs = prompt + inputs
    print(cot_inputs)
    response = model.predict(instruction, cot_inputs)
    print(response)
    cot_inputs = inputs
    while True:
        q = inputs("子问题：")
        if (q == "q"):
            break
        cot_inputs += "提问：" + q
        response = model.predict(instruction, cot_inputs)
        print(response)
        cot_inputs += response


def self_consistency(instruction, inputs, model):
    example = '''问:题目:小明每天早上花费10分钟时间走到学校，如果小明家距离学校2公里，那么他每分钟走多少米?答:这是一个关于速度、路程、时间的数学问题。我们可以通过公式:速度=路程÷时间来解决。
因为小明每天早上走2公里，所以他的路程为2千米。而他每天早上要花费10分钟时间走到学校，因此他的时间为10分钟，即600秒。所以小明每分钟走的距离为2公里/ 600秒 = 0.0033公里/秒或3.3米/秒。
笞案:小明每分钟走3.3米。
问:题目:今天小明骑自行车从家到学校用了20分钟，回家用了25分钟。如果小明在上学和回家的路上的速度一样，那么他从家到学校的距离是学校到家的距离的百分之几?
答:
假设小明家到学校的距离为x千米，根据速度等于路程除以时间的公式，可以得出小明的速度为。家到学校的速度=x》20，学校到家的速度三x》25。因为小明在上学和回家的路上的速度一样，所以有:x/ 20 = x / 25，解出x= 5/4千米。
因此，家到学校的距离是学校到家的距离的百分之几，可以通过求比值得到。 x/5/4)x =4/5=0.8，即小明从家到学校的距离是学核到家的距离的百分之80。
问:题目:
鹿妈妈买了24个苹果，她想平均分给她的3只小鹿吃，每只小鹿可以分到几个苹果?答:
鹿妈妈买了24个苹果，平均分给3只小鹿吃，那么每只小鹿可以分到的苹果数就是总苹果数除以小鹿的只数。24÷3=8
每只小鹿可以分到8个苹果。所以，答案是每只小鹿可以分到8个苹果。
问:题目:小明有18支铅笔和24张纸，他想将它们分成每份相同的组，每组既有铅笔又有纸，问他最少可以分成几组，每组有多少支铅笔和多少张纸?答:
我们可以将问题转化为求18和24的最大公约数，以得到可以被同时整除的最大数量。然后，我们可以将总数分成这个数量的组，并确保每组既有铅笔又有纸。
首先,我们可以列出18和24 的因数分解式:18 =2×3 ×3
24 =2×2×2×3
它们的公共因数是2和3，因此它们的最大公约数是2×3 = 6。
因为每组既有铅笔又有纸，所以每组的铅笔和纸的数量都必须是最大公约数的因子。在这种情况下，因子包括1、2、3和6。因此，小明可以将铅笔和纸分成6 组，每组有3支铅笔和4张纸。
问:小明有7个橙子，他想平均分给他的3个朋友。每个朋友能分到几个橙子?答:
小明手中有7个橙子，要平均分给3个朋友，我们可以用除法来解决这个问题。Step 1:将7个橙子（被除数）除以3（除数）:
7: 3 = 2-……—1
(能整除,余数为1)
Step 2:因为不能完全平均分，所以小明的3-个朋友可以得到⒉个橙子，再把剩下的一个橙子分给其中的一个朋友。
问：'''



if __name__ == "__main__":
    ckpt_dir = "/home/jim/nas/lilxiaochen/kdd_cup_v2/models/llama3/Meta-Llama-3-8B-Instruct"
    model = ShopBenchBaseModel(ckpt_dir)

    cot_dir = {"no": no_cot, "zero": zero_shot, "few": few_shot, "lm": least2most}

    instruction = "You are a helpful shop assistant,you can understand the meaning of words very well.Please respond to the questions strictly in accordance with the user's requirements.Your response should include such format: The answer is."
    # 两辆汽车从相距500千米的两城同时出发，相向而行．一辆摩托车以每小时80千米的速度在两辆汽车之间不断往返联络．已知这两辆汽车的速度分别是每小时40千米和60千米，求两汽车相遇时，摩托车共行驶了多少千米？
    while True:
        type = input("cot type: ")
        inputs = input("query: ")
        if inputs == "q":
            break
        cot_method = cot_dir.get(type, no_cot)
        response = cot_method(instruction, inputs, model)
        print(response)
