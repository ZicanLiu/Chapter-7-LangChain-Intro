# 7.2 节代码：naming_consultant.py
import os
import time
from langchain_community.llms import QianfanLLMEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#从之前设置的环境变量中安全地加载API密钥等信息
#假设 QIANFAN_ACCESS_KEY 和 QIANFAN_SECRET_KEY 已在终端通过 source -/.bashrc 设置
os.environ["QIANFAN_ACCESS_KEY"] = os.getenv("QIANFAN_ACCESS_KEY")
os.environ["QIANFAN_SECRET_KEY"] = os.getenv("QIANFAN_SECRET_KEY")

def generate_company_names(product):
    # 1. 加载大模型“积木” （Models）
    # 我们明确要求加载百度文心大模型ERNIE-Bot-4.
    llm = QianfanLLMEndpoint(model="ERNIE-Bot-4")

    # 2. 创建大模型提示词“积木” （Prompts)
    # input_variables=["product"]: 我们告诉模板，它期待一个名为 product 的用户输入
    prompt_template = PromptTemplate(
        input_variables=["product"],
        template="你是一位富有经验的，给新创公司命名的专家。如果某个公司生产{product}, 你能否给这个公司起三个有创意的名字？"
    )

    # 3. 使用链条“积木”做连接（Chains)
    # LLMChain 工具将被加载的大模型和提示词模板连接起来。
    name_chain = LLMChain(llm=llm, prompt=prompt_template)

    print("AI命名顾问正在思考中,请稍候...")
    start_time = time.time()

    #4.运行Chain(invoke)
    #我们调用Chain的invoke方法，将实际的用户输入作为字典传递给它
    response = name_chain.invoke({"product": product})
    end_time = time.time()
    print(f"思考完成，耗时：{end_time - start_time:.2f}秒")

    #获取大模型生成的内容
    return response["text"]

def new_func(product, name_chain):
    response = name_chain.invoke({"product":product})
    return response
# 以下是主程序运行逻辑，接收用户输入并打印结果
if __name__ == "__main__":
    print("欢迎使用公司命名服务！")
    company_product = input("您的公司生产什么产品？")

    try:
        ai_names = generate_company_names(company_product)

        print("\n以下是我们给出的一些有创意的公司名字:")
        print("------------------------------")
        print(ai_names)

    except Exception as e:
        print(f"\n发生错误,无法连接到大模型或密钥无效。错误信息:{e}")
    
