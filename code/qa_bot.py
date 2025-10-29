# 7.4 节代码:qa_bot.py
import os
import time
from langchain_community.llms import QianfanLLMEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#从之前设置的环境变量中安全地加载API密钥等信息
#假设 QIANFAN_ACCESS_KEY 和 QIANFAN_SECRET_KEY 已在终端通过 source -/.bashrc 设置
os.environ["QIANFAN_ACCESS_KEY"] = os.getenv("QIANFAN_ACCESS_KEY")
os.environ["QIANFAN_SECRET_KEY"] = os.getenv("QIANFAN_SECRET_KEY")

# --- 组装AI核心 ---
llm = QianfanLLMEndpoint(model="ERNIE-Bot-4")

# 提示词: 赋予AI助手角色
prompt = PromptTemplate(
    input_variables=["question"],
    template="你是一位乐于助人的AI助手。请回答以下问题: {question}"
)

#连接器: 将模型和提示词连接成一个工具
chain = LLMChain(llm=llm, prompt=prompt)

# --- 对话循环: 实现持续互动 ---
if __name__ == "__main__":
    print("你好！我是你的回答机器人。问我任何问题。")
    print("输入 'exit' 即可结束对话。")

    while True:
        user_question = input("\n你: ")

        #退出条件: 如果用户输入 'exit',则跳出循环
        if user_question.lower() == "exit":
            print("机器人: 再见！很高兴与你聊天。")
            break # break 命令会立即停止无限循环        
        try:
           # 运行 Chain, 将用户问题传递给AI
            print("机器人正在思考...")
            start_time = time.time()
            response = chain.invoke({"question": user_question})
            end_time = time.time()
            print(f"思考耗时: {end_time - start_time:.2f} 秒")

            #打印AI的回复
            print("机器人: " + response["text"])

        except Exception as e:
            print(f"机器人: 抱歉，连接失败或发生错误。错误信息: {e}")