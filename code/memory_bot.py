# 8.1 节代码: memory_bot.py
import os
import time
from langchain_community.llms import QianfanLLMEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory # <-- 导入新的记忆模块

# --- Part 1: setup ---
#从之前设置的环境变量中安全地加载API密钥等信息
#假设 QIANFAN_ACCESS_KEY 和 QIANFAN_SECRET_KEY 已在终端通过 source -/.bashrc 设置
os.environ["QIANFAN_ACCESS_KEY"] = os.getenv("QIANFAN_ACCESS_KEY")
os.environ["QIANFAN_SECRET_KEY"] = os.getenv("QIANFAN_SECRET_KEY")

# --- Part2: 组装AI核心与记忆 ---

#积木1： Model (AI 的大脑)
llm = QianfanLLMEndpoint(model="ERNIE-Bot-4")

# 新积木： Memory (AI的记事本)
# 我们创建记忆实例，并指定 memory_key 为"chat_history"
memory = ConversationBufferMemory(memory_key="chat_history")

# 积木2： Prompt (更智能的指令)
#提示词现在需要两个输入变量：对话历史和用户问题
prompt = PromptTemplate(
    # 关键：新增 chat_history 作为输入变量
    input_variables=["chat_history", "question"],
    template="""你是一位乐于助人的AI助手。以下是你和人类的对话历史。
    
    对话历史:
    {chat_history}
    
    人类: {question}
    AI:"""
)

# 积木 3: Chain (连接器)
# 关键: 创建链时，我们现在传入memory对象，赋予链记忆能力
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True # 开启 verbose, 让我们看到幕后发生了什么
)

# --- Part 3: 对话循环 ---
if __name__ == "__main__":
    print("你好!我是你拥有记忆的Q&A机器人。")
    print("输入 'exit' 即可结束对话。")

    while True:
        user_question = input("\n你: ")

        if user_question.lower() == "exit":
            print("机器人: 再见！很高兴与你聊天。")
            break

        try:
            #使用 .predict() 方法运行链，LangChain 会自动处理记忆的存取
            print("机器人正在思考...")
            start_time = time.time()
            # 关键： 我们只需要传入 question, chat_history 会被memory 自动注入
            response = chain.predict(question=user_question)
            end_time = time.time()
            print(f"思考耗时：{end_time - start_time:.2f}秒")

            print("机器人： " + response)

        except Exception as e:
            print(f"机器人：抱歉，连接失败或发生错误。错误信息：{e}") 