# 8.2 节代码: tutor_bot.py
import os
import time
from langchain_community.llms import QianfanLLMEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory # <-- 导入新的记忆模块

# --- Part 1: setup ---
os.environ["QIANFAN_ACCESS_KEY"] = os.getenv("QIANFAN_ACCESS_KEY")
os.environ["QIANFAN_SECRET_KEY"] = os.getenv("QIANFAN_SECRET_KEY")

# --- Part2: 组装AI Tutor的核心 ---

#积木1&2： Model和memory 保持不变
llm = QianfanLLMEndpoint(model="ERNIE-Bot-4")
memory = ConversationBufferMemory(memory_key="chat_history")

# 积木3：The Prompt (AI的人设和职位描述)
#这是我们新的，强大的提示词模板！
template ="""
你是一位名叫Alex的友好,耐心且善于鼓励的英语导师。
你的目标是帮助非英语母语使用者练习英语口语。
你必须总是提出一个后续问题，以保持对话的进行。
保持你的回答简洁，通常一到两句话。
如果用户犯了语法错误或拼写错误，你必须温和地纠正它。首先，提供用户句子的修正版本，然后，在新的一段中，提供一个简单、一句话的解释。
不要修正与大小写或标点相关的问题，因为重点在于口语。

当前对话：
{chat_history}
Human: {question}
AI:
"""


prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=template
)

# 积木 4: The Chain (连接器)
# 链的结构保持不变，但它现在使用我们新的、复杂的提示词
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False# 关闭 verbose, 以获得更自然的聊天体验
)

# --- Part 3: 对话循环 ---
if __name__ == "__main__":
    print("你好!我是你拥有记忆的 Alex 机器人。")
    print("输入 'exit' 即可结束对话。")

    while True:
        user_question = input("\n你: ")

        if user_question.lower() == "exit":
            print("Alex: 再见！很高兴与你聊天。")
            break

        try:
            print("Alex正在思考...")
            start_time = time.time()
            response = chain.predict(question=user_question)
            end_time = time.time()
            print(f"思考耗时：{end_time - start_time:.2f}秒")

            print("Alex: " + response)

        except Exception as e:
            print(f"Alex:抱歉，连接失败或发生错误。错误信息：{e}") 