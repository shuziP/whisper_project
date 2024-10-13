import os
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

file_path = r'D:\project\zp_llm_key.txt'    #apikey存放地址


try:  
    with open(file_path, 'r', encoding='utf-8') as file:  
        openai_api_key = file.read() 
except FileNotFoundError:  
    print(f"文件 {file_path} 未找到，请检查路径是否正确。")

def call_glm(model="glm-4-flash", temperature=0.95, openai_api_key=openai_api_key, user_input="你好"):
    if not openai_api_key:
        print("API key is missing.")
        return "API key is missing."

    if not user_input.strip():
        print("User input cannot be empty.")
        return "User input cannot be empty."

    print(f"正在调用模型: {model}，温度设定: {temperature}")

    try:
        llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=openai_api_key,
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
        )

        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "你是一个有用的助手："
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )

        messages = [
            ("system", "你是一个有用的助手"),
            ("human", user_input),
        ]

        # 调用模型
        ai_msg = llm.invoke(messages)
        print("模型返回结果:", ai_msg)

        return ai_msg.content

    except Exception as e:
        # 捕获所有异常并返回错误信息
        error_message = f"模型调用出错: {str(e)}"
        print(error_message)
        return error_message

if __name__ == '__main__':
    call_glm(user_input= "你能介绍一下你自己吗？")
