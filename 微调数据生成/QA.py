import os
import json
from typing import List
from tqdm import tqdm
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 设置 Moonshot API Key
os.environ["MOONSHOT_API_KEY"] = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ1c2VyLWNlbnRlciIsImV4cCI6MTczMTA1ODk1NiwiaWF0IjoxNzIzMjgyOTU2LCJqdGkiOiJjcXJqYzMzamZpaDRxNWo5MzlhMCIsInR5cCI6InJlZnJlc2giLCJzdWIiOiJjbzVuMWptY3A3ZjJocDY1bGpkMCIsInNwYWNlX2lkIjoiY281bjFqbWNwN2YyaHA2NWxqY2ciLCJhYnN0cmFjdF91c2VyX2lkIjoiY281bjFqbWNwN2YyaHA2NWxqYzAifQ.st1I1LchZqPkDPLo97II-9qHUR9SE_cV19FlfS1BabggzMb-NrKMdQ31aeT8opFitugtH6twjLD919PSfdKliA"

# 文档处理函数
def split_document(filepath):
    loader = UnstructuredFileLoader(filepath)
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=128
    )
    documents = loader.load_and_split(text_spliter)
    return documents

# Prompt部分
QA_PAIRS_SYSTEM_PROMPT = """  
<Context></Context> 标记中是一段文本，学习和分析它，并整理学习成果：  
- 提出问题并给出每个问题的答案。  
- 答案需详细完整，尽可能保留原文描述。  
- 答案可以包含普通文字、链接、代码、表格、公示、媒体链接等 Markdown 元素。  
- 最多提出 30 个问题。  
"""

QA_PAIRS_HUMAN_PROMPT = """  
请按以下格式整理学习成果:
[
  {{
    "instruction": "人类指令/问题（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）"
  }}
]
下面的填充是示例:  
[
  {{
    "instruction": "识别并解释给定列表中的两个科学理论：细胞理论和日心说。",
    "input": "",
    "output": "细胞理论是生物科学的一个理论，它认为所有生命体都是由微小的基本单元——细胞所构成。这是生物学的一个基础理论，认为细胞是所有生物的基本结构和功能单位，所有的生物都是由一个或多个细胞组成，细胞只能通过细胞分裂产生新的细胞。"
  }},
  {{
    "instruction": "输入三支篮球队的名称并生成一个适当的口号。",
    "input": "输入：俄克拉荷马城雷霆队，芝加哥公牛队，布鲁克林网队。",
    "output": "输出： 雷霆，公牛和网队：各显神通，角逐群雄!"
  }},  
  {{
    "instruction": "请根据以下文章构建一个需要回答的问题。",
    "input": "",
    "output": "我不好意思，您没有提供任何信息或文章来供我根据它构建问题。 请提供相应的文章或信息，以便我能够帮助您构建问题。"
  }}
]  

我们开始吧!  

<Context>  
{text}  
<Context/>
"""

# LLM部分替换为MoonshotChat
def create_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_PAIRS_SYSTEM_PROMPT),
        ("human", QA_PAIRS_HUMAN_PROMPT)
    ])
    llm = MoonshotChat(model="moonshot-v1-128k")  # 使用MoonshotChat
    parser = JsonOutputParser(pydantic_object=QaPairs)
    chain = prompt | llm | parser
    return chain

# 结果模型定义
class QaPair(BaseModel):
    instruction: str = Field(description='问题内容')
    #"input": "人类输入（选填）",
    input: str = Field(description='人类输入（针对问题内容,选填）')
    output: str = Field(description='问题的回答')

class QaPairs(BaseModel):
    qas: List[QaPair] = Field(description='问答对列表')




def main():
    chain = create_chain()
    documents = split_document('data/12.txt')  # 替换为你的文档路径

    # # 打印分块结果
    # for i, doc in enumerate(documents):
    #     print(f"Document chunk {i + 1}:")
    #     print(doc.page_content)
    #     print("-" * 80)

    # 继续处理文档
    with open('dataset.json', 'a', encoding='utf-8') as f:  # 打开文件，使用 'a' 模式进行追加写入
        bar = tqdm(total=len(documents))
        for idx, doc in enumerate(documents):
            print(doc.page_content)
            # 调试API响应
            print(f"Processing document chunk {idx + 1}")
            out = chain.invoke({'text': doc.page_content})
            print(f"API response for chunk {idx + 1}: {out}")

            # 无论返回什么，直接写入文件
            f.write(json.dumps(out, ensure_ascii=False, indent=2) + ",\n")  # 实时写入并添加换行
            f.flush()  # 确保数据立即写入磁盘
            bar.update(1)
        bar.close()

if __name__ == '__main__':
    main()






