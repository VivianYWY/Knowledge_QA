from langchain_community.utilities import SQLDatabase
#from langchain_openai import OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

# 连接到数据库
db = SQLDatabase.from_uri("sqlite:///demo.db")

# llm
# 修改为配置的OPENAI_API_KEY
api_key = "EMPTY"

# 修改为服务地址和端口
api_url = "http://127.0.0.1:1110/v1"

model= "qwen14B_int4"
#model = "deepseek14B"


#llm = OpenAI(model_name=model,openai_api_key=api_key,openai_api_base=api_url)
llm = ChatOpenAI(model_name=model,openai_api_key=api_key,openai_api_base=api_url)

# 创建SQL Agent

def _handle_error(error) -> str:
    return str(error).split("</think>")[1]

if "deepseek" in model:
    agent_executor = create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=db, llm=llm),
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            #handle_parsing_errors=_handle_error,
            handle_parsing_errors="Check your output and make sure it conforms! Do not output an action and a final answer at the same time."
            )
else:
    agent_executor = create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=db, llm=llm),
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )

# 使用Agent执行SQL查询
questions = [
#    "总共有多少用户？",
   #  "不重复的用户总共有多少？",
#    "哪个用户的年龄最大？",
   # "所有用户的金额合计是多少？",
    "金额排名前三的用户是哪些？"
]

# 循环提问
for question in questions:
    response = agent_executor.invoke(question)
    print(response)

