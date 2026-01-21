import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num: float) -> float:
    """
    parameter num: a number to triple
    returns: the triple of the input number
    """
    return float(num * 3)

tools = [TavilySearch(max_results=1), triple]

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    temperature=0,
)

model = llm.bind_tools(tools)

if __name__ == "__main__":
    result = model.invoke("What is the triple of 5?")
    print("-" * 100)
    print("result: ", result)
    print("-" * 100)
    print("content: ", result.content)
    print("-" * 100)
    print("tool_calls: ", result.tool_calls)
    print("-" * 100)