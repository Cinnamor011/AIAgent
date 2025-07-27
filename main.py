"""
7/27/25
This is a basic AI research agent using claude. This project features 3 tools. User prompts
what they want to research about and the agent structures the input into a model. 
Note: There is no API key for this project :( I know I'm sad too... This project is just for demonstration
purposes only. 
"""
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

#load environment variable file (.env)
load_dotenv()

#class ResearchResponse inherits from BaseModel
#Specifies all of the fields that I want as output from LLM call
class ResearchResponse(BaseModel):
    #response generates a topic of type string
    topic: str
    #response generates a summary of type string
    summary: str
    #response generates a list of sources of type string
    sources: list[str]
    #response generates a list of tools of type string
    tools_used: list[str]

#chose LLM, specifies the model used, api key used
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022, '''api key''' ")

#takes output of LLM and parses it into class ResearchResponse
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

#system message is information to the LLM so it knows what it's supposed to be doing
#partially going to fill into this prompt by passing the format instructions. Uses
#parser and takes pydantic model, turns it into a string that then gives to the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
#creates the agent from langchain
#passes llm, prompt, and tools
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)
"""
invoke llm run locally from computer
response = llm.invoke("What is the meaning of life?")
print(response)
"""
#Tests agent
#verbose=True sees thought process from agent, to disable verbose=False
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("What can I help you research? ")

#Use agent executor to generate response
#Pass in prompt variable, note they're encolsed in {}, can include multiple variables
# ex: "human", "{query}, {name}"), in prompt
# ({"query": "What is the capital of Texas?", "name": "Alice})

raw_response = agent_executor.invoke({"query": "What is the capital of Texas?"})
print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)