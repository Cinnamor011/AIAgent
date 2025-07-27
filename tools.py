#Tools section - One for wikipedia, one for duckduckgo, one custom tool

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

#Save to file python function
#Take in datatype and filename
#data is pydantic model
def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

#Wrap save_to_txt file as tool
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a txt file.",
)

#tool can search internet
search = DuckDuckGoSearchRun()

#tool name is called search (no spaces)
#function called run
#Description for the tool
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information"
)

#Wikipedia tool
#Pass top 1 result
#Pass through 100 characters max from doc content

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

#convert into tool, passes api wrapper. Can pass as a tool directly to langchain
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

#save into file with python functoin
