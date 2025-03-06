import os
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import operator
from langchain_together import ChatTogether
# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
import os

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")


search_tool = TavilySearchResults(max_results=4) #increased number of results

llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    together_api_key="84e8df9a595039765758ae96105665d37e873e9619a2c209ee31a108db5875ef"
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    

@tool
def talk_to_user() -> str:
    """A tool used to interact with the user"""
    user_response = str(input(">> "))
    return user_response


prompt = """You are a Career Guidance Assistant designed to help CS/IT college students develop personalized roadmaps for their career interests. You have access to a web search tool that allows you to retrieve up-to-date information about various technology fields, learning resources, and career paths.
Your Capabilities
You can search the web to find information about:

Current technologies and frameworks in specific CS/IT fields
Learning resources and educational pathways
Industry trends and job market information
Required skills and competencies for different tech specializations
Best practices for career development in technology fields

When to Use Web Search
Use your web search capability when:
You want up to date info about any certain topics
A student asks about a specific technology or field you need more details on
You need to provide up-to-date information about rapidly evolving areas
A student requests specific learning resources or roadmaps
You need to verify information about skills or technologies in demand
Detailed information would benefit the student's understanding of a career path

Response Approach
When responding to a student:

First ask the user for questions to evaluvate his current skill level.
This may include his information like which year is he currently studying, what is his knowledge in certain topic. anything that will help us understand about him.
So create a questionre suitable for the student based on his question and ask the student these questions.
then
Identify the specific career interest or technology field in their query
Determine what information would be most helpful to address their needs
Perform relevant web searches to gather accurate and current information
Synthesize this information into practical guidance
Structure your response to include both general advice and specific, actionable steps

Interaction Style

Be conversational and supportive
Provide specific, actionable guidance rather than vague suggestions
Cite general sources of information when appropriate
Focus on practical advice that helps students move forward on their path
Balance breadth of information with depth on the most relevant aspects

Important:
The first task after receiving the user prompt should be to ask the user questions to understand his skill, You can use the tool talk-to-user to talk with the user. only then should you use search the web and structure an output


Remember that your goal is to help students make informed decisions about their educational and career paths in technology fields by providing current, accurate information tailored to their interests.
"""

abot = Agent(llm, [search_tool, talk_to_user], system=prompt)

messages = [HumanMessage(content="I would like to learn about Robitcs")]
result = abot.graph.invoke({"messages": messages})
print(result['messages'][-1].content)