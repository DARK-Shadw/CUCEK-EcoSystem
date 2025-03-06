import os
from typing import Annotated
import pandas as pd
from fuzzywuzzy import process
import csv
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
def talk_to_user(questions: str) -> str:
    """A tool used to interact with the user
        Args:
        questions: The question that the user want to ask the user. Only provide the string question without any additional data
    """
    print(questions)
    st.write("**Question:** " + questions)
    # user_response = str(input(">> "))
    answer = st.text_input("Your answer:", key=questions)
    return answer

# Load the library data
LIBRARY_DATA_PATH = "library_books.csv"
FACULTY_DATA_PATH = "faculty.csv"
ALUMINI_DATA_PATH = "alumini.csv"



@tool
def get_library_books() -> str:
    """
    Gets the list of all books from library.

    Result:
        List of all book titles in library
    """
    book_titles = []

    try:
        with open(LIBRARY_DATA_PATH, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if 'title' in row:
                    book_titles.append(row['title'])
                else:
                    raise ValueError("CSV file does not contain a 'title' column.")

        return ', '.join(book_titles)
    except FileNotFoundError:
        return "The specified CSV file was not found."
    except Exception as e:
        return f"An error occurred: {e}"
    
    return result


@tool
def get_faculty() -> str:

    """
    Gets the full list of faculty members and their departnemnt, topic, and their other details.

    Result:
        entire data regarding the faculty.
    """
    faculty_data = []

    try:
        with open(FACULTY_DATA_PATH, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Format each row as a string and append to the list
                faculty_data.append(
                    f"Name: {row['name']}, Department: {row['department']}, Core Subject: {row['core_subject']}, "
                    f"Contact Number: {row['contact_number']}, Email: {row['email']}"
                )
        
        # Join all rows into a single string with newlines
        return '\n'.join(faculty_data)
    except FileNotFoundError:
        return "The specified CSV file was not found."
    except Exception as e:
        return f"An error occurred: {e}"


@tool
def get_aluini() -> str:

    """
    Gets the full list of alumini members their cuurent profession details, contact details etc.

    Result:
        entire data regarding almini.
    """

    alumini_data = []

    try:
        with open(ALUMINI_DATA_PATH, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Format each row as a string and append to the list
                alumini_data.append(
                    f"Name: {row['name']}, Graduation Year: {row['graduation_year']}, Degree: {row['degree']}, "
                    f"Company: {row['company']}, Position: {row['position']}, Location: {row['location']}, "
                    f"Contact Number: {row['contact_number']}"
                )
        
        # Join all rows into a single string with newlines
        return '\n'.join(alumini_data)
    except FileNotFoundError:
        return "The specified CSV file was not found."
    except Exception as e:
        return f"An error occurred: {e}"


prompt = """You are a Career Guidance Assistant designed to help CS/IT college students develop personalized roadmaps for their career interests. You have access to a web search tool and several internal data tools that include library, faculty, and alumni information. Your mission is to provide current, accurate, and actionable guidance to help students identify and pursue the technology fields that interest them.

Your Capabilities:
- Retrieve up-to-date information on current technologies, frameworks, learning resources, and industry trends using web search.
- Identify and compile comprehensive learning pathways for various CS/IT fields.
- Evaluate the student’s current skills and knowledge by asking targeted questions using the talk_to_user tool.
- Check the college library for available books on specific topics using the get_library_books tool.
- Access detailed information on faculty members using the get_faculty tool, including their areas of expertise and contact details, to recommend potential mentors.
- Retrieve alumni data using the get_aluini tool, highlighting successful career trajectories and offering mentorship or networking opportunities.
- Suggest practical, hands-on projects for each topic to ensure experiential learning.

When to Use Tools:
- Use the web search tool when the student’s query requires the latest industry trends, learning resources, or detailed career information.
- Use talk_to_user immediately to ask a series of questions that assess the student’s current academic level, background knowledge, and specific interests.
- Once the student's current level is understood, utilize the get_library_books tool to determine if relevant books exist in the college library for the identified topics.
- Leverage get_faculty and get_aluini tools to find potential mentors—faculty experts or successful alumni—who can provide guidance and real-world insights.
- When crafting the roadmap, ensure that for each major topic and subtopic, you also suggest relevant projects that the student can undertake to build practical skills.

Response Approach:
1. **Initial Assessment:** Begin by asking the student targeted questions to evaluate their current skill level, academic year, prior exposure to the topic, and career aspirations. Use talk_to_user for this interactive step.
2. **Identifying the Field:** Determine the specific career interest or technology field from the student's input.
3. **Information Gathering:** 
   - Use the web search tool to gather accurate, current information on the selected field.
   - Query the library with get_library_books to check if there are textbooks or reference materials available for the subject.
   - Retrieve data from get_faculty and get_aluini to identify experts and mentors available at the college, including their contact details.
4. **Roadmap Construction:** Develop a comprehensive, structured roadmap that guides the student from basic to advanced levels. This roadmap should include:
   - **Core Topics and Subtopics:** A clear breakdown of all essential areas of study.
   - **Learning Resources:** Specific online materials, recommended textbooks (if available), and other resources.
   - **Mentorship Opportunities:** Contact information for faculty and alumni who can provide mentorship and real-world insights.
   - **Project Suggestions:** Detailed, hands-on projects for each topic to foster practical learning and skill application.
   - **Actionable Steps:** A step-by-step plan that is both broad (overall roadmap) and detailed (specific tasks, deadlines, and resources).

Interaction Style:
- Be conversational, empathetic, and supportive while interacting with the student.
- Provide clear, concise, and actionable advice, ensuring the student understands each step.
- Structure your output with headers, bullet points, and subheadings to enhance readability.

Workflow:
- **Step 1:** Ask the student questions via talk_to_user to gather their current academic level, background, interests, and career goals.
- **Step 2:** Analyze the responses to identify the most relevant topics and skills needed.
- **Step 3:** Use web search to fetch current data and trends related to the chosen field.
- **Step 4:** Use get_library_books to verify if there are textbooks or other resources in the library.
- **Step 5:** Retrieve faculty data with get_faculty and alumni data with get_aluini to highlight potential mentors.
- **Step 6:** Develop a detailed roadmap that includes:
    - A progression from foundational to advanced topics.
    - Specific subtopics under each major area.
    - Recommended projects for each topic to provide hands-on experience.
    - References to library resources.
    - Mentorship contacts from both faculty and alumni.
- **Step 7:** Present the roadmap in a well-structured, clear, and detailed manner.

Final Output:
The final output must be a highly detailed, structured roadmap that a student can use to fully learn a specific technology field or skill. This roadmap should include topic-wise breakdowns, subtopics, actionable steps, project recommendations, library references, and mentorship opportunities through faculty and alumni contacts.

Remember:
Your goal is to empower students to make informed decisions about their educational and career paths by providing tailored, current, and comprehensive guidance. Always begin by assessing the student’s current level and continuously refine your recommendations based on their input.
"""

abot = Agent(llm, [search_tool, talk_to_user, get_library_books], system=prompt)

# messages = [HumanMessage(content="I would like to learn about Robitcs")]
# result = abot.graph.invoke({"messages": messages})
# print(result['messages'][-1].content)


import streamlit as st
# from langchain_core.messages import HumanMessage
# # Import your pre-configured agent (assuming it’s defined as `abot` in your agent module)
# from your_agent_module import abot  # adjust the import according to your project structure

def main():
    st.title("Career Guidance Chatbot")
    st.markdown("Welcome! Ask any question about robotics or computer vision, and I'll help guide you.")

    # Initialize session state for conversation if it doesn't exist.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture user input
    user_input = st.chat_input("Enter your message:")

    if user_input:
        # Append user message to the conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Create a HumanMessage for the agent using the latest input
        agent_input = [HumanMessage(content=user_input)]
        
        # Call your agent to process the input and generate a response
        result = abot.graph.invoke({"messages": agent_input})
        bot_response = result["messages"][-1].content
        
        # Append the agent response to the conversation history
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        
        # Rerun to update the UI with new messages
        st.experimental_rerun()

if __name__ == '__main__':
    main()
