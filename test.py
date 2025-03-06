from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_together import ChatTogether


llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    together_api_key="84e8df9a595039765758ae96105665d37e873e9619a2c209ee31a108db5875ef"
)


# Create the CSV agent with allow_dangerous_code set to True
agent = create_csv_agent(
    llm=llm,
    path='library_books.csv',
    allow_dangerous_code=True,
    verbose=True
)

response = agent.run("Which books are available on robotics?")
print(response)