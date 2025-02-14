from crewai import Agent, Crew, Process, Task

from tools.llm import LLM
from tools.pine_cone_tool import rag
from tools.web_scraper import web_scraper

chat_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=2000),
    role="chat_agent",
    goal="Handle general conversation and non-furniture-related queries.",
    backstory="A friendly, knowledgeable chat agent for casual conversation.",
    allow_delegation=True,
    verbose=True,
)

chat_task = Task(
    description="""Chat with the user about: {query} for example, say hello to the users and greet them. If they asked about the temprature you can answer. Act as a general chatbot""",
    agent=chat_agent,
    expected_output="A friendly conversation response.",
)

rag_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=2000),
    role="rag_agent",
    goal="Use Pinecone to find furniture matching the user's query",
    backstory="An expert in interior design and furniture recommendations",
    tools=[rag],
    allow_delegation=True,
    verbose=True,
)

rag_task = Task(
    description="""Find furniture matching: {query}""",
    agent=rag_agent,
    expected_output="A list of furniture items",
)

scrap_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=2000),
    role="scrap_agent",
    goal=(
        "analyze furniture_listings from amazon of egypt, comparing each product's "
        "specifications against user-defined input criteria to recommend the best match."
    ),
    backstory=(
        "Leveraging advanced web serch techniques and natural language processing, "
        "this agent navigates the amazon platform to extract detailed product specifications "
        "from furniture_listings. It evaluates attributes such as design, dimensions, materials, "
        "and functionality to provide a tailored recommendation based on the provided criteria."
    ),
    tools=[web_scraper],
    context=[rag_task],
    allow_delegation=True,
    verbose=True,
)
scraper_task = Task(
    description=(
        "including dimensions, design details, material composition, and functionality. Compare these "
        "details against the following input criteria: {query} "
        "and identify the top  best fit the description."
        "if you don't found any product so say no furniture exist."
        "just show after the( ## Final Answer:) "
        "write just the results"
        "give me the url of each product that you give me in the description"
    ),
    agent=scrap_agent,
    expected_output=(
        "the best 2 fit furniture with the input description"
        "Each one in desciption format"
        "in string format not json format"
        "just show after the( ## Final Answer:) "
        "write just the results"
    ),
)

router_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=2000),
    role="Router",
    goal="""
        Your task is to analyze an incoming {query} and choose the single best specialized agent to delegate it to. You have three experts in your system:
        • The rag_agent: excels at high-confidence vector retrieval of furniture recommendations using Pinecone. It returns a score for each recommendation. 
        • The scrap_agent: specializes in dynamically extracting detailed product listings and specifications from online sources. 
        • The chat_agent: handles all other conversational or non-furniture queries.
        Evaluate the language, context, and intent of the query. 
        If the query is furniture-related, first simulate running the rag_agent:
        If the rag_agent returns one or more results with a high confidence score (e.g., above 0.7), select the rag_agent.
        If the rag_agent's scores are low (e.g., below 0.7) or no high-confidence results are found, select the scrap_agent. 
        If the query is not furniture-related or is purely conversational, choose the chat_agent.
    """,
    backstory="""
        You are the Manager Router, a highly perceptive and experienced decision-maker trained on vast amounts of specialized and general language data. 
        With deep domain expertise in natural language understanding and retrieval-augmented systems, you excel at discerning subtle contextual clues.
        Your critical role is to ensure every user query is handled by the expert best suited to provide an accurate, timely, and satisfying response.
        Your decisions improve the overall user experience by directing queries to the most capable agent, saving time and boosting precision.
    """,
    verbose=True,
    allow_delegation=True,
    # allowed_agents=["chat_agent", "rag_agent", "scrap_agent"],
)

router_task = Task(
    description="""
        When presented with a {query}, first determine its domain and intent.
        Ask yourself:
          - Is this query specifically about furniture or interior design?
          - If so, simulate running the rag_agent:
              • If the rag_agent returns one or more results with high confidence (e.g., any result with a score ≥ 0.7),
                choose "rag_agent".
              • Otherwise, if the rag_agent's scores are low or no high-confidence results are found, choose "scrap_agent".
          - If the query is not furniture-related or is purely conversational,
            choose "chat_agent".
        Return exactly one of the following strings: "rag_agent", "scrap_agent", or "chat_agent".
    """,
    agent=router_agent,
    expected_output="One of the strings: rag_agent, scrap_agent, or chat_agent",
)


crew = Crew(
    # manager_llm=LLM.llm(temperature=0),
    manager_agent=router_agent,
    agents=[chat_agent, rag_agent, scrap_agent],
    tasks=[chat_task, rag_task, scraper_task],
    # memory=True,
    process=Process.hierarchical,
)
query = "hello"
inputs = {"query": query}
result = crew.kickoff(inputs=inputs)
print(result)
# while True:

# Your response must be a single string naming the best agent from the list: 'Furniture RAG Agent', 'Web Scraping Agent', or 'General Chat Agent'. Do not include any extra commentary—simply return the chosen agent’s name.
# Return a single string naming the chosen agent: either "rag_agent", "scrap_agent", or "chat_agent".
