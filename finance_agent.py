import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


from openai import OpenAI
import json

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import pandas as pd
from io import StringIO

# ===== Agent State and Prompts Setup ===== 

# Load environment variables from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")

llm = "gpt-4.1-nano-2025-04-14"

model = ChatOpenAI(api_key=openai_key, model=llm)

from tavily import TavilyClient

tavily = TavilyClient(api_key=tavily)

from typing import List
# Using update version of pydantic for structured data
from pydantic import BaseModel

# Define the state of the agent
class AgentState(TypedDict):
    task: str # The current task description
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int

# Structured data container that helps in managing the agent's state
class Queries(BaseModel):
    queries: List[str]

# Define the prompts for each node - IMPROVE AS NEEDED
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Gather the financial data for the given company. Provide detailed financial data."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Analyze the provided financial data and provide detailed insights and analysis."""
RESEARCH_COMPETITORS_PROMPT = """You are a researcher tasked with providing information about similar companies for performance comparison. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""
COMPETE_PERFORMANCE_PROMPT = """You are an expert financial analyst. Compare the financial performance of the given company with its competitors based on the provided data. 

Here is the competitor information:
{content}

**MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
FEEDBACK_PROMPT = """You are a reviewer. Provide detailed feedback and critique for the provided financial comparison report. Include any additional information or revisions needed."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a comprehensive financial report based on the analysis, competitor research, comparison, and feedback provided."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""

# ======= Creating All Nodes =======
# This node gathers financial data from a CSV file and prepares it for analysis.
def gather_financials_node(state: AgentState):
    # Read the CSV file into a pandas DataFrame
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file)) # Assuming csv_file is a string containing CSV data

    # Convert the DataFrame to a string
    financial_data_str = df.to_string(index=False)

    # Combine the financial data string with the task
    combined_content = (
        f"{state['task']}\n\nHere is the financial data:\n\n{financial_data_str}"
    )

    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content),
    ]

    response = model.invoke(messages)
    return {"financial_data": response.content}

# This node analyzes the financial data gathered in the previous step.
def analyze_data_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["financial_data"]),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}

# This node researches competitors based on the task description and the analysis provided.
def research_competitors_node(state: AgentState):
    content = state.get("content", []) # Initialize content if it's None
    for competitor in state["competitors"]:
        # Generate queries to research competitors
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
                HumanMessage(content=competitor),
            ]
        )
        # loops through the generated queries and performs searches using Tavily
        for q in queries.queries:
            # For each query, perform a search using Tavily
            response = tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
    return {"content": content}

# This node compares the financial performance of the company with its competitors.
def compare_performance_node(state: AgentState):
    content = "\n\n".join(state.get("content", []))
    # Creates a HumanMessage object with the task and analysis
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
    )
    messages = [
        # Add placeholder for the system message
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }

# This node critiques the financial comparison report.
def research_critique_node(state: AgentState):
    # Generate queries to research the critique provided by the user
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["feedback"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}

# This node collects feedback on the financial comparison report.
def collect_feedback_node(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}

# This node writes the final financial report based on the analysis, competitor research, comparison, and feedback.
def write_report_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"report": response.content}

# check if the agent should continue or end based on the number of revisions
def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "collect_feedback"
# ========== End of Node Definitions ==========

# ===== Building the StateGraph =====
builder = StateGraph(AgentState)

builder.add_node(
    "gather_financials",
    gather_financials_node,
)
builder.add_node(
    "analyze_data",
    analyze_data_node,
)
builder.add_node(
    "research_competitors",
    research_competitors_node,
)
builder.add_node(
    "compare_performance",
    compare_performance_node,
)
builder.add_node(
    "collect_feedback",
    collect_feedback_node,
)
builder.add_node(
    "research_critique",
    research_critique_node,
)
builder.add_node(
    "write_report",
    write_report_node,
)

builder.set_entry_point("gather_financials")

# Add conditional edges
builder.add_conditional_edges(
    "compare_performance",
    should_continue,
    {END: END, "collect_feedback": "collect_feedback"},
)

# Add edges between nodes
edges = [
    ("gather_financials", "analyze_data"),
    ("analyze_data", "research_competitors"),
    ("research_competitors", "compare_performance"),
    ("collect_feedback", "research_critique"),
    ("research_critique", "compare_performance"),
    ("compare_performance", "write_report"),
]
for source, target in edges:
    builder.add_edge(source, target)
# ========== End of StateGraph Building ==========

# ========== Compile the Graph ==========
# ![NOTE] SqliteSaver.from_conn_string is a context manager that creates a new SQLite database in memory.
with SqliteSaver.from_conn_string(":memory:") as memory:
    graph = builder.compile(checkpointer=memory)

    # ==== For Console Testing ====
    # def read_csv_file(file_path):
    #     with open(file_path, "r") as file:
    #         print("Reading CSV file...")
    #         return file.read()


    # if __name__ == "__main__":
    #     task = "Analyze the financial performance of our (MegaAICo) company compared to competitors"
    #     competitors = ["Microsoft", "Nvidia", "Google"]
    #     csv_file_path = (
    #         "./data/financials.csv" 
    #     )

    #     if not os.path.exists(csv_file_path):
    #         print(f"CSV file not found at {csv_file_path}")
    #     else:
    #         print("Starting the conversation...")
    #         csv_data = read_csv_file(csv_file_path)

    #         initial_state = {
    #             "task": task,
    #             "competitors": competitors,
    #             "csv_file": csv_data,
    #             "max_revisions": 2,
    #             "revision_number": 1,
    #         }
    #         config = {"configurable": {"thread_id": "1"}}

    #         for s in graph.stream(initial_state, config):
    #             print(s)
    # === End Console Testing ===

    # ======= Streamlit Integration =======
    import streamlit as st


    def main():
        st.title("Financial Performance Reporting Agent")

        task = st.text_input(
            "Enter the task:",
            "Analyze the financial performance of our (MegaAICo) company compared to competitors",
        )
        competitors = st.text_area("Enter competitor names (one per line):").split("\n")
        max_revisions = st.number_input("Max Revisions", min_value=1, value=2)
        uploaded_file = st.file_uploader(
            "Upload a CSV file with the company's financial data", type=["csv"]
        )

        if st.button("Start Analysis") and uploaded_file is not None:
            # Read the uploaded CSV file
            csv_data = uploaded_file.getvalue().decode("utf-8")

            initial_state = {
                "task": task,
                "competitors": [comp.strip() for comp in competitors if comp.strip()],
                "csv_file": csv_data,
                "max_revisions": max_revisions,
                "revision_number": 1,
            }
            thread = {"configurable": {"thread_id": "1"}}

            final_state = None
            for s in graph.stream(initial_state, thread):
                st.write(s)
                final_state = s

            if final_state and "report" in final_state:
                st.subheader("Final Report")
                st.write(final_state["report"])


    if __name__ == "__main__":
        main()
    # ==== End Streamlit Integration ====
# ========== End of Graph Compilation ==========


# ========== Visualize the Graph ==========
# Note: The graph visualization part is optional and requires additional dependencies.
from IPython.display import Image, display

try:
    # Get the current working directory
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, "Finance_agent_graph.png")

    # Save the graph to a file
    png_data = graph.get_graph().draw_mermaid_png()

    with open("Finance_agent_graph.png", "wb") as f:
        f.write(png_data)

    print("Graph saved as Finance_agent_graph.png")
except Exception as e:
    # This requires some extra dependencies and is optional
    print("Error saving graph:", e)
# ========== End of Graph Visualization ==========

