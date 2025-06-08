import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults


# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")

llm = "gpt-4.1-nano-2025-04-14"

model = ChatOpenAI(api_key=openai_key, model=llm)

# ----- Set up a AI agent with tools -----
# STEP 1: Build a Basic Chatbot
from langgraph.graph.message import add_messages
# add_messages specifies how this should be updated when the state changes

# Define the state of the agent
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages] # Annotated allows to add metadata

# Create a graph builder
graph_builder = StateGraph(State)

# STEP 2: Create tools
searchtool = TavilySearchResults(max=2)
tools = [searchtool] # List of tools to be used by the agent, list type so it can be extended later
# res = tool.invoke("What is the capital of France?")
# print(res)

# Prebuilt toolNode function
import json
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition



model_with_tools = model.bind_tools(tools)
# res = model_with_tools.invoke("What's a 'node' in LangGraph?")
# print(res)

# Create bot
def bot(state: State):
    # The bot function is the first node in the graph. It takes the state as input
    # and returns a response based on the messages in the state.
    print(state["messages"])
    return {"messages": [model_with_tools.invoke(state["messages"])]}

# Instantiate the ToolNode with the tools
tool_node = ToolNode(tools=[searchtool])
graph_builder.add_node("tools", tool_node)

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
)

# STEP 3: build the first node for the graph

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("bot", bot)

# Step 4: add an entry point and end to the graph
graph_builder.set_entry_point("bot")  # it could be any node

# ADD MEMORY "NODE"
from langgraph.checkpoint.sqlite import SqliteSaver


# ========== MEMORY CODE STARTS HERE ==========
# ![NOTE] SqliteSaver.from_conn_string is a context manager that creates a new SQLite database in memory.
with SqliteSaver.from_conn_string(":memory:") as memory:
    # Step 5: Compile the graph
    graph = graph_builder.compile(checkpointer=memory)
    # MEMORY CODE CONTINUES ===
    # Now we can run the chatbot and see how it behaves
    # PICK A TRHEAD FIRST
    config = {
        "configurable": {"thread_id": 1}
    }  # a thread where the agent will dump its memory to

    # ---- Test memory
    user_input = "Hi there! My name is Bond. and I have been happy for 100 years"

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )

    for event in events:
        event["messages"][-1].pretty_print()  # Print the last message in the event

    user_input = "do you remember my name, and how long have I been happy for?"

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )

    for event in events:
        event["messages"][-1].pretty_print()


    snapshot = graph.get_state(config)
    print("Snapshot:\n", snapshot)

# res = graph.invoke({"messages": ["Hello, how are you?"]})
# print(res)

# ----- EXTRA: Print the graph structure -----
from IPython.display import Image, display

try:
    # Get the current working directory
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, "prebuilt_agent_lngraph_tools.png")

    # Save the graph to a file
    png_data = graph.get_graph().draw_mermaid_png()

    with open("prebuilt_agent_lngraph_tools.png", "wb") as f:
        f.write(png_data)

    print("Graph saved as prebuilt_agent_lngraph_tools.png")
except Exception as e:
    # This requires some extra dependencies and is optional
    print("Error saving graph:", e)


# Step 6: Stream the graph
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values"):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break

#         stream_graph_updates(user_input)
#     except:
#         # fallback if input() is not available
#         user_input = "You couldn't get my input"
#         print("User: " + user_input)
#         stream_graph_updates(user_input)
#         break


