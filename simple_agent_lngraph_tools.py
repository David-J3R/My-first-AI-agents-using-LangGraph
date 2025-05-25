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

model_with_tools = model.bind_tools(tools)
# res = model_with_tools.invoke("What's a 'node' in LangGraph?")
# print(res)

#STEP 2.1: implement a BasicToolNode that checks the most recent message
# in the state and calls tools if the message contains tool_calls
# !Note: You can replace this with the prebuilt tools_condition to be more concise
# Code from: https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/?h=basictoolnode#5-create-a-function-to-run-the-tools
import json
from langchain_core.messages import ToolMessage

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# Instantiate the BasicToolNode with the tools
tool_node = BasicToolNode(tools=[searchtool])
graph_builder.add_node("tools", tool_node)

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "bot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)



# STEP 3: build the first node for the graph
def bot(state: State):
    # The bot function is the first node in the graph. It takes the state as input
    # and returns a response based on the messages in the state.
    print(state["messages"])
    return {"messages": [model_with_tools.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("bot", bot)

# Step 4: add an entry point and end to the graph
graph_builder.set_entry_point("bot")  # it could be any node

# Step 5: Compile the graph
graph = graph_builder.compile()

# res = graph.invoke({"messages": ["Hello, how are you?"]})
# print(res)

# EXTRA: Print the graph structure
from IPython.display import Image, display

try:
    # Get the current working directory
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, "simple_agent_lngraph_tools.png")

    # Save the graph to a file
    png_data = graph.get_graph().draw_mermaid_png()

    with open("simple_agent_lngraph_tools.png", "wb") as f:
        f.write(png_data)

    print("Graph saved as simple_agent_lngraph_tools.png")
except Exception as e:
    # This requires some extra dependencies and is optional
    print("Error saving graph:", e)


# Step 6: Stream the graph
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break


