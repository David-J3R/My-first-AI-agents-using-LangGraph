import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END

# Load environment variables from .env file
load_dotenv

openai_key = os.getenv("OPENAI_API_KEY")

llm = "gpt-4.1-nano-2025-04-14"

model = ChatOpenAI(api_key=openai_key, model=llm)

# STEP 1: Build a Basic Chatbot
from langgraph.graph.message import add_messages
# add_messages specifies how this should be updated when the state changes

# Define the state of the agent
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    messages: Annotated[list, add_messages] # Annotated allows to add metadata

# STEP 2: build the first node for the graph
def bot(state: State):
    # The bot function is the first node in the graph. It takes the state as input
    # and returns a response based on the messages in the state.
    print(state["messages"])
    return {"messages": [model.invoke(state["messages"])]}

graph_builder = StateGraph(State)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("bot", bot)

# Step 3: add an entry point and end to the graph
graph_builder.set_entry_point("bot")  # it could be any node

graph_builder.set_finish_point("bot")

# Step 5: Compile the graph
graph = graph_builder.compile()

# res = graph.invoke({"messages": ["Hello, how are you?"]})
# print(res)

# Step 6: Stream the graph
while True:
    # Get user input
    user_input = input("You: ")

    # Check if the user wants to exit
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Exiting the chat.")
        break

    # Invoke the graph with the user input
    # The .stream() method returns an iterable of events
    # each "event" is a dictionary with the node name as the key; represent a step in the graph
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


