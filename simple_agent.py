import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-4.1-nano-2025-04-14"

client = OpenAI(api_key=openai_key)

response = client.chat.completions.create(
    model=llm_model,
    messages=[
        {"role": "system", "content": "you are a helpful assistant."},
        {"role": "user", "content": "Who is Pancho Villa?"}
    ],
)

# print(response.choices[0].message.content)

# Create a simple agent
class SimpleAgent:
    def __init__(self, system="you are a helpful assistant."):
        self.system = system
        self.messages = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute() # Call the function execute and then add it to our assistant
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def execute(self):
        response = client.chat.completions.create(
            model=llm_model,
            temperature=0.0,
            messages=self.messages,
        )
        return response.choices[0].message.content

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

planet_mass:
e.g. planet_mass: Earth
returns the mass of a planet in the solar system

Example session:

Question: What is the combined mass of Earth and Mars?
Thought: I should find the mass of each planet using planet_mass.
Action: planet_mass: Earth
PAUSE

You will be called again with this:

Observation: Earth has a mass of 5.972 × 10^24 kg

You then output:

Answer: Earth has a mass of 5.972 × 10^24 kg

Next, call the agent again with:

Action: planet_mass: Mars
PAUSE

Observation: Mars has a mass of 0.64171 × 10^24 kg

You then output:

Answer: Mars has a mass of 0.64171 × 10^24 kg

Finally, calculate the combined mass.

Action: calculate: 5.972 + 0.64171
PAUSE

Observation: The combined mass is 6.61371 × 10^24 kg

Answer: The combined mass of Earth and Mars is 6.61371 × 10^24 kg
""".strip() # remove leading and trailing whitespace

# Implement the functions actions
def calculate(what):
    return eval(what)

def planet_mass(planet):
    masses = {
        "Mercury": 0.33011,
        "Venus": 4.8675,
        "Earth": 5.972,
        "Mars": 0.64171,
        "Jupiter": 1898.19,
        "Saturn": 568.34,
        "Uranus": 86.813,
        "Neptune": 102.413,
    }
    return f"{planet} has a mass of {masses[planet]} × 10^24 kg"

known_actions = {"calculate": calculate, "planet_mass": planet_mass}


# Create an instance of the SimpleAgent
agent = SimpleAgent(system=prompt)