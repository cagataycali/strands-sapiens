# Example: drive every Sapiens2 tool from a Strands agent via natural language.

from strands import Agent
from strands_sapiens import TOOLS

agent = Agent(
    tools=TOOLS,
    system_prompt=(
        "You are a human-centric vision assistant. "
        "Use the sapiens_* tools to answer image questions. "
        "Prefer the 0.4b model unless the user asks otherwise."
    ),
)

print(agent("What sapiens2 checkpoints are installed on this machine?"))

print(
    agent(
        "Segment /tmp/sapiens2_test/input/human.jpg and save the output to "
        "/tmp/sapiens2_test/output. Save the raw prediction too."
    )
)
