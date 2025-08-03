import os
from dotenv import load_dotenv
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel
import asyncio

load_dotenv("../.env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define Guardrail Output
class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

# Define Guardrail Agent
guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

# Define Guardrail Function
async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

# Define Specialist Agents
math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)


# Define Main Function
async def main():
    # Example 1: History question
    try:
        history_question = "who was the first president of the united states?"
        print("History Question:", history_question)
        result = await Runner.run(triage_agent, history_question)
        print("History Question Result:", result.final_output, "\n")
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

    # Example 2: Math question
    try:
        math_question = "What is 2 + 2?"
        print("Math Question:", math_question)
        result = await Runner.run(triage_agent, math_question)
        print("Math Question Result:", result.final_output, "\n")
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)

    # Example 3: General/philosophical question
    try:
        general_question = "What is the meaning of life?"
        print("General Question:", general_question)
        result = await Runner.run(triage_agent, general_question)
        print("General Question Result:", result.final_output, "\n")
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e)



asyncio.run(main())