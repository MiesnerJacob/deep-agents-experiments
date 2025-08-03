import os
import asyncio
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)


load_dotenv("../.env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class StartupQuestionOutput(BaseModel):
    is_startup_question: bool
    reasoning: str

guardrail_agent = Agent( 
    name="Startup Question Guardrail",
    instructions="Check if the user is asking you to do their startup question.",
    output_type=StartupQuestionOutput,
)


@input_guardrail
async def startup_question_guardrail( 
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=not result.final_output.is_startup_question,  # Block startup questions
    )


agent = Agent(  
    name="Startup Copilot",
    instructions="You are a startup copilot. You help customers with their startup questions.",
    input_guardrails=[startup_question_guardrail],
)

async def main():
    current_question = input("Enter your startup question: ")
    
    while True:
        try:
            result = await Runner.run(agent, current_question)
            print(f"Success: {result.final_output}")
            return result.final_output

        except InputGuardrailTripwireTriggered:
            print("Please ask a startup-related question")
            current_question = input("New question: ")


if __name__ == "__main__":
    asyncio.run(main())