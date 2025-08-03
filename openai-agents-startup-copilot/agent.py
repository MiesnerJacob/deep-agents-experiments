import os
import asyncio
import base64
import tempfile
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    ImageGenerationTool,
)   


load_dotenv("../.env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class StartupQuestionOutput(BaseModel):
    is_startup_question: bool
    reasoning: str

guardrail_agent = Agent( 
    name="Startup Question Guardrail",
    instructions="Check if the user is defining a problem they want to solve.",
    model="gpt-4o-mini",
    output_type=StartupQuestionOutput
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

class IdeaOutput(BaseModel):
    use_cases: list[str] = Field(min_length=3, max_length=3, description="A list of three use cases for the startup.")
    mission: str = Field(description="A mission statement for the startup.")
    audience: str = Field(description="The target audience for the startup.")

idea_agent = Agent(  
    name="Idea Generator",
    instructions="You are an idea generator. You generate ideas for startups. You expand the idea into use cases, mission, and audience. You are given a problem and you need to generate a startup idea that would solve the problem.",
    model="gpt-4o",
    output_type=IdeaOutput,
    input_guardrails=[startup_question_guardrail]
)

class NamingOutput(BaseModel):
    names: list[str] = Field(min_length=3, max_length=3, description="A list of three alternative names for the startup.")

naming_agent = Agent(  
    name="Naming Agent",
    model="gpt-4o",
    instructions="You are a naming agent. You generate names for startups. You create three alternative names for the startup.",
    output_type=NamingOutput,
)

class NameSelectionOutput(BaseModel):
    name: str = Field(description="The name of the startup.")

name_selector = Agent(
    name="NameSelector",
    instructions="""
    Pick the best name from the list of candidates based on clarity, brand appeal, and alignment with the startup concept.
    Respond ONLY with the name.
    """,
    model="gpt-4o-mini",
    output_type=NameSelectionOutput,
)

logo_agent = Agent(
    name="Logo Agent",
    instructions="""
    Generate a logo for the startup using the ImageGenerationTool.
    
    IMPORTANT: You must use the ImageGenerationTool to create an actual image.
    Do not return placeholder text or markdown links.
    """,
    model="gpt-4o-mini", 
    tools=[
        ImageGenerationTool(
            tool_config={"type": "image_generation", "quality": "low"},
        )
    ]
)

class PitchOutput(BaseModel):
    pitch: str = Field(description="A pitch for the startup. It should be a short pitch that is easy to understand and remember.")

pitch_agent = Agent(  
    name="Pitch Agent",
    instructions="You are a pitch agent. You generate pitches for startups. You create a pitch for the startup. Do not use any fake testimonials or fake numbers.",
    model="gpt-4o",
    output_type=PitchOutput,
)

class PitchRefinerOutput(BaseModel):
    pitch_ready: bool = Field(description="Whether the pitch is ready to be used.")
    pitch_critique: str = Field(description="A critique of the pitch. If the pitch is not ready, you should critique it and ask for improvements.")

pitch_refiner_agent = Agent(  
    name="Pitch Refiner",
    instructions="You are a pitch refiner. You refine the pitch for the startup. You check if the pitch is ready to be used. If not, you critique the pitch and ask for improvements.",
    model="gpt-4o-mini",
    output_type=PitchRefinerOutput,
)

class LaunchChecklistOutput(BaseModel):
    checklist: list[str] = Field(min_length=10, max_length=10, description="A checklist for launching the startup.")

launch_checklist_agent = Agent(  
    name="Launch Checklist",
    instructions="You are a launch checklist agent. You create a launch checklist for the startup.",
    model="gpt-4o-mini",
    output_type=LaunchChecklistOutput,
)

class SummaryOutput(BaseModel):
    summary: str = Field(description="A summary of the startup.")

summary_agent = Agent(  
    name="Summary",
    instructions="You are a summary agent. You summarize the startup.",
    model="gpt-4o",
    output_type=SummaryOutput,
)

async def main():
    current_question = input("Enter your problem you want to solve: ")
    
    while True:
        try:
            print("💡 Developing core idea...")
            idea_result = await Runner.run(idea_agent, current_question)
            
            print("🎨 Creating branding assets in parallel...")
            branding_tasks = [
                Runner.run(naming_agent, f"Create names for: {idea_result.final_output}"),
                Runner.run(logo_agent, f"Create logo for: {idea_result.final_output}")
            ]
            naming_result, logo_result = await asyncio.gather(*branding_tasks)

            print("🏷️ Selecting name...")
            name_result = await Runner.run(name_selector, f"Select the best name from the list: {', '.join(naming_result.final_output.names)}")
            
            print("💼 Creating and refining pitch...")
            pitch_context = f"""
            Name: {name_result.final_output.name}
            Mission: {idea_result.final_output.mission}
            Audience: {idea_result.final_output.audience}
            Use Cases: {', '.join(idea_result.final_output.use_cases)}
            """
            
            print("✍️ Creating initial pitch...")
            pitch_result = await Runner.run(pitch_agent, f"Create a pitch for: {pitch_context}")
            current_pitch = pitch_result.final_output.pitch
            
            for attempt in range(3):
                print(f"🔎 Critiquing pitch (attempt {attempt + 1}/3)...")
                pitch_refiner_result = await Runner.run(
                    pitch_refiner_agent, 
                    f"Review this pitch: {current_pitch}"
                )
                
                if pitch_refiner_result.final_output.pitch_ready:
                    print("✅ Pitch is ready!")
                    break
                    
                if attempt <= 2:
                    print("🔄 Improving pitch based on feedback...")
                    critique = pitch_refiner_result.final_output.pitch_critique
                    
                    improvement_context = f"""
                    Previous pitch: {current_pitch}
                    
                    Critique and improvement suggestions: {critique}
                    
                    Original context: {pitch_context}
                    
                    Please create an improved pitch addressing the critique.
                    """
                    
                    pitch_result = await Runner.run(pitch_agent, improvement_context)
                    current_pitch = pitch_result.final_output.pitch
                else:
                    print("⚠️ Max attempts reached, using current pitch")
            
            print("📝 Creating launch checklist...")
            checklist_result = await Runner.run(
                launch_checklist_agent, 
                f"Create launch checklist for: {pitch_context}"
            )
            
            print("📊 Creating final summary...")
            summary_context = f"""
            Startup Summary:
            Mission: {idea_result.final_output.mission}
            Target Audience: {idea_result.final_output.audience}
            Use Cases: {', '.join(idea_result.final_output.use_cases)}
            Company Name: {name_result.final_output.name}
            Final Pitch: {current_pitch}
            Launch Checklist: {', '.join(checklist_result.final_output.checklist)}
            """
            
            summary_result = await Runner.run(summary_agent, summary_context)
            
            print("="*50 + "\n")
            print(f"{name_result.final_output.name}")
            print(f"LOGO: {logo_result.final_output}")
            print("="*50 + "\n")
            print(f"Mission: {idea_result.final_output.mission}", "\n")
            print(f"Audience: {idea_result.final_output.audience}", "\n")
            print("Use Cases:")
            for use_case in idea_result.final_output.use_cases:
                print(f"• {use_case}")
            print("\n" + "Sales Pitch:" + "\n" + current_pitch, "\n")
            print("Launch Checklist:")
            for index, item in enumerate(checklist_result.final_output.checklist):
                print(f"{index + 1}. {item}")
            print("\n" + f"Summary: {summary_result.final_output.summary}")
            
            return

        except InputGuardrailTripwireTriggered:
            print("Please ask a startup-related question")
            current_question = input("New problem: ")


if __name__ == "__main__":
    asyncio.run(main())