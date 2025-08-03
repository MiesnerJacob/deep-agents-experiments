import os
from agents import Agent, Runner, WebSearchTool, RunConfig, set_default_openai_client, HostedMCPTool
from typing import List, Dict, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
from prompts import RESEARCH_INSTRUCTION_AGENT_PROMPT, CLARIFYING_AGENT_PROMPT


load_dotenv("../.env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=600.0)
set_default_openai_client(client)


# ─────────────────────────────────────────────────────────────
# Structured outputs (needed only for Clarifying agent)
# ─────────────────────────────────────────────────────────────
class Clarifications(BaseModel):
    questions: List[str]

# ─────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────
research_agent = Agent(
    name="Research Agent",
    model="o4-mini-deep-research-2025-06-26",
    instructions="Perform deep empirical research based on the user's instructions.",
    tools=[WebSearchTool(),
    ]
)

instruction_agent = Agent(
    name="Research Instruction Agent",
    model="gpt-4o-mini",
    instructions=RESEARCH_INSTRUCTION_AGENT_PROMPT,
    handoffs=[research_agent],
)

clarifying_agent = Agent(
    name="Clarifying Questions Agent",
    model="gpt-4o-mini",
    instructions=CLARIFYING_AGENT_PROMPT,
    output_type=Clarifications
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "Decide whether clarifications are required.\n"
        "• If yes → call transfer_to_clarifying_questions_agent\n"
        "• If no  → call transfer_to_research_instruction_agent\n"
        "Return exactly ONE function-call."
    ),
    handoffs=[clarifying_agent, instruction_agent],
)


# ─────────────────────────────────────────────────────────────
#  Research functions
# ─────────────────────────────────────────────────────────────
async def basic_research(query: str) -> str:
    """Run research with simple agent handoffs."""
    # Use simple run instead of streaming - much cleaner
    result = await Runner.run(
        triage_agent,
        query,
        run_config=RunConfig(tracing_disabled=True),
    )
    
    if not isinstance(result.final_output, Clarifications):
        return result.final_output
    else:
        print("\n" + "="*50)
        print("🤔 CLARIFICATION NEEDED:")
        print("="*50)
        print("The agent needs more information to provide better research:")
        print()
        
        replies = []
        for i, q in enumerate(result.final_output.questions, 1):
            print(f"{i}. {q}")
            user_answer = input("   Your answer: ").strip()
            if not user_answer:
                user_answer = "No specific preference."
            replies.append(f"**{q}**\n{user_answer}")
        
        print("\n📝 Continuing research with your answers...")
        
        # Continue with clarified query
        clarified_query = f"{query}\n\nAdditional context:\n" + "\n\n".join(replies)
        final_result = await Runner.run(
            instruction_agent,  # Skip triage, go directly to research
            clarified_query,
            run_config=RunConfig(tracing_disabled=True),
        )
        return final_result.final_output


if __name__ == "__main__":
    try:
        user_query = input("Enter your research query: ")
        if not user_query.strip():
            print("❌ Empty query provided. Please enter a research question.")
            exit(1)
            
        print(f"🔍 Starting deep research...")
        result = asyncio.run(basic_research(user_query))
        
        print("\n" + "="*60)
        print("📊 RESEARCH RESULTS:")
        print("="*60)
        print(result)
        
    except KeyboardInterrupt:
        print("\n❌ Research interrupted by user.")
    except Exception as e:
        print(f"❌ Error during research: {e}")
        import traceback
        traceback.print_exc()