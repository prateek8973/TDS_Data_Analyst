from crewai import Agent, Task, Crew, Process
from typing import Dict
from tools import build_all_tools  # ⬅️ new: one place to assemble every tool
import os
import random
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

# Collect all GOOGLE_API_KEY* env vars
GOOGLE_API_KEYS = [
    v for k, v in os.environ.items() if k.startswith("GOOGLE_API_KEY") and v
]
if not GOOGLE_API_KEYS:
    raise ValueError("No GOOGLE_API_KEY1/2/3 found in environment")

def get_api_key() -> str:
    """Return a random API key for load balancing."""
    return random.choice(GOOGLE_API_KEYS)

# Wrapper function to pass to CrewAI as LLM
class GeminiLLM:
    def __init__(self, model="gemini/gemini-2.5-flash", temperature=0, api_key=None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or get_api_key()

    def __call__(self, prompt: str) -> str:
        resp = completion(
            model=self.model,
            api_key=self.api_key,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return resp.choices[0].message.content

def build_crew(workdir: str, saved_files: Dict[str, str]):
    # ⬅️ Build one comprehensive toolset (includes GitHub, Serper, YouTube, file tools, scraping, PythonExec)
    all_tools = build_all_tools(workdir=workdir, saved_files=saved_files)

    # Each crew gets a fresh LLM with a rotated key
    chat_llm = GeminiLLM(api_key=get_api_key())

    planner = Agent(
        role="Planner",
        goal="Create optimal sub-queries and tool plan",
        backstory="Expert project planner for data & analysis tasks",
        allow_delegation=False,
        verbose=True,
        llm=chat_llm,
        memory=False,
        system_prompt="""You are a meticulous planning agent.

1) Identify exactly what the user needs.
2) Use uploaded files when relevant; if the query also requires web/search/scraping/YouTube/GitHub, include them.
3) Do NOT impose a fixed priority order. Combine multiple sources when needed (files + search + scraping + YouTube + GitHub).
4) For each step: specify the tool, data to collect, validation checks, and calculations.
5) Never fabricate data. Prefer focused scraping (selectors) over full-page dumps.
""",
        tools=all_tools  # ⬅️ give full toolset
    )

    developer = Agent(
        role="Developer",
        goal="Write correct, sandbox-safe Python code for each sub-task",
        backstory="Senior Python engineer",
        allow_delegation=False,
        verbose=True,
        llm=chat_llm,
        system_prompt="You are a senior Python developer focused on accuracy.",
        tools=all_tools  # ⬅️ full toolset available
    )

    reviewer = Agent(
        role="Reviewer",
        goal="Review and correct code before execution",
        backstory="Principal engineer",
        allow_delegation=False,
        verbose=True,
        llm=chat_llm,
        system_prompt="You are a strict code reviewer focused on correctness.",
        tools=all_tools  # ⬅️ full toolset available
    )

    executor = Agent(
        role="Executor",
        goal="Run reviewed code in sandbox and return outputs",
        backstory="Execution agent",
        allow_delegation=False,
        verbose=True,
        llm=chat_llm,
        tools=all_tools  # ⬅️ can execute Python and also call search/scrape/GitHub/YouTube if the plan includes them
    )

    analyst = Agent(
        role="Analyst",
        goal="Synthesize outputs and produce final answer",
        backstory="Data analyst",
        allow_delegation=False,
        verbose=True,
        llm=chat_llm,
        system_prompt="You synthesize results into accurate final answers.",
        tools=all_tools  # ⬅️ full toolset available (for ad-hoc lookups if needed)
    )

    reporter = Agent(
        role="Reporter",
        goal="Ensure output matches requested format",
        backstory="Format validator",
        allow_delegation=False,
        verbose=True,
        llm=chat_llm,
        system_prompt="You validate and format the final answer.",
        tools=all_tools  # ⬅️ full toolset available
    )

    if saved_files:
        file_context = (
            "The user has uploaded the following files for analysis: "
            f"{', '.join(saved_files.keys())}. Use these wherever relevant."
        )
    else:
        file_context = (
            "The user has not uploaded any data files beyond questions.txt. "
            "You may use web/search/scraping/YouTube/GitHub tools as needed."
        )

    t1 = Task(
        description=f"""
{file_context}

Create a detailed, step-by-step plan to answer the user's question.
The user's question is: '{{{{question}}}}'
""",
        agent=planner,
        expected_output="Step-by-step plan with filenames, tools, and validation steps."
    )

    t2 = Task(
        description="Using the plan, write Python code snippets where needed.",
        agent=developer,
        expected_output="JSON list of code objects."
    )

    t3 = Task(
        description="Review developer's code and correct if necessary.",
        agent=reviewer,
        expected_output="'APPROVED' or corrected JSON."
    )

    t4 = Task(
        description="Execute the code list and/or fetch data via the planned tools.",
        agent=executor,
        expected_output="JSON list of execution results or retrieved data."
    )

    t5 = Task(
        description="Synthesize results into the final answer.",
        agent=analyst,
        expected_output="Final answer only."
    )

    t6 = Task(
        description="Validate final output formatting.",
        agent=reporter,
        expected_output="Corrected, format-compliant output."
    )

    crew = Crew(
        agents=[planner, developer, reviewer, executor, analyst, reporter],
        tasks=[t1, t2, t3, t4, t5, t6],
        process=Process.sequential,
        verbose=True
    )
    return crew
