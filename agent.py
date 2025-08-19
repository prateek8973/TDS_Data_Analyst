import os

# ---- Environment setup ----
# Hugging Face Spaces writable dirs (both ephemeral here)
TMP_DIR = "/tmp/.mem0"
PERSIST_DIR = "/tmp/.embedchain"

# Override embedchain + mem0 default paths
os.environ["EMBEDCHAIN_CONFIG_DIR"] = PERSIST_DIR
os.environ["MEM0_DIR"] = TMP_DIR
os.environ["CREWAI_STORAGE_PATH"] = "/tmp/crewai_storage"
os.environ["XDG_DATA_HOME"] = os.environ["CREWAI_STORAGE_PATH"]
os.environ["HOME"] = "/tmp"
# Make sure dirs exist
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(os.environ["CREWAI_STORAGE_PATH"], exist_ok=True)
os.makedirs(os.path.join("/tmp", ".local", "share"), exist_ok=True)
from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool
from typing import Dict
from tools import build_all_tools
import random
from dotenv import load_dotenv

load_dotenv()

#  Load API keys
GOOGLE_API_KEYS = [v for k, v in os.environ.items() if k.startswith("GOOGLE_API") and v]
if not GOOGLE_API_KEYS:
    raise ValueError("No GOOGLE_API_KEY1/2/3 found in environment")

def get_api_key() -> str:
    return random.choice(GOOGLE_API_KEYS)

# Gemini LLM wrapper with fallback
from litellm import completion
import re

class GeminiLLM:
    def __init__(self, model="gemini/gemini-1.5-flash", api_key=None, temperature=0):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def __call__(self, prompt: str, image_base64: str = None):
        """
        If image_base64 is provided, sends multimodal request.
        Otherwise, sends plain text request.
        """

        if image_base64:
            # multimodal request: text + base64 image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        else:
            # text-only request
            messages = [{"role": "user", "content": prompt}]

        try:
            resp = completion(
                model=self.model,
                api_key=self.api_key,
                messages=messages,
                temperature=self.temperature
            )
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            # fallback dummy JSON if API fails
            return {
                "status": "error",
                "message": "LLM call failed",
                "error": str(e),
                "dummy_output": "No valid response due to API rate limit or failure."
            }

def stop_if_answered(output: str) -> bool:
    return "FINAL ANSWER" in output

def build_crew(workdir: str, saved_files: Dict[str, str]):
    all_tools = build_all_tools(workdir=workdir, saved_files=saved_files)
    chat_llm = GeminiLLM(api_key=get_api_key())

    # Initialize the Code Interpreter tool
    code_interpreter = CodeInterpreterTool()

    planner = Agent(
        role="Planner",
        goal="Create optimal sub-queries and tool plan",
        backstory="Expert project planner for data & analysis tasks",
        allow_delegation=False,
        verbose=True,
        llm=chat_llm,
        memory=False,
        system_prompt="You're a meticulous planning agent...",
        tools=all_tools
    )

    developer = Agent(
        role="Developer",
        goal="Write correct, sandbox-safe Python code",
        backstory="Senior Python engineer",
        allow_delegation=False,
        verbose=True,
        llm=chat_llm,
        system_prompt="You are a senior Python developer focused on accuracy.",
        tools=all_tools
    )

    reviewer = Agent(
        role="Reviewer",
        goal="Review and correct code before execution",
        backstory="Principal engineer",
        allow_delegation=False,
        verbose=True,
        llm=chat_llm,
        system_prompt="You are a strict code reviewer focused on correctness.",
        tools=all_tools
    )

    executor = Agent(
        role="Executor",
        goal="Run reviewed code and return outputs",
        backstory="Execution agent with secure interpreter",
        allow_delegation=False,
        verbose=True,
        llm=chat_llm,
        tools=[code_interpreter]  # Exclusive execution tool
    )

    analyst = Agent(
        role="Analyst",
        goal="Synthesize outputs, ensure correctness, and return the final formatted answer as per user request",
        backstory="Data analyst who also ensures formatting correctness",
        allow_delegation=False,
        verbose=True,
        llm=chat_llm,
        system_prompt=(
            "You are the final analyst. "
            "Your responsibilities include:\n"
            "1. Synthesizing outputs from previous steps.\n"
            "2. Ensuring the final answer matches the format the user requested.\n"
            "3. Wrapping the final output inside JSON if needed.\n"
            "4. If something fails, return: {\"answer\": \"Dummy response due to error or quota limit.\"}"
        ),
        tools=all_tools
    )

    file_context = (
    f"The user uploaded files: " +
    ", ".join([f"{name} (at {path})" for name, path in saved_files.items()])
    if saved_files else
    "No uploaded files beyond questions.txt."
)

    tasks = [
        Task(description=f"{file_context}\n\nCreate a detailed plan for answering '{{question}}'.", agent=planner, expected_output="Step-by-step plan", stop_condition=stop_if_answered),
        Task(description="Write Python code snippets where needed.", agent=developer, expected_output="Python code implementing the solution", stop_condition=stop_if_answered),
        Task(description="Review developer's code.", agent=reviewer, expected_output="Reviewed and corrected code", stop_condition=stop_if_answered),
        Task(description="Execute code and/or fetch data.", agent=executor, expected_output="Execution results", stop_condition=stop_if_answered),
        Task(description="Synthesize results into final formatted answer.", agent=analyst, expected_output="Final formatted answer matching user request", stop_condition=stop_if_answered),
    ]

    crew = Crew(
        agents=[planner, developer, reviewer, executor, analyst],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        max_iterations=1
    )

    return crew
