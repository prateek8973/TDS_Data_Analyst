import os, json, subprocess, sys
from typing import List, Dict, Any
import dill
from crewai.tools import BaseTool
from pydantic import Field

# Import only the tools you want available (no Firecrawl, no RAG)
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    ScrapeElementFromWebsiteTool,
    CSVSearchTool,
    DirectoryReadTool,
    DOCXSearchTool,
    FileReadTool,
    JSONSearchTool,
    PDFSearchTool,
    PGSearchTool,
    TXTSearchTool,
    VisionTool,
    XMLSearchTool,
    GithubSearchTool,              # ⬅️ include GitHub
    MDXSearchTool,
    YoutubeChannelSearchTool,
    YoutubeVideoSearchTool
)

# ---- Optional thin wrappers (kept for compatibility if referenced elsewhere) ----
class SearchTool(SerperDevTool):
    def __init__(self):
        super().__init__()

class ScrapeTool(ScrapeWebsiteTool):
    def __init__(self):
        super().__init__()


import base64

def persist_base64_files(base64_files: Dict[str, str], workdir: str) -> Dict[str, str]:
    """
    base64_files: {filename: base64_string}
    """
    saved = {}
    for fname, b64content in base64_files.items():
        path = os.path.join(workdir, os.path.basename(fname))
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64content))
        saved[fname] = path
    return saved


# ---- File persistence for uploads ----
def persist_uploads(files: List, workdir: str) -> Dict[str, str]:
    saved = {}
    for f in files:
        path = os.path.join(workdir, os.path.basename(f.filename))
        with open(path, "wb") as out:
            out.write(f.file.read())
        saved[f.filename] = path
    return saved

# ---- Python execution tool with sandboxing ----
class PythonExecTool(BaseTool):
    name: str = "python_exec"
    description: str = "Run reviewed Python code in isolated subprocess with timeout and resource limits."

    workdir: str
    saved_files: Dict[str, str] = Field(default_factory=dict)
    timeout_sec: int = 60

    def __post_init__(self):
        os.makedirs(self.workdir, exist_ok=True)

    def _run(self, code_list_json: str) -> str:
        try:
            code_list = json.loads(code_list_json)
        except Exception as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})

        results = []
        for item in code_list:
            step_id = item.get("step_id")
            filename = os.path.basename(item.get("filename", f"step_{step_id or 'x'}.py"))
            code = item.get("code", "")

            # Replace uploaded file references with actual paths
            for name, path in self.saved_files.items():
                if f'"{name}"' in code or f"'{name}'" in code:
                    code = code.replace(f'"{name}"', f'"{path}"').replace(f"'{name}'", f"'{path}'")

            fpath = os.path.join(self.workdir, filename)
            with open(fpath, "w", encoding="utf-8") as fp:
                fp.write(code)

            res = self._run_file(fpath)
            res.update({"step_id": step_id, "filename": filename})
            results.append(res)

        return json.dumps(results)

    def _run_file(self, path: str) -> Dict[str, Any]:
        state_file = os.path.join(self.workdir, "_state.pkl")
        wrapper = f"""
import dill, os
if os.path.exists(r"{state_file}"):
    with open(r"{state_file}", "rb") as f:
        globals().update(dill.load(f))

with open(r"{path}", "r", encoding="utf-8") as f:
    code = f.read()
exec(code, globals())

with open(r"{state_file}", "wb") as f:
    dill.dump(globals(), f)
"""
        temp_path = path + ".wrapped.py"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(wrapper)

        process = subprocess.Popen(
            [sys.executable, temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=self.timeout_sec)

        return {"returncode": process.returncode, "stdout": stdout, "stderr": stderr}

# ---- Build a single, comprehensive toolset (for all agents) ----
def build_all_tools(workdir: str, saved_files: Dict[str, str]):
    tools = []

    # File / local tools
    tools += [
        CSVSearchTool(),
        DOCXSearchTool(),
        JSONSearchTool(),
        PDFSearchTool(),
        PGSearchTool(),
        TXTSearchTool(),
        XMLSearchTool(),
        VisionTool(),
        FileReadTool(),
        DirectoryReadTool(),
        MDXSearchTool(),
    ]

    # Web scraping
    tools += [
        ScrapeWebsiteTool(),
        ScrapeElementFromWebsiteTool(),
    ]

    # Search (Serper)
    if os.getenv("SERPER_API_KEY"):
        tools.append(SerperDevTool())

    # GitHub (token/key optional; include if available)
    if os.getenv("GITHUB_API_KEY") or os.getenv("GITHUB_TOKEN"):
        try:
            tools.append(GithubSearchTool())
        except Exception:
            pass

    # YouTube (API key optional; include if available)
    if os.getenv("YOUTUBE_API_KEY"):
        tools += [YoutubeChannelSearchTool(), YoutubeVideoSearchTool()]

    # Python execution (stateful across steps within the same workdir)
    tools.append(PythonExecTool(workdir=workdir, saved_files=saved_files))

    return tools

# (Optional) keep get_optimal_tools for compatibility, but without Firecrawl/RAG
def get_optimal_tools(file_path=None, question=None):
    tools = []

    # File-based selection
    if file_path:
        ext = file_path.split('.')[-1].lower()
        if ext == 'csv':
            tools.append(CSVSearchTool())
        elif ext in ['xls', 'xlsx', 'docx']:
            tools.append(DOCXSearchTool())
        elif ext == 'json':
            tools.append(JSONSearchTool())
        elif ext == 'pdf':
            tools.append(PDFSearchTool())
        elif ext in ['db', 'sqlite', 'sql']:
            tools.append(PGSearchTool())
        elif ext == 'xml':
            tools.append(XMLSearchTool())
        elif ext == 'txt':
            tools.append(TXTSearchTool())
        elif ext in ['jpg', 'jpeg', 'png']:
            tools.append(VisionTool())
        else:
            tools.append(FileReadTool())

    # Question-based selection
    if question:
        q = question.lower()
        if "github" in q:
            try:
                tools.append(GithubSearchTool())
            except Exception:
                pass
        if "youtube" in q or "video" in q:
            if os.getenv("YOUTUBE_API_KEY"):
                tools += [YoutubeChannelSearchTool(), YoutubeVideoSearchTool()]
        if any(k in q for k in ["scrape", "web", "website"]):
            tools += [ScrapeWebsiteTool(), ScrapeElementFromWebsiteTool()]
        if any(k in q for k in ["search", "google"]):
            if os.getenv("SERPER_API_KEY"):
                tools.append(SerperDevTool())
        if any(k in q for k in ["md", "markdown", "documentation"]):
            tools.append(MDXSearchTool())

    # Always include PythonExecTool for execution
    tools.append(PythonExecTool(workdir="./workdir", saved_files={}))

    return tools
