import os
import re
import time
import traceback
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils.data_profiler import get_data_profile

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert data engineer and Python programmer.
Your job is to analyse a dataset profile and write clean, correct pandas code to fix data quality issues.

Rules:
1. You ONLY write Python code using pandas. No other imports are needed.
2. The dataframe is already loaded and available as the variable `df`.
3. Your code must end with `df` being the cleaned dataframe (do NOT reassign to a new variable name).
4. Wrap your code in a single ```python ... ``` block.
5. Before the code block, write a brief numbered Cleaning Plan (plain text).
6. Do NOT use df.inplace=True patterns that might conflict with chaining. Prefer `df = df.method()`.
7. Be conservative: only drop rows/columns when clearly justified.
"""


def _extract_code(text: str) -> str:
    """Pull the Python code block out of the LLM response."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return everything if no fence found
    return text.strip()


def _extract_plan(text: str) -> str:
    """Extract the cleaning plan section above the code block."""
    parts = text.split("```python")
    if parts:
        return parts[0].strip()
    return ""


def _safe_exec(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute agent-generated code in a sandboxed namespace.
    Only pandas and the dataframe are exposed — no builtins like open(), os, etc.
    """
    safe_globals = {"__builtins__": {}, "pd": pd}
    local_ns = {"df": df.copy()}
    exec(code, safe_globals, local_ns)  # noqa: S102
    result = local_ns.get("df")
    if not isinstance(result, pd.DataFrame):
        raise ValueError("Executed code did not produce a valid DataFrame named `df`.")
    return result


def run_cleaning_agent(
    df: pd.DataFrame,
    user_goal: str,
    max_retries: int = 3,
    model_name: str = "gemini-2.0-flash",
) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Runs the agentic cleaning loop.

    Returns:
        cleaned_df  : The cleaned DataFrame (original if all retries fail)
        final_code  : The last code string that succeeded (or last attempted)
        logs        : List of log strings for the UI
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    logs: list[str] = []
    final_code = ""

    if not api_key:
        logs.append("❌ **Error:** GOOGLE_API_KEY is not set. Please add your key in the sidebar.")
        return df, final_code, logs

    # Build the LLM
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1,
        convert_system_message_to_human=True,
    )

    profile = get_data_profile(df)

    # Initial prompt
    user_prompt = f"""Here is the data profile for the uploaded dataset:

{profile}

User's cleaning goal: {user_goal}

Please provide:
1. A numbered Cleaning Plan explaining each step.
2. A single ```python``` code block that cleans the dataframe `df` according to the plan.
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    attempt = 0
    last_error = ""

    while attempt < max_retries:
        attempt += 1
        response_text = ""
        logs.append(f"---\n**🔄 Attempt {attempt} / {max_retries}** — Calling {model_name}...")

        try:
            response = llm.invoke(messages)
            response_text = response.content

            # Extract plan & code
            plan = _extract_plan(response_text)
            code = _extract_code(response_text)
            final_code = code

            if plan:
                logs.append(f"**📋 Cleaning Plan (Attempt {attempt}):**\n\n{plan}")

            logs.append(f"**💻 Generated Code (Attempt {attempt}):**\n```python\n{code}\n```")

            # Try executing
            cleaned_df = _safe_exec(code, df)
            logs.append(f"✅ **Success on attempt {attempt}!** DataFrame cleaned: "
                        f"{df.shape} → {cleaned_df.shape}")
            return cleaned_df, code, logs

        except Exception as e:
            last_error = traceback.format_exc()
            is_rate_limit = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)

            if is_rate_limit:
                logs.append(f"⏳ **Rate limit hit (Attempt {attempt}).** Waiting 15 seconds before retrying...")
                time.sleep(15)
            else:
                logs.append(f"❌ **Execution failed (Attempt {attempt}):**\n```\n{last_error}\n```")

            if attempt < max_retries:
                if not is_rate_limit:
                    logs.append("🔁 **Feeding error back to agent for self-correction...**")
                    messages.append(HumanMessage(content=response_text))
                    messages.append(HumanMessage(
                        content=f"The code above raised the following error:\n\n```\n{last_error}\n```\n\n"
                                f"Please fix the code and try again. Only return the corrected ```python``` block."
                    ))

    logs.append(f"❌ **Agent failed after {max_retries} attempts.** Returning original dataframe.")
    return df, final_code, logs
