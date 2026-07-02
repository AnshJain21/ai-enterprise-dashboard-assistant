"""
Data Q&A over a pandas DataFrame.

Pattern used: instead of pasting raw rows into the prompt (slow, expensive,
and prone to the model 'inventing' numbers), we ask the model to write a
single pandas expression against a known DataFrame `df`, execute that
expression ourselves, and then ask the model to explain the result in
plain language. This keeps every number grounded in your real data.
"""
import re
import pandas as pd
from utils.llm_client import chat

FORBIDDEN_PATTERN = re.compile(
    r"\b(import|open|exec|eval|os\.|sys\.|subprocess|__|to_csv|to_excel|to_sql|read_)\b"
)


def _describe_df(df: pd.DataFrame) -> str:
    dtypes = "\n".join(f"- {col} ({dtype})" for col, dtype in df.dtypes.items())
    sample = df.head(3).to_markdown(index=False)
    return f"Columns:\n{dtypes}\n\nSample rows:\n{sample}\n\nTotal rows: {len(df)}"


def _generate_pandas_expr(question: str, df: pd.DataFrame) -> str:
    schema = _describe_df(df)
    system = (
        "You write a single pandas expression (one line, no assignment, "
        "no imports, no comments) that answers the user's question about "
        "a DataFrame called `df`. Reply with ONLY the expression, nothing else. "
        "Never use file I/O, exec/eval, or any function starting with 'to_' or 'read_'."
    )
    prompt = f"DataFrame info:\n{schema}\n\nQuestion: {question}\n\nPandas expression:"
    expr = chat(prompt, system_instruction=system).strip()
    # strip markdown code fences if the model added them anyway
    expr = re.sub(r"^```(?:python)?|```$", "", expr, flags=re.MULTILINE).strip()
    return expr


def answer_question(question: str, df: pd.DataFrame) -> dict:
    """
    Returns {"expression": str, "result": Any, "answer": str, "error": str|None}
    """
    expr = _generate_pandas_expr(question, df)

    if FORBIDDEN_PATTERN.search(expr):
        return {
            "expression": expr,
            "result": None,
            "answer": "I generated an unsafe expression and blocked it. Try rephrasing the question.",
            "error": "blocked_unsafe_expression",
        }

    try:
        # Restricted eval: only `df` and pandas builtins are reachable.
        result = eval(expr, {"__builtins__": {}}, {"df": df, "pd": pd})
    except Exception as e:  # noqa: BLE001
        return {
            "expression": expr,
            "result": None,
            "answer": f"I couldn't compute that ({e}). Try rephrasing the question.",
            "error": str(e),
        }

    result_str = result.to_string() if hasattr(result, "to_string") else str(result)
    # keep the explanation grounded in the actual computed result
    explain_prompt = (
        f"Question: {question}\n"
        f"Computed result:\n{result_str}\n\n"
        "Explain this result to a business user in 1-3 concise sentences. "
        "Do not invent numbers beyond what's shown above."
    )
    answer = chat(explain_prompt)

    return {"expression": expr, "result": result, "answer": answer, "error": None}
