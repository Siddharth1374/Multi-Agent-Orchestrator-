"""
LangGraph pipeline for the Multi-Agent Orchestrator UI.

Sequential execution:

    START
      ↓
orchestrator
      ↓
researcher
      ↓
analyst
      ↓
writer
      ↓
     END

Each agent reads the shared state and adds its own output.

Environment variables:

    MODEL
        Groq model name.
        Default: openai/gpt-oss-120b

    GROQ_API_KEY
        My Groq API key.

    TAVILY_API_KEY
        Optional. Enables real web search for the researcher.

    DEMO=1
        Forces demo mode without making API calls.
"""

import os
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# Load environment variables


try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass



# Configuration


# This is a Groq model.
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")

# Groq is the provider for this project.
PROVIDER = "groq"



# Agent information


AGENTS = [
    {
        "id": "orchestrator",
        "name": "Orchestrator",
        "role": "Coordinator",
        "description": "Plans and coordinates the overall task execution",
        "key": "plan",
    },
    {
        "id": "researcher",
        "name": "Researcher",
        "role": "Data Collection",
        "description": "Gathers and validates information from multiple sources",
        "key": "research",
    },
    {
        "id": "analyst",
        "name": "Analyst",
        "role": "Data Processing",
        "description": "Analyzes data and extracts meaningful insights",
        "key": "analysis",
    },
    {
        "id": "writer",
        "name": "Writer",
        "role": "Content Generation",
        "description": "Turns the findings into a polished final report",
        "key": "report",
    },
]



# Shared state


class State(TypedDict, total=False):
    task: str
    plan: str
    research: str
    analysis: str
    report: str



# Demo mode


def is_demo() -> bool:
    """
    Return True when demo mode is enabled or no Groq API key is available.
    """

    if os.getenv("DEMO", "").lower() in {"1", "true", "yes"}:
        return True

    return not bool(os.getenv("GROQ_API_KEY"))



# LLM


_llm = None


def _get_llm():
    """
    Create and cache the Groq LLM.
    """

    global _llm

    if _llm is None:
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to your .env file."
            )

        _llm = ChatGroq(
            model=MODEL,
            temperature=0,
            max_retries=6,
        )

    return _llm



# Convert AI response to text


def _text(message) -> str:
    """
    Convert an AIMessage content object into a plain string.
    """

    content = message.content

    if isinstance(content, str):
        return content

    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )



# Ask LLM


def ask(system: str, user: str, demo_text: str) -> str:
    """
    Call the Groq LLM.

    If demo mode is enabled, return predefined text instead.
    """

    if is_demo():
        time.sleep(1.2)
        return demo_text

    response = _get_llm().invoke(
        [
            ("system", system),
            ("user", user),
        ]
    )

    return _text(response)



# Optional web search


def web_search(query: str) -> str:
    """
    Optional real web search using Tavily.

    If TAVILY_API_KEY is not available, return an empty result.
    """

    if is_demo() or not os.getenv("TAVILY_API_KEY"):
        return ""

    try:
        from langchain_tavily import TavilySearch

        result = TavilySearch(max_results=5).invoke(
            {"query": query}
        )

        items = (
            result.get("results", [])
            if isinstance(result, dict)
            else []
        )

        return "\n\n".join(
            f"- {item.get('title')}: "
            f"{item.get('content')} "
            f"({item.get('url')})"
            for item in items
        )

    except Exception as exc:
        return f"(web search failed: {exc})"



# Agent 1: Orchestrator


def orchestrator(state: State) -> State:

    task = state["task"]

    plan = ask(
        system=(
            "You are the Orchestrator of a team with three agents: "
            "Researcher, Analyst and Writer. "
            "Create a short numbered plan with a maximum of 5 steps. "
            "Explain what each agent should do for the user's task. "
            "Be concrete and brief."
        ),
        user=task,
        demo_text=(
            f"**Plan for:** {task}\n\n"
            "1. Researcher: collect key facts, definitions and recent developments.\n"
            "2. Analyst: compare the findings and identify important insights, risks and trade-offs.\n"
            "3. Writer: produce a concise and well-structured final report."
        ),
    )

    return {
        "plan": plan
    }



# Agent 2: Researcher


def researcher(state: State) -> State:

    task = state["task"]

    sources = web_search(task)

    research = ask(
        system=(
            "You are the Researcher. "
            "Following the plan, gather and validate the key facts needed "
            "for the task. Return organized bullet points. "
            "If web results are provided, use them and mention sources. "
            "Clearly flag anything uncertain."
        ),
        user=(
            f"Task:\n{task}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Web results:\n{sources or '(none)'}"
        ),
        demo_text=(
            "## Key Findings\n\n"
            "- The topic has several well-documented core concepts and common use cases.\n"
            "- Adoption is growing, with active open-source communities and tooling.\n"
            "- Main challenges include complexity, cost control and evaluation.\n\n"
            "*(Demo mode: add a Groq API key for real results.)*"
        ),
    )

    return {
        "research": research
    }



# Agent 3: Analyst


def analyst(state: State) -> State:

    analysis = ask(
        system=(
            "You are the Analyst. "
            "Using the research, extract the most important insights. "
            "Discuss strengths, weaknesses, risks and trade-offs. "
            "Be specific and avoid repeating the research word-for-word."
        ),
        user=(
            f"Task:\n{state['task']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Research:\n{state['research']}"
        ),
        demo_text=(
            "## Insights\n\n"
            "- **Strength:** clear structure makes complex workflows easier to understand.\n"
            "- **Risk:** multiple agents can increase latency and token cost.\n"
            "- **Recommendation:** start simple and add specialized agents only where they provide value."
        ),
    )

    return {
        "analysis": analysis
    }



# Agent 4: Writer


def writer(state: State) -> State:

    report = ask(
        system=(
            "You are the Writer. "
            "Produce the final deliverable for the user's task in clear Markdown. "
            "Use short headings and bullets where useful. "
            "Use only the research and analysis provided. "
            "Keep the final answer concise and actionable."
        ),
        user=(
            f"Task:\n{state['task']}\n\n"
            f"Research:\n{state['research']}\n\n"
            f"Analysis:\n{state['analysis']}"
        ),
        demo_text=(
            "# Final Report\n\n"
            "## Summary\n\n"
            "Multi-agent systems divide a complex task into specialized roles "
            "that pass state between one another.\n\n"
            "## Recommendations\n\n"
            "- Start with the simplest workable design.\n"
            "- Add specialized agents where they clearly improve quality.\n"
            "- Monitor latency, cost and output quality."
        ),
    )

    return {
        "report": report
    }



# Build LangGraph


def build_graph():

    builder = StateGraph(State)

    builder.add_node(
        "orchestrator",
        orchestrator,
    )

    builder.add_node(
        "researcher",
        researcher,
    )

    builder.add_node(
        "analyst",
        analyst,
    )

    builder.add_node(
        "writer",
        writer,
    )

    # START → Orchestrator
    builder.add_edge(
        START,
        "orchestrator",
    )

    # Orchestrator → Researcher
    builder.add_edge(
        "orchestrator",
        "researcher",
    )

    # Researcher → Analyst
    builder.add_edge(
        "researcher",
        "analyst",
    )

    # Analyst → Writer
    builder.add_edge(
        "analyst",
        "writer",
    )

    # Writer → END
    builder.add_edge(
        "writer",
        END,
    )

    return builder.compile()


# Create the compiled graph
graph = build_graph()