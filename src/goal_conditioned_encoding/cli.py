import csv
import os
import yaml
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from tabulate import tabulate
from tqdm import tqdm


from .llm import chat_model, configure_llm_cache


class Encoding(BaseModel):
    """Single observation interpretation."""

    content: str = Field(
        ...,
        description="2-4 sentence interpretation from this perspective. Specific, no hedging.",
    )


class Summary(BaseModel):
    """Summary."""

    content: str = Field(..., description="Summary. Structured, specific, no preamble.")


class Answer(BaseModel):
    answer: str = Field(..., description="Direct answer to the question")


class Judgment(BaseModel):
    relevance: int = Field(
        ..., ge=1, le=5, description="Does it answer what was asked?"
    )
    completeness: int = Field(
        ..., ge=1, le=5, description="Are important details included?"
    )
    accuracy: int = Field(
        ..., ge=1, le=5, description="Is it correct given the context?"
    )
    clarity: int = Field(..., ge=1, le=5, description="Is it clearly expressed?")
    notes: str | None = None

    @property
    def score(self) -> float:
        return (self.relevance + self.completeness + self.accuracy + self.clarity) / 4


PERSPECTIVES: Final[dict[str, str]] = {
    "relationship": """Interpret this observation from a RELATIONSHIP perspective.
Focus on: trust, commitment, reciprocity, partnership health.
- What does this signal about commitment or trust?
- What expectations does it create?
- How does it affect the relationship trajectory?
Produce 2-4 specific sentences. Ignore financial/competitive details.""",
    "risk": """Interpret this observation from a RISK perspective.
Focus on: liability, precedent, compliance, exposure, mitigation needs.
- What precedents does this establish?
- What exposure or liability does it create?
- What documentation or action is needed?
Produce 2-4 specific sentences. Ignore relationship/competitive noise.""",
    "financial": """Interpret this observation from a FINANCIAL perspective.
Focus on: margins, costs, revenue, budget variance, quantitative terms.
- What are the exact dollar amounts and percentages?
- How does this affect projections?
- What is the P&L impact?
Produce 2-4 specific sentences. Preserve ALL numbers precisely. Ignore relationship noise.""",
    "competitive": """Interpret this observation from a COMPETITIVE perspective.
Focus on: market signals, alternatives mentioned, leverage shifts, positioning.
- What does this reveal about the client's options?
- What does it suggest about our market position?
- How does leverage shift?
Produce 2-4 specific sentences. Ignore internal relationship dynamics.""",
    "unified": """Record this observation factually. No interpretation.
Capture: what happened, who was involved, when, what was said, what numbers.
Do not infer meaning. Do not assess implications. Just log facts.
Produce 2-4 factual sentences.""",
    "unified-interpretive": """Interpret this observation from ALL perspectives simultaneously.

Analyze implications for:
- RELATIONSHIP: trust/commitment signals, expectations created
- FINANCIAL: dollar amounts, percentages, margin/cost impact
- RISK: precedents, exposure, mitigation needs
- COMPETITIVE: market signals, leverage shifts, positioning

Produce 4-6 sentences covering all relevant dimensions. Be specific.""",
}


def memory_file(perspective: str, mem_dir: Path) -> Path:
    return mem_dir / f"{perspective}.txt"


def clear_memory(mem_dir: Path) -> None:
    for p in PERSPECTIVES.keys():
        memory_file(p, mem_dir).unlink(missing_ok=True)


def read_memory(perspective: str, mem_dir: Path) -> str:
    mem = memory_file(perspective, mem_dir)
    if mem.exists():
        with open(mem, "r") as f:
            return f.read().strip()
    return ""


def write_memory(perspective: str, mem_dir: Path, content: str) -> None:
    mem = memory_file(perspective, mem_dir)
    mem_dir.mkdir(parents=True, exist_ok=True)
    with open(mem, "w") as f:
        f.write(content + "\n")


def encode(
    perspective: str,
    observation: str,
    model: str,
    token_limit: int,
) -> str:
    messages = [
        SystemMessage(
            content=f"{PERSPECTIVES[perspective]}\nResponse limit: {token_limit} tokens"
        ),
        HumanMessage(content=observation),
    ]
    res = chat_model(model).with_structured_output(Encoding).invoke(messages)
    if not isinstance(res, Encoding):
        raise ValueError("failed to parse Encoding")
    return res.content


def integrate(
    perspective: str,
    mem_dir: Path,
    encoding: str,
    model: str,
    token_limit: int,
) -> None:
    memory = read_memory(perspective, mem_dir)
    messages = [
        SystemMessage(
            content=f"Summarise the input.\nResponse limit: {token_limit} tokens"
        ),
        HumanMessage(content=f"{memory}\n\n{encoding}"),
    ]
    summary = chat_model(model).with_structured_output(Summary).invoke(messages)
    if not isinstance(summary, Summary):
        raise ValueError("failed to parse Summary")
    write_memory(perspective, mem_dir, summary.content)


def retrieve(
    query: str,
    memory: str,
    model: str,
) -> str:
    messages = [
        SystemMessage(
            content="Answer the question using only information from the memory."
        ),
        HumanMessage(
            content=f"""
MEMORY:
{memory}

QUESTION:
{query}
"""
        ),
    ]
    answer = chat_model(model).with_structured_output(Answer).invoke(messages)
    if not isinstance(answer, Answer):
        raise ValueError("failed to parse Answer")
    return answer.answer


def judge(
    query: str,
    answer: str,
    ctx: str,
    model: str,
) -> Judgment:
    messages = [
        SystemMessage(content="You are evaluating an answer's quality."),
        HumanMessage(
            content=f"""
QUESTION:
{query}

ANSWER:
{answer}

GROUND TRUTH CONTEXT:
{ctx}

Rate 1-5 on:
- relevance: Does it answer what was asked?
- completeness: Are important details included?
- accuracy: Is it correct given the context?
- clarity: Is it clearly expressed?
"""
        ),
    ]
    judgment = chat_model(model).with_structured_output(Judgment).invoke(messages)
    if not isinstance(judgment, Judgment):
        raise ValueError("failed to parse Judgment")
    return judgment


def parse_args():
    parser = ArgumentParser(description="Goal Conditoined Encoding CLI")
    parser.add_argument("--scenario", help="Path to scenario YAML file")
    parser.add_argument(
        "--model",
        help="LLM model to use for encoding and retrieval (default: gpt-5-nano)",
        default="gpt-5-nano",
    )
    parser.add_argument(
        "--judge-model",
        help="LLM model to use for judging (default: gpt-5-mini)",
        default="gpt-5-mini",
    )
    parser.add_argument(
        "--buffer-dir",
        help="Directory for buffer files (default: .buffers)",
        default=".buffers",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing buffers instead of clearing them",
    )
    parser.add_argument("--output", help="Path to an output directory")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM caching")
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of times to run the scenario (default: 1)",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=500,
        help="Number of tokens a single perspective may use (instruction only)",
    )

    args = parser.parse_args()
    assert args.scenario, "scenario required"
    return args


def print_results(query_res: list[tuple[str, str, str, Judgment]]) -> None:
    scores = defaultdict(dict)
    for query, perspective, answer, judgment in query_res:
        scores[query][perspective] = judgment.score

    perspectives = sorted({p for _, p, _, _ in query_res})

    rows = []
    for query in scores:
        row = [query[:50] + "..." if len(query) > 50 else query]
        row += [f"{scores[query].get(p, 0):.2f}" for p in perspectives]
        rows.append(row)

    avgs = []
    for p in perspectives:
        p_scores = [scores[q][p] for q in scores if p in scores[q]]
        avgs.append(f"{sum(p_scores) / len(p_scores):.2f}" if p_scores else "-")
    rows.append(["AVERAGE"] + avgs)

    print(tabulate(rows, headers=["Query"] + perspectives, tablefmt="grid"))


def save(
    model: str,
    judge_model: str,
    scenario: str,
    path: Path,
    query_res: list[tuple[str, str, str, Judgment]],
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "judge_model",
                "scenario",
                "query",
                "perspective",
                "answer",
                "judgment_relevance",
                "judgment_completeness",
                "judgment_accuracy",
                "judgment_clarity",
            ]
        )

        for q, p, a, j in query_res:
            writer.writerow(
                [
                    model,
                    judge_model,
                    scenario,
                    q,
                    p,
                    a,
                    j.relevance,
                    j.completeness,
                    j.accuracy,
                    j.clarity,
                ]
            )


def main():
    args = parse_args()
    load_dotenv()
    configure_llm_cache(not args.no_cache)

    print(
        tabulate(
            [["Goal Conditioned Encoder", ""]]
            + [[v, getattr(args, v)] for v in vars(args)],
            headers="firstrow",
            tablefmt="grid",
        )
    )

    for i in range(args.n):
        data = []

        if args.n > 1:
            print(f"\n=== Run {i + 1} / {args.n} ===")

        mem_dir = Path(args.buffer_dir)
        if not args.append:
            clear_memory(mem_dir)

        with open(args.scenario, "r") as f:
            scenario = yaml.safe_load(f)

        for obs in tqdm(
            scenario.get("observations", []), desc="Processing Observations", unit="obs"
        ):
            for p in tqdm(
                PERSPECTIVES.keys(), desc="Encoding", unit="perspective", leave=False
            ):
                token_limit = (
                    args.token_limit * 4
                    if p == "unified-interpretive"
                    else args.token_limit
                )
                encoding = encode(p, obs, args.model, token_limit)
                integrate(
                    p,
                    mem_dir,
                    encoding,
                    args.model,
                    token_limit,
                )

        ctx = "\n".join(scenario.get("observations", []))

        for q in tqdm(scenario.get("queries"), desc="Processing Queries", unit="query"):
            query, goal = q["text"], q["aligned_goal"]
            for p in tqdm(
                PERSPECTIVES.keys(),
                desc="Perspectives",
                unit="perspective",
                leave=False,
            ):
                memory = read_memory(p, mem_dir)
                res = retrieve(query, memory, args.model)
                judgment = judge(query, res, ctx, args.judge_model)
                data.append((query, p, res, judgment))

            # arbiter
            memory = "\n".join(
                read_memory(p, mem_dir)
                for p in ("relationship", "risk", "financial", "competitive")
            )
            res = retrieve(query, memory, args.model)
            judgement = judge(query, res, ctx, args.judge_model)
            data.append((query, "arbiter", res, judgement))

        print_results(data)

        if dir_path := args.output:
            ts = datetime.now().isoformat().replace(":", "_")
            save(
                model=args.model,
                judge_model=args.judge_model,
                scenario=args.scenario,
                path=Path(dir_path) / f"{ts}.csv",
                query_res=data,
            )
