"""
Side-by-side RAGAS evaluation: Pinecone vs Qdrant.

Runs the same 4 test cases through both backends and prints a comparison table.
Results saved to logs/backend_comparison.csv.

Usage:
    python scripts/compare_backends.py
"""
import os
import sys
import warnings
import asyncio
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datasets import Dataset

from app.core.config import settings
from app.services.rag_service import RAGService

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from openai import OpenAI as OpenAIClient
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

TEST_CASES = [
    {
        "question": "I've been feeling extremely anxious lately, especially before social situations. My heart races and I just want to escape. What can I do?",
        "mood": "anxious",
        "ground_truth": "Cognitive Behavioral Therapy (CBT) recommends techniques such as cognitive restructuring to challenge negative thoughts about social situations, and gradual exposure therapy to habituate to the anxiety rather than escaping it. Deep breathing and progressive muscle relaxation can help manage physical symptoms like a racing heart.",
    },
    {
        "question": "I feel so down and useless. I've lost interest in hobbies I used to love and have zero energy to do anything.",
        "mood": "depressed",
        "ground_truth": "Treatment guidelines for adult depression suggest Behavioral Activation (BA) as a core CBT technique, which involves scheduling and engaging in small, positive activities even when energy is low to break the cycle of depression. It also suggests identifying and challenging automatic negative thoughts.",
    },
    {
        "question": "My partner and I keep arguing over small things. Every conversation turns into a fight and I feel disconnected.",
        "mood": "lonely",
        "ground_truth": "In relationship conflict, therapists recommend Active Listening techniques where partners mirror and validate each other's feelings before responding. Emotional connection is rebuilt by expressing underlying needs (like connection or fear of abandonment) using 'I' statements instead of accusatory 'You' statements.",
    },
    {
        "question": "I can't stop worrying about my health. Every minor symptom makes me think I have a terminal disease, and I keep googling symptoms.",
        "mood": "anxious",
        "ground_truth": "Health anxiety (hypochondriasis) is treated using CBT by identifying and limiting safety-seeking behaviors like excessive googling and body checking. Cognitive restructuring helps evaluate the realistic probability of disease versus catastrophic misinterpretation of normal bodily sensations.",
    },
]

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _patch_ragas_metrics():
    groq_client = OpenAIClient(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    evaluator_llm = llm_factory("llama-3.1-8b-instant", client=groq_client)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        LCHuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"local_files_only": True},
        )
    )
    faithfulness.llm = evaluator_llm
    context_precision.llm = evaluator_llm
    context_recall.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings


async def _generate_answer(llm: ChatGroq, question: str, context: str, mood: str) -> str:
    system_prompt = (
        "You are a compassionate, professional AI psychologist. "
        "Use the clinical context provided (if any) to give a grounded, empathetic response. "
        "Respond in 3-5 sentences. Do not diagnose — offer coping strategies and validation."
    )
    context_block = f"\n\nClinical context:\n{context}" if context else ""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Patient mood: {mood}{context_block}\n\nPatient message: {question}"),
    ]
    response = await llm.ainvoke(messages)
    return response.content


async def _run_for_backend(backend_name: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  Backend: {backend_name.upper()}")
    print(f"{'='*55}")

    rag_service = RAGService(backend_override=backend_name)
    llm = ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0.7,
    )

    records = []
    for case in TEST_CASES:
        question, mood, ground_truth = case["question"], case["mood"], case["ground_truth"]
        print(f"\n  Q: '{question[:55]}...' (mood={mood})")

        try:
            context = await rag_service.retrieve_clinical_context(question, mood)
            contexts = [context] if context else [""]
            print(f"  Context: {len(context)} chars")
        except Exception as e:
            print(f"  Context ERROR: {e}")
            contexts = [""]

        try:
            answer = await _generate_answer(llm, question, contexts[0], mood)
            print(f"  Answer: {len(answer)} chars")
        except Exception as e:
            print(f"  Answer ERROR: {e}")
            answer = ""

        records.append({
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth,
        })
        await asyncio.sleep(2)

    dataset = Dataset.from_pandas(pd.DataFrame(records))
    run_config = RunConfig(max_workers=1, max_retries=5, max_wait=30, timeout=120)
    results = evaluate(dataset=dataset, metrics=METRICS, run_config=run_config)

    df = results.to_pandas()
    means = {m: round(float(df[m].mean()), 4) for m in METRIC_NAMES if m in df.columns}
    print(f"\n  Scores: {means}")
    return means


def _print_comparison(pinecone: dict, qdrant: dict):
    print("\n")
    print("=" * 66)
    print("  RAGAS COMPARISON: PINECONE  vs  QDRANT  (mean over 4 cases)")
    print("=" * 66)
    print(f"  {'Metric':<25} {'Pinecone':>10} {'Qdrant':>10} {'Delta':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    rows = []
    for m in METRIC_NAMES:
        p = pinecone.get(m, float("nan"))
        q = qdrant.get(m, float("nan"))
        d = q - p if (p == p and q == q) else float("nan")
        d_str = f"{d:+.4f}" if d == d else "   N/A"
        print(f"  {m:<25} {p:>10.4f} {q:>10.4f} {d_str:>10}")
        rows.append({"metric": m, "pinecone": p, "qdrant": q, "delta_qdrant_minus_pinecone": d})
    print("=" * 66)

    output_dir = Path(__file__).resolve().parent.parent / "logs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "backend_comparison.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


async def main():
    print("Patching RAGAS metrics with Groq evaluator LLM...")
    _patch_ragas_metrics()

    pinecone_scores = await _run_for_backend("pinecone")
    qdrant_scores = await _run_for_backend("qdrant")

    _print_comparison(pinecone_scores, qdrant_scores)


if __name__ == "__main__":
    asyncio.run(main())
