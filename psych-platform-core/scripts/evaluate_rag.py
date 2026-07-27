import os
import sys
import warnings
import asyncio
from pathlib import Path
import pandas as pd
from datasets import Dataset

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.services.rag_service import RAGService

# Ragas — use old-style singleton metrics (they inherit from Metric, which
# ragas.evaluation.aevaluate checks via isinstance). The new collections metrics
# inherit from SimpleBaseMetric instead and fail that check in ragas 0.4.x.
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings

# Groq via OpenAI-compatible client (for RAGAS evaluator)
from openai import OpenAI as OpenAIClient

# Groq via LangChain (for answer generation)
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


def _build_groq_chat() -> ChatGroq:
    return ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0.7,
    )


async def _generate_with_groq(llm: ChatGroq, question: str, context: str, mood: str) -> str:
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


async def generate_evaluation_dataset(rag_service: RAGService, test_cases: list):
    """Runs test cases through RAG + Groq to produce contexts and answers."""
    llm = _build_groq_chat()
    records = []

    for case in test_cases:
        question = case["question"]
        mood = case["mood"]
        topic = case["topic"]
        ground_truth = case["ground_truth"]

        print(f"\n--- Processing: '{question[:60]}...' (Mood: {mood}) ---")

        try:
            context = await rag_service.retrieve_clinical_context(question, mood)
            contexts = [context] if context else [""]
            print(f"  [OK] Context retrieved ({len(context)} chars)")
        except Exception as e:
            print(f"  [ERROR] Context retrieval failed: {e}")
            contexts = [""]

        answer = ""
        try:
            answer = await _generate_with_groq(llm, question, contexts[0], mood)
            print(f"  [OK] Answer generated ({len(answer)} chars)")
        except Exception as e:
            print(f"  [ERROR] Generation failed: {e}")

        records.append({
            "question": question,
            "topic": topic,
            "mood": mood,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth,
        })

        await asyncio.sleep(2)

    return records


def _topic_summary(df_results: pd.DataFrame, output_dir: Path) -> Path:
    """Build and save per-topic metric summaries."""
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    if "topic" not in df_results.columns:
        raise ValueError("Cannot build topic summary: 'topic' column missing in evaluation results")

    rows = []
    for topic, group in df_results.groupby("topic", dropna=False):
        row = {
            "topic": topic if pd.notna(topic) else "unknown",
            "n": int(len(group)),
        }
        for metric in metric_cols:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                row[f"{metric}_avg"] = float(values.mean()) if not values.empty else None
                row[f"{metric}_min"] = float(values.min()) if not values.empty else None
                row[f"{metric}_max"] = float(values.max()) if not values.empty else None
            else:
                row[f"{metric}_avg"] = None
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(by="topic").reset_index(drop=True)
    summary_path = output_dir / "rag_evaluation_topic_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def _print_topic_summary(summary_df: pd.DataFrame):
    """Print concise per-topic metrics in console."""
    if summary_df.empty:
        print("No topic-level summary to print.")
        return

    print("\n=========================================")
    print("Per-Topic RAG Summary")
    print("=========================================")
    for _, row in summary_df.iterrows():
        print(
            f"- {row['topic']} (n={int(row['n'])}) | "
            f"faithfulness={row['faithfulness_avg']:.4f} "
            f"answer_relevancy={row['answer_relevancy_avg']:.4f} "
            f"context_precision={row['context_precision_avg']:.4f} "
            f"context_recall={row['context_recall_avg']:.4f}"
        )


def run_ragas_evaluation(records: list):
    """Runs RAGAS evaluation using Groq (OpenAI-compatible) as the evaluator LLM."""
    print("\n=========================================")
    print("Initializing Ragas Evaluation...")
    print("=========================================")

    df = pd.DataFrame(records)
    dataset = Dataset.from_pandas(df)

    # Build evaluator LLM via llm_factory (uses Groq's OpenAI-compatible endpoint).
    # gemma2-9b-it is used here because llama-3.3-70b-versatile fails instructor's
    # JSON schema validation despite generating valid JSON (Groq + instructor quirk).
    groq_client = OpenAIClient(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    evaluator_llm = llm_factory("llama-3.1-8b-instant", client=groq_client)
    # LangchainEmbeddingsWrapper exposes embed_query, which answer_relevancy requires
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        LCHuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"local_files_only": True},
        )
    )

    # Bind LLM and embeddings directly onto the singleton metric objects
    faithfulness.llm = evaluator_llm
    context_precision.llm = evaluator_llm
    context_recall.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    run_config = RunConfig(
        max_workers=1,
        max_retries=5,
        max_wait=30,
        timeout=120,
    )

    print("Running evaluation (this may take a few moments)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
    )

    print("\n=========================================")
    print("Ragas Evaluation Complete!")
    print("=========================================")
    print(results)

    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)
    df_results = results.to_pandas()

    # Ragas result column names can vary by version; align keys for a stable join.
    input_df = pd.DataFrame(records)
    if "question" in input_df.columns and "user_input" in df_results.columns:
        df_results = df_results.merge(
            input_df[["question", "topic", "mood"]],
            left_on="user_input",
            right_on="question",
            how="left",
        )
    elif "question" in df_results.columns:
        df_results = df_results.merge(
            input_df[["question", "topic", "mood"]],
            on="question",
            how="left",
        )
    output_path = output_dir / "rag_evaluation_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path.resolve()}")

    topic_summary_path = _topic_summary(df_results, output_dir)
    summary_df = pd.read_csv(topic_summary_path)
    _print_topic_summary(summary_df)
    print(f"Per-topic summary saved to: {topic_summary_path.resolve()}")

    return results


async def main():
    test_cases = [
        {
            "question": "I've been feeling extremely anxious lately, especially before social situations. My heart races and I just want to escape. What can I do?",
            "topic": "anxiety",
            "mood": "anxious",
            "ground_truth": "Cognitive Behavioral Therapy (CBT) recommends techniques such as cognitive restructuring to challenge negative thoughts about social situations, and gradual exposure therapy to habituate to the anxiety rather than escaping it. Deep breathing and progressive muscle relaxation can help manage physical symptoms like a racing heart.",
        },
        {
            "question": "I feel so down and useless. I've lost interest in hobbies I used to love and have zero energy to do anything.",
            "topic": "depression",
            "mood": "depressed",
            "ground_truth": "Treatment guidelines for adult depression suggest Behavioral Activation (BA) as a core CBT technique, which involves scheduling and engaging in small, positive activities even when energy is low to break the cycle of depression. It also suggests identifying and challenging automatic negative thoughts.",
        },
        {
            "question": "My partner and I keep arguing over small things. Every conversation turns into a fight and I feel disconnected.",
            "topic": "relationship",
            "mood": "lonely",
            "ground_truth": "In relationship conflict, therapists recommend Active Listening techniques where partners mirror and validate each other's feelings before responding. Emotional connection is rebuilt by expressing underlying needs (like connection or fear of abandonment) using 'I' statements instead of accusatory 'You' statements.",
        },
        {
            "question": "I can't stop worrying about my health. Every minor symptom makes me think I have a terminal disease, and I keep googling symptoms.",
            "topic": "anxiety",
            "mood": "anxious",
            "ground_truth": "Health anxiety (hypochondriasis) is treated using CBT by identifying and limiting safety-seeking behaviors like excessive googling and body checking. Cognitive restructuring helps evaluate the realistic probability of disease versus catastrophic misinterpretation of normal bodily sensations.",
        },
    ]

    os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY

    print("Initializing services...")
    rag_service = RAGService()

    records = await generate_evaluation_dataset(rag_service, test_cases)
    run_ragas_evaluation(records)


if __name__ == "__main__":
    asyncio.run(main())
