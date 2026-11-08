
from pathlib import Path
import sys
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(ROOT))
import mock_llm

SIMILARITY_THRESHOLD = 0.05


def load_policies():
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(POLICY_DIR.glob("*.txt"))
    }


def chunk_text(text, chunk_words=40, overlap=10):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(overlap=10):
    chunks = []

    for source, text in load_policies().items():
        for chunk in chunk_text(text, 40, overlap):
            chunks.append({"source": source, "text": chunk})

    return chunks


class PolicyRAG:
    def __init__(self, overlap=10):
        self.chunks = build_chunks(overlap)

        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [item["text"] for item in self.chunks]
        )

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked = scores.argsort()[::-1][:k]

        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(scores[i]),
                "text": self.chunks[i]["text"]
            }
            for i in ranked
        ]

    def answer(self, question):
        results = self.retrieve(question, k=3)

        if not results or results[0]["score"] < SIMILARITY_THRESHOLD:
            return {
                "answer": (
                    "I could not find that information in the "
                    "policy documents."
                ),
                "sources": results,
                "fallback": True
            }

        context = "\n".join(
            f"[{item['source']}] {item['text']}"
            for item in results
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "POLICY ASSISTANT: Answer only from the supplied "
                    "policy context. Do not invent facts. If the answer "
                    "is unavailable, say so clearly."
                )
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"
            }
        ]

        response = mock_llm.chat(messages)
        answer_text = response.choices[0].message.content

        fallback = (
            "could not find that information"
            in answer_text.lower()
            or "not available" in answer_text.lower()
        )

        return {
            "answer": answer_text,
            "sources": results,
            "fallback": fallback
        }


def evaluate(rag, evaluation_file, output_name):
    df = pd.read_csv(evaluation_file)
    results = []

    for _, row in df.iterrows():
        question = str(row["question"])
        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = str(row["expected_fact"]).strip()

        retrieved = rag.retrieve(question, k=3)
        answer_result = rag.answer(question)

        retrieved_docs = [item["source"] for item in retrieved]
        retrieved_text = " ".join(
            item["text"] for item in retrieved
        ).lower()

        is_unanswerable = expected_doc.upper() == "NONE"

        if is_unanswerable:
            doc_found = None
            fact_found = None
            fallback_correct = bool(answer_result["fallback"])
        else:
            doc_found = expected_doc in retrieved_docs
            fact_found = expected_fact.lower() in retrieved_text
            fallback_correct = None

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "retrieved_docs": "|".join(retrieved_docs),
            "top_score": retrieved[0]["score"] if retrieved else 0.0,
            "expected_doc_found": doc_found,
            "expected_fact_found": fact_found,
            "fallback": bool(answer_result["fallback"]),
            "fallback_correct": fallback_correct,
            "answer": answer_result["answer"]
        })

    result_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_DIR / output_name, index=False)

    answerable = result_df[result_df["expected_doc"] != "NONE"]
    unanswerable = result_df[result_df["expected_doc"] == "NONE"]

    doc_score = answerable["expected_doc_found"].fillna(False).astype(bool)
    fact_score = answerable["expected_fact_found"].fillna(False).astype(bool)
    fallback_score = (
        unanswerable["fallback_correct"].fillna(False).astype(bool)
    )

    print(f"\nEVALUATION: {output_name}")
    print("Total questions:", len(result_df))
    print("Answerable questions:", len(answerable))
    print("Unanswerable questions:", len(unanswerable))
    print("Expected document found:", int(doc_score.sum()), "/", len(answerable))
    print("Expected fact found:", int(fact_score.sum()), "/", len(answerable))
    print("Correct fallback responses:", int(fallback_score.sum()), "/", len(unanswerable))

    failures = answerable.loc[~(doc_score & fact_score)]

    if not failures.empty:
        print("\nANSWERABLE QUESTION FAILURES:")
        for _, row in failures.iterrows():
            print("-", row["question"])
            print("  Expected:", row["expected_doc"])
            print("  Retrieved:", row["retrieved_docs"])

    fallback_failures = unanswerable.loc[~fallback_score]

    if not fallback_failures.empty:
        print("\nFALLBACK FAILURES:")
        for _, row in fallback_failures.iterrows():
            print("-", row["question"])
            print("  Answer:", row["answer"])

    return result_df


def main():
    evaluation_file = RAW_DIR / "policy_eval.csv"

    print("POLICY DOCUMENT CHUNKING")
    rag = PolicyRAG(overlap=10)

    print("Total chunks:", len(rag.chunks))

    for source in sorted(set(item["source"] for item in rag.chunks)):
        count = sum(item["source"] == source for item in rag.chunks)
        print(f"{source}: {count}")

    print("\nRETRIEVAL TEST")
    query = (
        "What delivery time makes a completed order "
        "eligible for a full refund?"
    )

    for item in rag.retrieve(query, k=3):
        print(item["source"], round(item["score"], 4), item["text"])

    print("\nOVERLAP 10 EVALUATION")
    evaluate(
        rag,
        evaluation_file,
        "policy_rag_evaluation.csv"
    )

    print("\nZERO-OVERLAP COMPARISON")
    zero_rag = PolicyRAG(overlap=0)

    print("Zero-overlap chunks:", len(zero_rag.chunks))

    zero_results = evaluate(
        zero_rag,
        evaluation_file,
        "policy_rag_zero_overlap.csv"
    )

    answerable = zero_results[
        zero_results["expected_doc"] != "NONE"
    ]

    zero_doc = answerable["expected_doc_found"].fillna(False).astype(bool)
    zero_fact = answerable["expected_fact_found"].fillna(False).astype(bool)
    failed = answerable.loc[~(zero_doc & zero_fact)]

    print("\nZERO-OVERLAP ANALYSIS")

    if not failed.empty:
        for _, row in failed.iterrows():
            print("-", row["question"])
            print("  Expected policy:", row["expected_doc"])
            print("  Retrieved:", row["retrieved_docs"])

        print(
            "Explanation: Without overlap, information near chunk "
            "boundaries can be separated across chunks, reducing "
            "the chance of retrieving the complete relevant passage."
        )
    else:
        print(
            "No answerable question failed both top-3 document and "
            "fact checks. No boundary failure was demonstrated."
        )

    print("\nPOLICY RAG OUTPUTS SAVED")
    print("- outputs/policy_rag_evaluation.csv")
    print("- outputs/policy_rag_zero_overlap.csv")


if __name__ == "__main__":
    main()
