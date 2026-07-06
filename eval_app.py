from collections import defaultdict

import gradio as gr
import pandas as pd

from eval import evaluate_all_answers, evaluate_all_retrieval


def score_color(score: float, good_threshold: float, warning_threshold: float) -> str:
    if score >= good_threshold:
        return "#1f9d55"
    if score >= warning_threshold:
        return "#f59e0b"
    return "#dc2626"


def build_metric_card(title: str, score: float, good_threshold: float, warning_threshold: float) -> str:
    color = score_color(score, good_threshold, warning_threshold)
    return (
        "<div style='flex:1; min-width:160px; padding:16px; border-radius:12px; "
        "background:#f8fafc; border:1px solid #e2e8f0;'>"
        f"<div style='font-size:14px; color:#475569; margin-bottom:8px;'>{title}</div>"
        f"<div style='font-size:32px; font-weight:700; color:{color};'>{score:.3f}</div>"
        "</div>"
    )


def build_retrieval_summary(avg_mrr: float, avg_ndcg: float) -> str:
    cards = [
        build_metric_card("Average MRR", avg_mrr, 0.9, 0.75),
        build_metric_card("Average nDCG", avg_ndcg, 0.9, 0.75),
    ]
    return (
        "<div style='display:flex; gap:16px; flex-wrap:wrap;'>"
        + "".join(cards)
        + "</div>"
    )


def build_answer_summary(avg_accuracy: float, avg_completeness: float, avg_relevance: float) -> str:
    cards = [
        build_metric_card("Average Accuracy", avg_accuracy, 4.5, 4.0),
        build_metric_card("Average Completeness", avg_completeness, 4.5, 4.0),
        build_metric_card("Average Relevance", avg_relevance, 4.5, 4.0),
    ]
    return (
        "<div style='display:flex; gap:16px; flex-wrap:wrap;'>"
        + "".join(cards)
        + "</div>"
    )


def empty_category_frame(value_column: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["category", value_column])


def run_retrieval_evaluation(progress=gr.Progress()) -> tuple[str, pd.DataFrame]:
    category_scores: dict[str, list[float]] = defaultdict(list)
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []

    for test, result, current_progress in evaluate_all_retrieval():
        progress(current_progress, desc=f"Evaluating retrieval: {test.category}")
        mrr_scores.append(result.mrr)
        ndcg_scores.append(result.ndcg)
        category_scores[test.category].append(result.mrr)

    if not mrr_scores:
        return build_retrieval_summary(0.0, 0.0), empty_category_frame("avg_mrr")

    category_df = pd.DataFrame(
        [
            {"category": category, "avg_mrr": sum(scores) / len(scores)}
            for category, scores in sorted(category_scores.items())
        ]
    )

    summary_html = build_retrieval_summary(
        sum(mrr_scores) / len(mrr_scores),
        sum(ndcg_scores) / len(ndcg_scores),
    )
    return summary_html, category_df


def run_answer_evaluation(progress=gr.Progress()) -> tuple[str, pd.DataFrame]:
    category_scores: dict[str, list[float]] = defaultdict(list)
    accuracy_scores: list[float] = []
    completeness_scores: list[float] = []
    relevance_scores: list[float] = []

    for test, result, current_progress in evaluate_all_answers():
        progress(current_progress, desc=f"Evaluating answers: {test.category}")
        accuracy_scores.append(result.accuracy)
        completeness_scores.append(result.completeness)
        relevance_scores.append(result.relevance)
        category_scores[test.category].append(result.accuracy)

    if not accuracy_scores:
        return build_answer_summary(0.0, 0.0, 0.0), empty_category_frame("avg_accuracy")

    category_df = pd.DataFrame(
        [
            {"category": category, "avg_accuracy": sum(scores) / len(scores)}
            for category, scores in sorted(category_scores.items())
        ]
    )

    summary_html = build_answer_summary(
        sum(accuracy_scores) / len(accuracy_scores),
        sum(completeness_scores) / len(completeness_scores),
        sum(relevance_scores) / len(relevance_scores),
    )
    return summary_html, category_df


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Melodex Evaluation Dashboard") as demo:
        gr.Markdown("# Melodex Evaluation Dashboard")

        with gr.Group():
            gr.Markdown("## Retrieval Evaluation")
            retrieval_button = gr.Button("Run Retrieval Evaluation", variant="primary")
            retrieval_summary = gr.HTML(value=build_retrieval_summary(0.0, 0.0))
            retrieval_plot = gr.BarPlot(
                value=empty_category_frame("avg_mrr"),
                x="category",
                y="avg_mrr",
                title="Average MRR by Question Category",
                tooltip=["category", "avg_mrr"],
                height=350,
            )

        with gr.Group():
            gr.Markdown("## Answer Evaluation")
            answer_button = gr.Button("Run Answer Evaluation", variant="primary")
            answer_summary = gr.HTML(value=build_answer_summary(0.0, 0.0, 0.0))
            answer_plot = gr.BarPlot(
                value=empty_category_frame("avg_accuracy"),
                x="category",
                y="avg_accuracy",
                title="Average Accuracy by Question Category",
                tooltip=["category", "avg_accuracy"],
                height=350,
            )

        retrieval_button.click(
            fn=run_retrieval_evaluation,
            outputs=[retrieval_summary, retrieval_plot],
        )
        answer_button.click(
            fn=run_answer_evaluation,
            outputs=[answer_summary, answer_plot],
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
