import gradio as gr
from rag import question_answer
def main():

    # Simple gradio interface for now
    view = gr.ChatInterface(
        fn = question_answer,
        title = "Melodex 🎶",
    )

    view.launch()


if __name__ == "__main__":
    main()
