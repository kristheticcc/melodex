# Imports
import gradio as gr
from phase1_chat import chat
def main():

    # Simple gradio interface for now
    view = gr.ChatInterface(
        fn = chat,
        title = "Melodox",
    )

    view.launch()


if __name__ == "__main__":
    main()
