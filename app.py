import gradio as gr

from rag import question_answer


CUSTOM_CSS = """
body, .gradio-container {
    background:
        radial-gradient(circle at top, rgba(123, 63, 228, 0.18), transparent 32%),
        linear-gradient(180deg, #0b0b12 0%, #11111a 45%, #09090f 100%);
    color: #f5f7fb;
}

.melodex-shell {
    max-width: 920px;
    margin: 0 auto;
    padding: 28px 20px 24px;
}

.melodex-header {
    margin-bottom: 18px;
}

.melodex-title {
    margin: 0;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f8faff;
}

.melodex-subtitle {
    margin: 8px 0 0;
    color: #b8bed3;
    font-size: 1rem;
}

#melodex-chatbot {
    border: 1px solid rgba(173, 181, 255, 0.14);
    border-radius: 16px;
    background: rgba(17, 19, 30, 0.88);
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
}

#melodex-input textarea {
    background: rgba(20, 22, 34, 0.96) !important;
    color: #f5f7fb !important;
}

.gr-button-primary {
    background: linear-gradient(90deg, #7c4dff 0%, #e044a7 100%) !important;
    border: none !important;
}
"""


def respond(message, history):
    history = history or []
    answer = question_answer(message, history)
    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, updated_history, ""


def main():
    with gr.Blocks() as view:
        history_state = gr.State([])

        with gr.Column(elem_classes=["melodex-shell"]):
            gr.HTML(
                """
                <div class="melodex-header">
                    <h1 class="melodex-title">Melodex</h1>
                    <p class="melodex-subtitle">
                        Ask about artists, albums, members, and the sound behind the music.
                    </p>
                </div>
                """
            )

            chatbot = gr.Chatbot(
                value=[],
                height=540,
                elem_id="melodex-chatbot",
            )

            prompt = gr.Textbox(
                placeholder="Ask Melodex about an artist, album, or member...",
                lines=1,
                max_lines=4,
                container=False,
                elem_id="melodex-input",
            )

            gr.Examples(
                examples=[
                    ["Who is Hanni from NewJeans?"],
                    ["Who are the members of TXT?"],
                ],
                inputs=prompt,
            )

            prompt.submit(
                respond,
                inputs=[prompt, history_state],
                outputs=[chatbot, history_state, prompt],
            )

    view.launch(css=CUSTOM_CSS, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
