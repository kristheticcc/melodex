# Imports
import gradio as gr
from knowledge import build_knowledge_dict
def main():
    print("Hello from melodex!")
    k = build_knowledge_dict()
    print(k["twice"])


if __name__ == "__main__":
    main()
