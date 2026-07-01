# Imports
import glob
from pathlib import Path
# Function for building knowledge dictionary
def build_knowledge_dict():
    knowledge = {}

    # For reading all files in the knowledge_base/artists directory
    filenames = glob.glob("knowledge_base/artists/*")

    for filename in filenames:
        name = Path(filename).stem
        cleaned_name = name.replace("_", " ")
        with open(filename, "r", encoding = "utf-8") as f:
            knowledge[cleaned_name] = f.read()


    # For reading all files in the knowledge_base/albums directory
    filenames = glob.glob("knowledge_base/albums/*")

    for filename in filenames:
        name = Path(filename).stem
        cleaned_name = name.replace("_", " ")
        with open(filename, "r", encoding = "utf-8") as f:
            knowledge[cleaned_name] = f.read()

    return knowledge




