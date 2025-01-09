import datasets
import jinja2
from pathlib import Path
import os

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()

    return content

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    raise Exception(dir_path)
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template("template.j2")

    def process_single_doc(doc):
        prompt_directory = Path(doc["prompt"])
        return {
            "query": template.render({
                "description": read_file(prompt_directory / "description"),
                "program1": read_file(prompt_directory / "program1"),
                "program2": read_file(prompt_directory / "program2"),
            }),
            "choices": ["No", "Yes"],
            "gold": 2 if (doc["answer"] == "Yes") else 1,
            "label": 2 if (doc["answer"] == "Yes") else 1,
            "split": "train",
        }

    return dataset.map(process_single_doc)