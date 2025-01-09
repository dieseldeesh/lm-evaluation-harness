import datasets
import jinja2
import os
from pathlib import Path

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
    return content

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:

    dir_path = os.path.dirname(os.path.realpath(__file__))
    templateLoader = jinja2.FileSystemLoader(searchpath=dir_path)
    templateEnv = jinja2.Environment(loader=templateLoader)

    def process_single_doc(doc):
        prompt_directory = Path(dir_path) / doc["prompt"]
        template = templateEnv.get_template(doc["template"])
        return {
            "query": template.render({
                "description": read_file(prompt_directory / "description"), # sometimes description won't exist...
                "program1": read_file(prompt_directory / "program1"),
                "program2": read_file(prompt_directory / "program2"),
            }),
            "choices": ["No", "Yes"],
            "gold": 2 if (doc["answer"] == "Yes") else 1,
            "label": 2 if (doc["answer"] == "Yes") else 1,
        }

    return {
        "train": [],
        "validation": dataset.map(process_single_doc)
    }