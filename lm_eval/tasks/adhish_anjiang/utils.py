import datasets
import jinja2
import os
from pathlib import Path

variables = ["description", "program1", "program2"]

def read_file(path):
    try:
        with open(path, 'r') as f:
            content = f.read()
        return content
    except:
        return None
    
def read_directory(directory):
    variable_dict = dict()
    
    for variable in variables:
        variable_value = read_file(directory / variable)
        if (variable_value):
            variable_dict[variable] = variable_value
    
    return variable_dict

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:

    dir_path = os.path.dirname(os.path.realpath(__file__))
    templateLoader = jinja2.FileSystemLoader(searchpath=dir_path)
    templateEnv = jinja2.Environment(loader=templateLoader)

    def process_single_doc(doc):
        prompt_directory = Path(dir_path) / doc["prompt"]
        template = templateEnv.get_template(doc["template"])

        return {
            "query": template.render(read_directory(prompt_directory)),
            "choices": ["No", "Yes"],
            "answer": doc["answer"],
        }

    return dataset.map(process_single_doc)