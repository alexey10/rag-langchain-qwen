import os
from app.config import DATA_PATH


def get_indexed_documents():

    return sorted([
        file
        for file in os.listdir(DATA_PATH)
        if file.endswith(".pdf")
    ])
