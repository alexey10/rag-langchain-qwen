import os

from app.config import (
    DATA_PATH,
    CHROMA_PATH
)

import shutil

def clear_workspace():

    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            os.remove(
                os.path.join(
                    DATA_PATH,
                    file
                )
            )

    if os.path.exists(
        CHROMA_PATH
    ):

        shutil.rmtree(
            CHROMA_PATH
        )
