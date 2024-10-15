import os

import pandas as pd
from dotenv import load_dotenv

from app.models.yandexgpt import Model
from app.utils.submit import generate_submit

if __name__ == "__main__":
    load_dotenv()


    llm = Model()


    def predict(row: pd.Series) -> str:
        return llm.ask(row["task"], row["authors_solution"], row["student_solution"])


    generate_submit(
        test_solutions_path="data/raw/test/solutions.xlsx",
        predict_func=predict,
        save_path="data/processed/submission.csv",
        use_tqdm=True,
    )
