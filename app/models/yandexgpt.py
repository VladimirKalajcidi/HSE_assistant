import os
from typing import Optional

import requests

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline


class Model():
    def __init__(self) -> None:
        self.model = HuggingFacePipeline.from_model_id(
                        model_id="mistralai/Mistral-Nemo-Instruct-2407",
                        task="text-generation",
                        pipeline_kwargs={
                            "max_new_tokens": 100,
                            "top_k": 50,
                            "temperature": 0.1,
                        },
                    )

    def ask(self, task: str, correct_solution: str, student_solution: str) -> Optional[str]:

        template = """Ты - профессиональный программист и ментор. Тебе предоставлены:
        1. Задание
        2. Правильное решение
        3. Решение ученика (которое необходимо проверить)
        Тебе необходимо дать ученику очень короткие подсказки о его ошибках, если они есть, но не давай ученику готовый код.
        Иначе будет плохо

        Задание: {task}
        Правильное решение: {correct_solution}
        Решение ученика: {student_solution}
        Комментарий преподавателя:  """

        prompt_template = PromptTemplate(
            input_variables=["task", "correct_solution", "student_solution"],
            template=template
        )

        prompt = prompt_template.format(
            task=task,
            correct_solution=correct_solution,
            student_solution=student_solution
        )

        answer = self.model(prompt)
        return answer


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    yandex_gpt = Model()
    print(yandex_gpt.ask())
