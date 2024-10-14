import os
from typing import Optional

import requests

from langchain.chains import LLMChain
from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate


class YandexModel():
    def __init__(self) -> None:
        self.model = YandexGPT(api_key="AQVN0HYDrL1Juo_atEB9TGWixMZHv1kbA1O-9W5n", 
                               folder_id="b1gmupmk7abqoalp07e5")

    def ask(self, task: str, correct_solution: str, student_solution: str) -> Optional[str]:
        template = """Ты - профессиональный программист и ментор. Тебе предоставлены:
        1. Задание
        2. Правильное решение
        3. Решение ученика (которое необходимо проверить)
        Тебе необходимо дать ученику очень короткие ответы о его ошибках, если они есть. 
        Не пиши исправленный код, просто намекни ученику, в каком месте ошибка. Еще раз:
        в твоих ответах не должно быть готового кода! Ты можешь давать подсказки, но писать код ты не можешь!
        Вот примеры:

        1. Задание: Реализуйте программу, которая проверит, что цвет используется только в проекте по созданию логотипа, но не в проекте по созданию дизайна сайта:
        Даны два списка logo_project и cite_project с кодами используемых цветов (строки).
        В переменную color считывается код цвета (строка). Этот код уже написан.
        Программа должна проверять, что код цвета color есть только в списке logo_project, и если да, то печатать True. 
        В остальных случаях программа печатает False. 
        Правильное решение: logo_project = ['#a7a8f0', '#a7f0ca', '#b3b4e4', '#e4b3cd', '#e4e3b3', '#c0ced7']
        cite_project = ['#e4e3b3', '#a7a8f0', '#ccb1e6', '#b4f99e', '#f9b59e', '#c0ced7']

        color = input()

        if color in logo_project and not(color in cite_project):
            print(True)
        else:
            print(False)
        Решение ученика: logo_project = ['#a7a8f0', '#a7f0ca', '#b3b4e4', '#e4b3cd', '#e4e3b3', '#c0ced7']
        cite_project = ['#e4e3b3', '#a7a8f0', '#ccb1e6', '#b4f99e', '#f9b59e', '#c0ced7']

        color = input()

        if color in logo_project and color in cite_project:
            print(True)
        else:
            print(False)
        Комментарий преподавателя: Ошибка в открытых тестах. 
        Обратите внимание на неверный оператор сравнения — необходимо проверить, что цвет не находится в списке cite_project.

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

        llm = YandexGPT(api_key="AQVN0HYDrL1Juo_atEB9TGWixMZHv1kbA1O-9W5n", 
                        folder_id="b1gmupmk7abqoalp07e5")

        answer = llm(prompt)
        return answer


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    yandex_gpt = YandexModel()
    print(yandex_gpt.ask())
