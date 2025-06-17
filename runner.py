"""
Main interface for the user to use, here he should be allowed to run everything needed : 
    DATA 
    - Import raw dataset
    - Trigger data preprocessing
    - Generate embedding model
    - Generate tf-idf matrix
    MODELS

    APP
"""
from pathlib import Path
import os

from data import import_datasets
from data import datasets_preprocess
from data import generate_embedding_model
from data import generate_tf_idf_matrix
ROOT_FOLDER = Path(__file__).parent

HEAD_ACCEPTED_CHOICE = [
    "1",
    "DATA",
    "2",
    "MODELS",
    "3",
    "APP",
]

def cls():
    os.system('cls' if os.name == 'nt' else 'clear')
    
def head_level_input_interpreter():
    try:
        cls()
        print("Hello, I'm here to help you interact with this project.")
        print("What would you like to interact with ? (enter the number or section)")
        print("1 - DATA")
        print("2 - MODELS")
        print("3 - APP")
        resp = input().upper()
        if resp not in HEAD_ACCEPTED_CHOICE:
            cls()
            print("Please provide a number or title of the entity with which you want to interact.")
            input()
            head_level_input_interpreter()
        if resp in HEAD_ACCEPTED_CHOICE[0:2]:
            data_level_input_interpreter()
        if resp in HEAD_ACCEPTED_CHOICE[2:4]:
            models_level_input_interpreter()
        if resp in HEAD_ACCEPTED_CHOICE[4:]:
            app_level_input_interpreter()
    except Exception as e:
        print(f"Somethin went wrong during evaluation of your input : {e}")


def data_level_input_interpreter():
    cls()
    print("DATA")
    print("Please choose one of the option below : (type '0' or 'home' to go back to main menu)")
    print("""
        1 - Import raw dataset
        2 - Trigger data preprocessing
        3 - Generate embedding model
        4 - Generate tf-idf matrix
        0 (home) - Go back to main menu
        """)
    resp = input().lower()
    if resp in ["0", "home"]:
        head_level_input_interpreter()

    resp = int(resp)
    if resp not in range(0,5):
        cls()
        print("Please provide a value between 0 and 4 included.")
        data_level_input_interpreter()
    if resp == 1:
        import_datasets.main()
    if resp == 2:
        datasets_preprocess.main()
    if resp == 3:
        generate_tf_idf_matrix.main()
    if resp == 4:
        generate_embedding_model.main()

def models_level_input_interpreter():
    print("models")
    

def app_level_input_interpreter():
    print("app")
    

if __name__ == "__main__":
    head_level_input_interpreter()