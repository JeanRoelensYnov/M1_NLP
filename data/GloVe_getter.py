import os
import requests
import zipfile
from pathlib import Path

PARENT_FOLDER = Path(__file__).parent
URL = "http://nlp.stanford.edu/data/glove.6B.zip"
OUTPUT_FOLDER = PARENT_FOLDER / "glove_models"

def download_glove():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    files_in_folder = OUTPUT_FOLDER.glob('*')
    files = [x for x in files_in_folder if x.is_file()]
    if len(files) == 4:
        print("All GloVe 6B models have already been downloaded")
        return
    print("Downloading GloVe 6B models..")
    response = requests.get(URL, stream=True, verify=False)
    OUTPUT_FILE = OUTPUT_FOLDER / "glove.6B.zip"
    with open(OUTPUT_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    
    print("Extracting..")
    with zipfile.ZipFile(OUTPUT_FILE, "r") as zf:
        zf.extractall(OUTPUT_FOLDER)
    
    print("Tidying the zip file..")
    os.remove(OUTPUT_FILE)

    print(f"GloVe 6B downloaded. You can find it in {OUTPUT_FOLDER}.")

if __name__ == "__main__":
    download_glove()

