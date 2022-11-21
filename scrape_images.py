from huggingface_hub import HfApi
import requests
from tqdm import tqdm
import os

# create output directory if it doesn't exist
if not os.path.exists("datasets/"):
    os.makedirs("datasets/")

TOP_K = 5
MAX_IMAGES = 4
api = HfApi()
models_list = api.list_models(author="sd-concepts-library", sort="likes", direction=-1)[:TOP_K]

for model in tqdm(models_list):
  model_id = model.modelId
  model_dir = os.path.join("datasets", model_id)
  os.makedirs(model_dir, exist_ok = True)

  for i in range(MAX_IMAGES):
        url = f"https://huggingface.co/{model_id}/resolve/main/concept_images/{i}.jpeg"
        image_download = requests.get(url)
        url_code = image_download.status_code
        if url_code == 200:
            file = open(os.path.join(model_dir, f"{i}.jpeg"), "wb") ## Creates the file for image
            file.write(image_download.content) ## Saves file content
            file.close()
