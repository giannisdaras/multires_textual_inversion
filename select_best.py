import prodigy
import random
import glob
import os
import re
import random
from prodigy.components.loaders import Images

ROOT_FOLDER_1 = "../dreambooth_new_prompts"
ROOT_FOLDER_2 = "../multires_dreambooth_new_prompts"
 

def bernoulli(p=0.5):
    return 1 if random.random() < p else 0


def before_db(examples):
    for eg in examples:
        del eg['options'][0]['image']
        del eg['options'][1]['image']
    return examples
    
@prodigy.recipe("dreambooth_comp")
def A_B_testing(dataset, label):
    def get_stream():
        for category_folder in glob.iglob(os.path.join(ROOT_FOLDER_1, "*")):
            prompts_file = os.path.join(ROOT_FOLDER_2, category_folder.split('/')[-1], "prompt_list.txt")
            with open(prompts_file, "r") as f:
                prompts = f.readlines()
            stream1 = Images(category_folder)
            stream2 = Images(os.path.join(ROOT_FOLDER_2, category_folder.split('/')[-1]))
            for eg1, eg2, prompt in zip(stream1, stream2, prompts):
                if bernoulli():
                    options = [{"id": 1, "image": eg1['image'], "path": eg1['path'], "gt": "baseline"}, {"id": 2, "image": eg2['image'], "path": eg2['path'], "gt": "ours"}]
                else:
                    options = [{"id": 1, "image": eg2['image'], "path": eg2['path'], "gt": "ours"}, {"id": 2, "image": eg1['image'], "path": eg1['path'], "gt": "baseline"}]

                # remove {number}: and also replace <S[0]> with the category name
                label = re.search(r"\d+: (.*)", prompt).group(1).replace("<S[0]>", category_folder.split("/")[-1])
                yield {"label": label, "options": options}



    return {
        "dataset": dataset,
        "view_id": "choice",
        "stream": get_stream(),
        "config": {
            "choice_style": "single",
            "choice_auto_accept": True,
        },
        "before_db": before_db,
    }