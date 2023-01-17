import prodigy
import random
import glob
import os
import re
import random
from prodigy.components.loaders import Images

ROOT_FOLDER_1 = "../dreambooth_new_prompts"
ROOT_FOLDER_2 = "../multires_dreambooth_new_prompts"
 

dataset_images = {
    "tape player": [
        "https://imgur.com/CTiTC1D.jpg",
        "https://imgur.com/WiOQyFn.jpg",
        "https://imgur.com/RLzKras.jpg",
        "https://imgur.com/fG6XHLm.jpg",
        "https://imgur.com/epS5lQq.jpg"
    ],
    "sports car": [
        "https://imgur.com/iR0S9RW.jpg",
        "https://imgur.com/daLRSER.jpg",
        "https://imgur.com/ct0rsLm.jpg",
        "https://imgur.com/m3WV2v0.jpg",
        "https://imgur.com/HEjhUPR.jpg"
    ],
    "papillon": [
        "https://imgur.com/UEJuQLI.jpg",
        "https://imgur.com/63sMr89.jpg",
        "https://imgur.com/W81gmdt.jpg",
        "https://imgur.com/8aLwZeb.jpg",
        "https://imgur.com/MUERWId.jpg"
    ],
    "monocycle": [
        "https://imgur.com/Qu6pGBM.jpg",
        "https://imgur.com/RqgN8ot.jpg",
        "https://imgur.com/cuK2a4q.jpg",
        "https://imgur.com/Fgi53rb.jpg",
        "https://imgur.com/VvxWfyv.jpg"
    ],
    "miniature poodle": [
        "https://imgur.com/dNbTmQg.jpg",
        "https://imgur.com/vrso7GV.jpg",
        "https://imgur.com/UYtW15D.jpg",
        "https://imgur.com/dNWDvcv.jpg",
        "https://imgur.com/9zJcQYp.jpg"
    ],
    "kingsnake": [
        "https://imgur.com/3qBbBDl.jpg",
        "https://imgur.com/swNiWiH.jpg",
        "https://imgur.com/duMzKyV.jpg",
        "https://imgur.com/6rX8MVw.jpg",
        "https://imgur.com/yflyViC.jpg"
    ],
    "hen": [
        "https://imgur.com/06WMKQG.jpg",
        "https://imgur.com/2npsBhZ.jpg",
        "https://imgur.com/SBvJyFH.jpg",
        "https://imgur.com/nmbUKhj.jpg",
        "https://imgur.com/hkhx3ib.jpg"
    ],
    "Gordon setter": [
        "https://imgur.com/5tZHAUT.jpg",
        "https://imgur.com/vplylHn.jpg",
        "https://imgur.com/3bkAImE.jpg",
        "https://imgur.com/VCKOLhf.jpg",
        "https://imgur.com/EEgCFQX.jpg"
    ],
    "convertible": [
        "https://imgur.com/RLdqr8Q.jpg",
        "https://imgur.com/tEJ0iy4.jpg",
        "https://imgur.com/FgulgjW.jpg",
        "https://imgur.com/T69RxjF.jpg",
        "https://imgur.com/4F8zHUP.jpg"
    ],
    "cardoon": [
        "https://imgur.com/F2fDSPT.jpg",
        "https://imgur.com/BX5pfFI.jpg",
        "https://imgur.com/mK3kAul.jpg",
        "https://imgur.com/fIdqa3l.jpg",
        "https://imgur.com/IPyoMKP.jpg"
    ]

}


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
            category_name = category_folder.split("/")[-1]
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
                yield {
                    "label": label, 
                    "options": options, 
                    "html": f"""
                    <div style="display: flex;">
                    <img src="{dataset_images[category_name][0]}" width="120px" height="120px" style="margin-right: 10px;"/>
                    <img src="{dataset_images[category_name][1]}" width="120px" height="120px" style="margin-right: 10px;"/>
                    <img src="{dataset_images[category_name][2]}" width="120px" height="120px" style="margin-right: 10px;"/>
                    <img src="{dataset_images[category_name][3]}" width="120px" height="120px" style="margin-right: 10px;"/>
                    <img src="{dataset_images[category_name][4]}" width="120px" height="120px""/>
                    </div>
                    """}


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
    # return {
    #     "dataset": dataset,
    #     "view_id": "blocks",
    #     "stream": get_stream(),
    #     "config": {
    #         "blocks": [
    #             {"view_id": "choice", "choice_style": "single", "choice_auto_accept": True,},
    #             # {"view_id": "html"},

    #         ]

    #     },
    #     "before_db": before_db,
    # }