import random 
from pipeline import MultiResPipeline, load_learned_concepts, DreamBoothMultiResPipeline
import torch
import os


def dummy_checker(images, **kwargs): return images, False


dog_class_dict = {
        "n02106166": ["Border collie", 'S', 'dog'],
        "n02106550": ["Rottweiler", 'S', 'dog'],
        "n02107142": ["Doberman, Doberman pinscher", 'S', 'dog'],
        "n02107574": ["Greater Swiss Mountain dog", 'S', 'dog'],
        "n02108000": ["EntleBucher", 'S', 'dog'],
        "n02108089": ["boxer", 'S', 'dog'],
        "n02108422": ["bull mastiff", 'S', 'dog'],
        "n02108551": ["Tibetan mastiff", 'S', 'dog'],
        "n02108915": ["French bulldog", 'S', 'dog'],
        "n02110185": ["Siberian husky", 'S', 'dog'],
}


checkpoint= ""
num_scales = ""
old_concept_dir = f""
diffusion_model_path = f""
synth_imgs_dir = f""


for key_ in dog_class_dict:
    concept_name, _, _ = dog_class_dict[key_]
    concept_path = os.path.join(synth_imgs_dir+f"/{concept_name}")
    
    if not os.path.exists()
        os.mkdir(concept_path)

    pipe = DreamBoothMultiResPipeline.from_pretrained(
        diffusion_model_path+f"/{key_}", 
        torch_dtype=torch.float16, 
        revision="fp16", 
        use_auth_token=True)
    pipe.safety_checker = dummy_checker 
    pipe = pipe.to("cuda")

    old_prompt_dir = os.path.join(old_concept_dir+f"{concept_name}/prompt_list.txt")

    file_indx = 0
    with open(old_prompt_dir, "r") as old_prompt:
        prompts = old_prompt.readlines()
        for i, line in enumerate(prompts):
            if (i % 5 == 0): # generated images are in batches of 5, sharing the same prompt
                _, prompt = line.split(": ")
                
                imgs = pipe(prompt, num_images_per_prompt=5, num_scales=1, seed=42)
                for img in imgs:
                    img_path = f"{concept_path}{concept_name}_{file_indx}.jpg"
                    img.save(img_path)
                    file_indx += 1
