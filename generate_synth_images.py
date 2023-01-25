from pipeline import MultiResPipeline, load_learned_concepts, DreamBoothMultiResPipeline
import torch
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--checkpoints_dir', metavar='DIR', nargs='?', default='/home/id4439/multires_textual_inversion/',
                    help='path to checkpoint')
parser.add_argument('--prompts_dir', metavar='DIR', nargs='?', default='/home/id4439/multiresolution_textual_inversion/datasets/prompts/',
                    help='path to checkpoint')
parser.add_argument('--num_scales', type=int, default=10)
parser.add_argument('--out_dir', metavar='DIR', default='/home/id4439/multiresolution_textual_inversion/datasets/dreambooth_images/')



def dummy_checker(images, **kwargs): return images, False


dog_class_dict = {
        "n02106166": ["Border collie", 'S', 'dog'],
        "n02106550": ["Rottweiler", 'S', 'dog'],
        "n02107142": ["Doberman, Doberman pinscher", 'S', 'dog'],
        "n02107574": ["Greater Swiss Mountain dog", 'Slse', 'dog'],
        "n02108000": ["EntleBucher", 'S', 'dog'],
        "n02108089": ["boxer", 'S', 'dog'],
        "n02108422": ["bull mastiff", 'S', 'dog'],
        "n02108551": ["Tibetan mastiff", 'S', 'dog'],
        "n02108915": ["French bulldog", 'S', 'dog'],
        "n02110185": ["Siberian husky", 'S', 'dog'],
}

args = parser.parse_args()

checkpoint= args.checkpoints_dir
num_scales = args.num_scales
prompts_dir = args.prompts_dir
synth_imgs_dir = args.out_dir


for key_ in dog_class_dict:
    concept_name, _, _ = dog_class_dict[key_]
    concept_path = os.path.join(synth_imgs_dir+f"/{concept_name}")
    
    if not os.path.exists(concept_path):
        os.mkdir(concept_path)

    pipe = DreamBoothMultiResPipeline.from_pretrained(
        checkpoint+f"800pp-{key_}", 
        torch_dtype=torch.float16, 
        revision="fp16", 
        use_auth_token=True)
    pipe.safety_checker = dummy_checker 
    pipe = pipe.to("cuda")

    old_prompt_dir = os.path.join(args.prompts_dir+f"{concept_name}_prompt_list.txt")

    file_indx = 0
    with open(old_prompt_dir, "r") as old_prompt:
        prompts = old_prompt.readlines()
        for i, line in enumerate(prompts):
            if (i % 5 == 0): # generated images are in batches of 5, sharing the same prompt
                _, prompt = line.split(": ")
                
                imgs = pipe(prompt, num_images_per_prompt=5, num_scales=1, seed=42)
                for img in imgs:
                    img_path = f"{concept_path}/{concept_name}_{file_indx}.jpg"
                    img.save(img_path)
                    file_indx += 1
