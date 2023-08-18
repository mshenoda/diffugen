#   DatasetDiffusion - Generating Labeled Image Datasets using Stable Diffusion Pipelines
#   Copyright (C) 2023  Michael Shenoda
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import torch
import random
import json
import yaml
import argparse
from tqdm import tqdm
from itertools import product
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from ultralytics import YOLO
from labeldiffusion import LabelDiffusion, create_generator, draw_bounding_boxes, draw_binary_mask, concat, random_unique_int_list

def generate_prompts_from_json(json_data):
    prompts = []
    
    for prompt_template in json_data["txt2img"]["prompts"]:
        for combination in product(
            json_data["txt2img"]["view_points"],
            json_data["txt2img"]["object_names"],
            json_data["txt2img"]["times_of_day"],
            json_data["txt2img"]["sky_conditions"],
            json_data["txt2img"]["weather_conditions"]
        ):
            view_point, object_name, time_of_day, sky_condition, weather_condition = combination
            prompt = prompt_template.format(
                view_point=view_point,
                object_name=object_name,
                time_of_day=time_of_day,
                sky_condition=sky_condition,
                weather_condition=weather_condition
            )
            prompts.append({
                "prompt": prompt,
                "prompt_template": prompt_template,
                "view_point": view_point,
                "object_name": object_name,
                "time_of_day": time_of_day,
                "sky_condition": sky_condition,
                "weather_condition": weather_condition
            })
    
    return prompts

def create_output_directories(dataset_name, pipe_name, shard_index):
    shard_dir = f"{shard_index:04d}"
    root_dir = f"dataset_output/{dataset_name}/{pipe_name}"
    images_dir = f"{root_dir}/images/{shard_dir}"
    masks_dir = f"{root_dir}/masks/{shard_dir}"
    attentions_dir = f"{root_dir}/attentions/{shard_dir}"
    bboxes_dir = f"{root_dir}/labels/{shard_dir}"
    results_dir = f"{root_dir}/visualizations/{shard_dir}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(attentions_dir, exist_ok=True)
    os.makedirs(bboxes_dir, exist_ok=True)
    return root_dir, images_dir, masks_dir, attentions_dir, bboxes_dir, results_dir

def print_prompt_info(prompt_info):
    prompt = prompt_info["prompt"]
    view_point = prompt_info["view_point"]
    object_name = prompt_info["object_name"]
    time_of_day = prompt_info["time_of_day"]
    sky_condition = prompt_info["sky_condition"]
    weather_condition = prompt_info["weather_condition"]

    print("=" * 100)
    print("  Prompt:", prompt)
    print("-" * 100)
    print("  View Point:", view_point)
    print("  Object Name:", object_name)
    print("  Time of Day:", time_of_day)
    print("  Sky Condition:", sky_condition)
    print("  Weather Condition:", weather_condition)
    print("=" * 100)

def create_class_id_mapping(dataset_to_model_class_map, classes_file_path):
    # Load the model classes from classes.txt
    with open(classes_file_path, 'r') as classes_file:
        model_classes = [line.strip() for line in classes_file.readlines()]

    index_mapping = {}

    for dataset_class_index, model_class in enumerate(dataset_to_model_class_map.values()):
        # Find the index of the model class in the model_classes list
        model_class_index = model_classes.index(model_class)
        
        # Store the index mapping
        index_mapping[dataset_class_index] = model_class_index

    return index_mapping

def save_json(data, filename):
    """
    Save a data as a JSON file.
    
    Args:
        data (any): The dictionary to be saved as JSON.
        filename (str): The name of the JSON file to be created.
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def generate_txt2img_dataset(dataset_config_path:str):
    # Read JSON data from file
    with open(dataset_config_path, "r") as json_file:
        json_data = json.load(json_file)

    # Extract general settings
    unsupervised = json_data["unsupervised"]
    label_model = json_data["label_model"]
    steps = json_data["steps"]
    width = json_data["image_size"]["width"]
    height = json_data["image_size"]["height"]
    dataset_to_model_class_map = json_data["dataset_to_model_class_map"]
    label_model_classes_file = json_data["label_model"]["classes"]
    images_per_shard = json_data["images_per_shard"]
    dataset_name = json_data["dataset_name"]

    # Extract txt2img settings
    negative_prompt = json_data["txt2img"]["negative_prompt"]
    seed_range = json_data["txt2img"]["seed_range"]
    seed_count_per_prompt = json_data["txt2img"]["seed_count_per_prompt"]
    image_count = json_data["txt2img"]["image_count"]
    textual_inversion = False
    if "textual_inversion" in json_data:
        textual_inversion = True
        textual_inversion_weights = json_data["txt2img"]["textual_inversion"]["weights"]

    # Create dataset to segmentation model class mappings
    class_id_mapping = create_class_id_mapping(dataset_to_model_class_map, label_model_classes_file)

    # Print the mappings
    print("Class Name Mapping:", dataset_to_model_class_map)
    print("Class Id Mapping:", class_id_mapping)

    # Generate prompts from the JSON data
    prompts_list = generate_prompts_from_json(json_data)

    sd_model_name = "Realistic_Vision_V4.0"
    sd_model_path = f"../models/{sd_model_name}" # Realistic_Vision_V4.0  Realistic_Vision_V5.1_noVAE
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=sd_model_path, torch_dtype=torch.float16)
    if textual_inversion:
        pipe.load_textual_inversion(textual_inversion_weights)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing(1)
    #pipe = pipe.to("cuda:0")
    
    yolo = None
    label_model_type = label_model["type"]
    if label_model_type == "YOLOSeg":
        yolo = YOLO(label_model["weights"])
    else:
        print(f"Error: model type{label_model_type} is not supported!")
        print("Supported label model types are: YOLOSeg")
        exit()

    label_pipe = LabelDiffusion(pipe, yolo, class_id_mapping)
    
    
    # Calculate the number of prompts to sample
    max_image_count = len(prompts_list)#*seed_count_per_prompt
    if image_count < 1:
        print(f"Error: image_count cannot be less than 1, image_count={image_count}")
        return
    
    image_count = max_image_count if image_count > max_image_count else image_count
    sample_size = int(image_count)
    
    # Randomly sample prompts
    random_prompts_list = random.sample(prompts_list, sample_size)
    print("Max possible image count: ", max_image_count)
    print("Configured image count: ", image_count)
    initial_seed = 9782091#random.randint(seed_range["min"], seed_range["max"]) # set to fixed seed, if need re-producablity 
    generator = create_generator(initial_seed) 
    
    shard_index = 1
    current_image_count = 0

    sd_model_info = {
        "name": sd_model_name,
        "torch_dtype": "torch.float16",
        "initial_seed": initial_seed
    }

    if textual_inversion:
        sd_model_info["load_textual_inversion"] = textual_inversion_weights

    for index, prompt_info in enumerate(random_prompts_list, start=1):
        print_prompt_info(prompt_info)

        prompt = prompt_info["prompt"]
        view_point = prompt_info["view_point"]
        object_name = prompt_info["object_name"]
        time_of_day = prompt_info["time_of_day"]
        sky_condition = prompt_info["sky_condition"]
        weather_condition = prompt_info["weather_condition"]

        root_dir, images_dir, masks_dir, attentions_dir, bboxes_dir, results_dir = create_output_directories(dataset_name, "txt2img", shard_index)
        
        sd_model_info_json_file = f"{root_dir}/sd_model_info.json"
        if os.path.exists(sd_model_info_json_file) == False:
            save_json(sd_model_info, sd_model_info_json_file)

        seeds = random_unique_int_list(seed_count_per_prompt, seed_range["min"], seed_range["max"])
        
        for seed in seeds:
            current_image_count+=1
            print("\nseed=", seed)
            print(f"current_image_count={current_image_count}/{image_count}", )
            generator.manual_seed(seed)
            output, labels, binary_mask, attention_map = label_pipe(object_name, prompt, negative_prompt, steps, generator, width, height, unsupervised=unsupervised)

            if len(labels) == 0:
                continue

            label_file_data = {
                "prompt_details": {
                    "seed": seed,
                    "steps": steps,
                    "prompt": prompt,
                    "prompt_template": prompt_info["prompt_template"],
                    "view_point" : view_point,
                    "object_name": object_name,
                    "time_of_day": time_of_day,
                    "sky_condition": sky_condition,
                    "weather_condition": weather_condition
                },
                "labels": labels
            }

            output_image = output.images[0]

            name = object_name
            basename = f"{name}_{view_point}_{time_of_day}_{sky_condition}_{weather_condition}_{seed}"
            image_filename = f"{basename}.png"
            label_filename = f"{basename}.json"

            save_json(label_file_data, f"{bboxes_dir}/{label_filename}")
            output_image.save(f"{images_dir}/{image_filename}")
            attention_map.save(f"{attentions_dir}/{image_filename}")
            binary_mask.save(f"{masks_dir}/{image_filename}")
            bbox_image = draw_bounding_boxes(output_image, labels)
            mask_image = draw_binary_mask(output_image, binary_mask)
            results_image = concat(concat(bbox_image, mask_image), attention_map)
            results_image.save(f"{results_dir}/{image_filename}")

            if index % images_per_shard == 0:
                shard_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Dataset using Text to Image Diffusion")
    parser.add_argument('config', type=str, help="Path to the dataset configuration JSON file")

    args = parser.parse_args()
    generate_txt2img_dataset