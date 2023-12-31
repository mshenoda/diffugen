#   DiffuGen - Generating Labeled Image Datasets using Stable Diffusion Pipelines
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
import json
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from ultralytics import YOLO
from labeldiffusion import LabelDiffusionImg2Img, create_generator, draw_bounding_boxes, draw_binary_mask, concat, random_unique_int_list

__all__ = ["generate_img2img_dataset"]

def extract_prompt_details_from_json(json_content):
    try:
        prompt_details = json_content["prompt_details"]
        return prompt_details
    except KeyError:
        return None

def extract_prompt_details_with_images(root_directory, image_format):
    prompt_details_with_images = []

    labels_directory = os.path.join(root_directory, "labels")
    images_directory = os.path.join(root_directory, "images")

    for shard_folder in os.listdir(labels_directory):
        shard_labels_directory = os.path.join(labels_directory, shard_folder)
        shard_images_directory = os.path.join(images_directory, shard_folder)

        if os.path.isdir(shard_labels_directory) and os.path.isdir(shard_images_directory):
            for root, dirs, files in os.walk(shard_labels_directory):
                for file in files:
                    if file.endswith('.json'):
                        json_path = os.path.join(root, file)
                        label_filename = os.path.splitext(file)[0]  # Remove ".json" extension
                        image_filename = label_filename + "." + image_format
                        image_path = os.path.join(shard_images_directory, image_filename)
                        
                        with open(json_path, 'r') as json_file:
                            try:
                                json_content = json.load(json_file)
                                prompt_details = extract_prompt_details_from_json(json_content)
                                if prompt_details:
                                    prompt_details["image_path"] = image_path
                                    prompt_details_with_images.append(prompt_details)
                            except json.JSONDecodeError:
                                print(f"Error loading JSON from file: {json_path}")

    return prompt_details_with_images

def generate_prompts(json_data):
    prompts = []
    img2img = json_data["img2img"]
    source_dataset_path = img2img["source_dataset"]
    txt2img_prompt_details_list = extract_prompt_details_with_images(os.path.join(source_dataset_path), json_data["image_format"])
    for txt2img_details in txt2img_prompt_details_list:
        if "prompt_template" in txt2img_details:
            prompt_template = txt2img_details["prompt_template"]
            for time_of_day in img2img["times_of_day"]:
                if time_of_day in txt2img_details["time_of_day"]:
                    continue
                for sky_condition in img2img["sky_conditions"]:
                    if sky_condition in txt2img_details["sky_condition"]:
                        continue
                    for weather_condition in img2img["weather_conditions"]:
                        if weather_condition in txt2img_details["weather_condition"]:
                            continue
                        prompt = prompt_template.format(
                            view_point=txt2img_details["view_point"],
                            object_name=txt2img_details["object_name"],
                            time_of_day=time_of_day,
                            sky_condition=sky_condition,
                            weather_condition=weather_condition
                        )
                        prompts.append({
                            "image_path": txt2img_details["image_path"],
                            "seed": txt2img_details["seed"],
                            "prompt": prompt,
                            "prompt_template": prompt_template,
                            "view_point": txt2img_details["view_point"],
                            "object_name": txt2img_details["object_name"],
                            "time_of_day": time_of_day,
                            "sky_condition": sky_condition,
                            "weather_condition": weather_condition
                        })
    return prompts

def create_output_directories(file_path, output_root, dataset_name):
    # Define the directory structure
    directory_structure = [
        'attentions', 'images', 'labels', 'masks', 'visualizations'
    ]

    # Split the file path
    parts = file_path.split(os.path.sep)

    # Remove the last two parts to get the "cars-extended" directory
    cars_extended_dir = os.path.join(output_root, dataset_name)

    # Create the img2img directory
    img2img_dir = os.path.join(cars_extended_dir, 'img2img')

    created_directories = []

    # Iterate through the directory structure to create subdirectories
    for sub_dir in directory_structure:
        sub_path = os.path.join(img2img_dir, sub_dir, parts[-2])
        os.makedirs(sub_path, exist_ok=True)
        created_directories.append(sub_path)

    return created_directories

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

def generate_img2img_dataset(dataset_config:str):
    # Read JSON data from file
    with open(dataset_config, "r") as json_file:
        json_data = json.load(json_file)

    # Extract general settings
    dataset_name = json_data["dataset_name"]
    dataset_root = json_data["dataset_root"]
    unsupervised = json_data["unsupervised"]
    label_model = json_data["label_model"]
    steps = json_data["steps"]
    width = json_data["image_size"]["width"]
    height = json_data["image_size"]["height"]
    image_format = json_data["image_format"]
    dataset_to_model_class_map = json_data["dataset_to_model_class_map"]
    label_model_classes_file = json_data["label_model"]["classes"]

    # Extract img2img settings
    negative_prompt = json_data["txt2img"]["negative_prompt"]
    stable_diffusion_model = json_data["img2img"]["stable_diffusion_model"]

    # Create dataset to segmentation model class mappings
    class_id_mapping = create_class_id_mapping(dataset_to_model_class_map, label_model_classes_file)

    # Print the mappings
    print("Class Name Mapping:", dataset_to_model_class_map)
    print("Class Id Mapping:", class_id_mapping)

    # Generate prompts from the JSON data
    prompts_list = generate_prompts(json_data)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path=stable_diffusion_model, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing(1)
    #pipe.load_textual_inversion(textual_inversion_weights)
    #pipe = pipe.to("cuda:0")

    sd_model_info = {
        "model": stable_diffusion_model,
        "torch_dtype": "torch.float16"
    }

    yolo = None
    label_model_type = label_model["type"]
    if label_model_type == "YOLOSeg":
        yolo = YOLO(label_model["weights"])
    else:
        print(f"Error: model type{label_model_type} is not supported!")
        print("Supported label model types are: YOLOSeg")
        exit()

    label_pipe = LabelDiffusionImg2Img(pipe, yolo, class_id_mapping)
    
    
    # Calculate the number of prompts to sample
    image_count = len(prompts_list)
    print("Image count: ", image_count)
    seed = 9782091 # set to fixed seed, if need re-producablity 
    generator = create_generator(seed) 

    current_image_count = 0
    for index, prompt_info in enumerate(prompts_list, start=1):
        print_prompt_info(prompt_info)

        prompt = prompt_info["prompt"]
        view_point = prompt_info["view_point"]
        object_name = prompt_info["object_name"]
        time_of_day = prompt_info["time_of_day"]
        sky_condition = prompt_info["sky_condition"]
        weather_condition = prompt_info["weather_condition"]
        image_path = prompt_info["image_path"]
        seed = prompt_info["seed"]
        print("input image_path=", image_path)
        attentions_dir, images_dir, labels_dir, masks_dir, vis_dir = create_output_directories(image_path, dataset_root, dataset_name)
        
        sd_model_info_json_file = f"{dataset_root}/{dataset_name}/img2img/stable_diffusion_model_info.json"
        if os.path.exists(sd_model_info_json_file) == False:
            save_json(sd_model_info, sd_model_info_json_file)

        current_image_count+=1
        print("\nseed=", seed)
        print(f"current_image_count={current_image_count}/{image_count}", )
        generator.manual_seed(seed)
        image = Image.open(image_path)
        output, labels, binary_mask, attention_map = label_pipe(image, object_name, prompt, negative_prompt, steps, generator, width, height, unsupervised=unsupervised)

        if len(labels) == 0:
            print("\n***!!! no labels, skipping image !!!***\n")
            return

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

        basename = f"{name}_{view_point}_{time_of_day}_{sky_condition}_{weather_condition}_{index}_{seed}"
        image_filename = f"{basename}.{image_format}"
        label_filename = f"{basename}.json"
        print(f"\n saving: {images_dir}/{image_filename} \n")
        save_json(label_file_data, f"{labels_dir}/{label_filename}")
        output_image.save(f"{images_dir}/{image_filename}")
        attention_map.save(f"{attentions_dir}/{image_filename}")
        binary_mask.save(f"{masks_dir}/{image_filename}")
        bbox_image = draw_bounding_boxes(output_image, labels)
        mask_image = draw_binary_mask(output_image, binary_mask)
        visualization_image = concat(concat(bbox_image, mask_image), attention_map)
        visualization_image.save(f"{vis_dir}/{image_filename}")
