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

import argparse
from generate import generate_txt2img_dataset, generate_img2img_dataset, generate_inpaint_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Datasets using Stable Diffusion Pipelines")
    parser.add_argument('mode', choices=['txt2img', 'img2img', 'inpaint'], help="Select the mode: txt2img, img2img, or inpaint")
    parser.add_argument('config', type=str, help="Path to the dataset configuration JSON file")

    args = parser.parse_args()

    if args.mode == 'txt2img':
        generate_txt2img_dataset(args.config)
    elif args.mode == 'img2img':
        generate_img2img_dataset(args.config)
    elif args.mode == 'inpaint':
        generate_inpaint_dataset(args.config)
