import os
import torch
import argparse

from tqdm import tqdm
from model import StylizationModel
from dataloader import get_stylization_dataloader

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = get_stylization_dataloader(content_dir=args.content_dir, style_dir=args.style_dir)
    model = StylizationModel(level=args.level, strength=args.strength, depth=args.depth)
    
    model = model.to(device)
    model.eval()
        
    print(f'Starting the style transfer on {device} device...')
    print(f'Stylization level: {args.level} with strength {args.strength}' + ('' if args.level == 'multi' else 'and depth {args.depth}'))
    print(f'Content directory: {args.content_dir}')
    print(f'Style directory: {args.style_dir}\n')

    for inputs in tqdm(dataloader):
        content, style = inputs['content'], inputs['style']
        content_name, style_name = inputs['content_name'], inputs['style_name']
        output_path = os.path.join(output_dir, 'style_'+style_name, content_name)
        os.makedirs(output_path, exist_ok=True)
        stylized_image = model(content=content, style=style)
        stylized_image.save(output_path+'.png')
        
    print(f'Done! Output directory: {args.output_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for the model.')
    parser.add_argument('--level', type=str, default='single', choices=['single', 'multi'], help='Level of style transfer.')
    parser.add_argument('--depth', type=int, default=4, choices=[1, 2, 3, 4, 5], help='Depth of the model for single level.')
    parser.add_argument('--strength', type=float, default=1., choices=[1, 2, 3, 4, 5], help='Strength of stylization. Should be in the range [0, 1].')
    parser.add_argument('--content_dir', type=str, default='data/val2017', help='Directory for the content images.')
    parser.add_argument('--style_dir', type=str, default='data/val2017', help='Directory for the style images.')
    parser.add_argument('--output_dir', type=str, default='results/', help='Directory for the stylized images.')
    args = parser.parse_args()
    main(arg)