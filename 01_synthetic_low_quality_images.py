from glob import glob
from tqdm import tqdm
from os.path import basename
from PIL import Image
from numpy.random import seed as np_seed
from lqid import img2matrix, random_band, random_noise, random_blur, random_exposure, random_dark, random_constant, random_glare
from argparse import ArgumentParser
from os.path import exists 
from os import makedirs


def generate_synthetic_images(path_input:str, path_output:str):
    dict_trasform = {
        "band": random_band, 
        "noise": random_noise, 
        "blur": random_blur, 
        "exposure": random_exposure, 
        "dark": random_dark, 
        "constant": random_constant, 
        "glare": random_glare
    }
    list_files_sharp = glob(path_input+"*")
    if not exists(path_output):
        makedirs(path_output)
    for k in dict_trasform.keys():
        if not exists(path_output+"/"+k):
            makedirs(path_output+"/"+k)
    for path_img in tqdm(list_files_sharp):
        dict_img = img2matrix(path_img)
        for k in dict_trasform.keys():
            try:
                matrix = dict_trasform[k](dict_img["rgb"])
                img = Image.fromarray(matrix.astype('uint8')).convert('RGB')
                img.save("%s/%s/%s_%s.jpeg" % (path_output, k, basename(path_img), k))
            except Exception as e:
                print(str(e))


if __name__ == "__main__":

    parser = ArgumentParser(description="Generate synthetic low quality images from high quality images.")
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        default="dataset_dms/sharp/",
        help="input folder containing good quality images"
    ) 
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default="dataset_synthetic/", 
        help="output folder containing generated low quality images"
    )
    parser.add_argument(
        '-s', '--seed', 
        type=int, 
        default=1102, 
        help="Random seed"
    )
    args = parser.parse_args()

    np_seed(args.seed)
    generate_synthetic_images(path_input=args.input, path_output=args.output)