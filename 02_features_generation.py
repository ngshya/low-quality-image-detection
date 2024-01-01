from glob import glob
from lqid import get_features_from_list
from argparse import ArgumentParser
from time import time
from os.path import split


def generate_features_df(path_dms:str, path_synthetic:str, output_csv:str):
    list_files = glob(path_dms+"/*/*") \
        + glob(path_synthetic+"/*/*")
    df = get_features_from_list(list_files)
    df["target"] = [split(split(p)[0])[-1] for p in df["path"]]
    df["dataset"] = [split(split(p)[0])[-2] for p in df["path"]]
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":

    parser = ArgumentParser(description="Generate features from images.")
    parser.add_argument(
        '-d', '--dms', 
        type=str, 
        default="dataset_dms",
        help="input folder containing dms images"
    ) 
    parser.add_argument(
        '-s', '--synthetic', 
        type=str, 
        default="dataset_synthetic", 
        help="input folder containing generated low quality images"
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default="df_public.csv", 
        help="output csv"
    )
    args = parser.parse_args()

    tms_start = time()
    generate_features_df(
        path_dms=args.dms, 
        path_synthetic=args.synthetic, 
        output_csv=args.output
    )
    print("Elapsed", (time()-tms_start)/60.0, "minutes")

