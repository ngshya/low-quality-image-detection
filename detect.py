from joblib import load
from pandas import DataFrame, Index
from argparse import ArgumentParser
from warnings import catch_warnings, simplefilter
from lqid import get_features_from_list

list_target = [
    'sharp', 
    'defocused_blurred', 
    'motion_blurred', 
    'blur', 
    'band', 
    'noise', 
    'exposure', 
    'glare', 
    'dark', 
    'constant'
]

def detect(path:str) -> DataFrame:
    model_multi = load("model_random_forest_multi.joblib")
    df = get_features_from_list(list_files=[path])
    predictions = model_multi.predict_proba(
        df.loc[:, [col for col in df.columns if col != "path"]]
    )
    df_out = DataFrame(predictions).T
    df_out = df_out.set_index(Index([list_target[idx] for idx in df_out.index]))
    df_out = df_out.set_axis(["prediction"], axis=1)
    df_out = df_out.sort_values(["prediction"], ascending=False)
    return df_out


if __name__ == "__main__":
    parser = ArgumentParser(description="Detect whether an image is of good quality.")
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        help="input path of the image", 
        required=True
    ) 
    args = parser.parse_args()
    with catch_warnings():
        simplefilter("ignore")
        print(detect(path=args.input))

