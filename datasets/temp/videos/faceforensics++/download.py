#!/usr/bin/env python
# Multithreaded FaceForensics++ downloader

import argparse
import os
import urllib.request
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join

FILELIST_URL = 'misc/filelist.json'
DEEPFAKE_DETECTION_URL = 'misc/deepfake_detection_filenames.json'

DATASETS = {
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}

ALL_DATASETS = list(DATASETS.keys())

COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks']
SERVERS = {
    'EU': 'http://canis.vc.in.tum.de:8100/',
    'EU2': 'http://kaldir.vc.in.tum.de/faceforensics/',
    'CA': 'http://falas.cmpt.sfu.ca:8100/'
}


def parse_args():
    parser = argparse.ArgumentParser("FaceForensics++ fast downloader")
    parser.add_argument("output_path")
    parser.add_argument("-d", "--dataset", default="all",
                        choices=ALL_DATASETS + ["all"])
    parser.add_argument("-c", "--compression", default="c40",
                        choices=COMPRESSION)
    parser.add_argument("-t", "--type", default="videos",
                        choices=TYPE)
    parser.add_argument("-n", "--num_videos", type=int)
    parser.add_argument("--server", default="EU2",
                        choices=list(SERVERS.keys()))
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()

    args.base_url = SERVERS[args.server] + "v3/"

    return args


def get_filelist(args, dataset_path):

    if "actors" in dataset_path or "DeepFakeDetection" in dataset_path:
        data = json.loads(
            urllib.request.urlopen(args.base_url + DEEPFAKE_DETECTION_URL)
            .read().decode()
        )
        if "actors" in dataset_path:
            return data["actors"]
        return data["DeepFakesDetection"]

    data = json.loads(
        urllib.request.urlopen(args.base_url + FILELIST_URL)
        .read().decode()
    )

    filelist = []

    if "original" in dataset_path:
        for pair in data:
            filelist += pair
    else:
        for pair in data:
            filelist.append("_".join(pair))
            filelist.append("_".join(pair[::-1]))

    return filelist


def download_one(url, path):

    if os.path.exists(path):
        return "skip"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        urllib.request.urlretrieve(url, path)
        return "done"
    except Exception as e:
        return str(e)


def threaded_download(filelist, base_url, out_dir, workers):

    futures = []
    results = {"done": 0, "skip": 0, "fail": 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:

        for f in filelist:
            url = base_url + f
            path = join(out_dir, f)

            futures.append(executor.submit(download_one, url, path))

        for future in tqdm(as_completed(futures), total=len(futures)):
            r = future.result()
            if r == "done":
                results["done"] += 1
            elif r == "skip":
                results["skip"] += 1
            else:
                results["fail"] += 1

    print("\nSummary:")
    print(results)


def main():

    args = parse_args()

    print("You must agree to the FaceForensics++ Terms of Use.")
    print("Press ENTER to continue or CTRL+C to abort.")
    input()

    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    for dataset in datasets:

        dataset_path = DATASETS[dataset]

        print(f"\nPreparing dataset: {dataset}")

        filelist = get_filelist(args, dataset_path)

        if args.num_videos:
            filelist = filelist[:args.num_videos]

        filelist = [f + ".mp4" for f in filelist]

        base_url = f"{args.base_url}{dataset_path}/{args.compression}/{args.type}/"

        out_dir = join(args.output_path,dataset_path,args.compression,args.type)

        print("Output:", out_dir)
        print("Files:", len(filelist))
        
        threaded_download(filelist, base_url, out_dir, args.workers)


if __name__ == "__main__":
    main()