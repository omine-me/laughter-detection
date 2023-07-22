# convert json to csv like this header:
# ['offset','duration','subsampled_offset','subsampled_duration','audio_path','label']

import json
import glob
import os.path as osp
import csv

basedir = r"D:\datasets\multi\time_anotation"
jsondir = osp.join(basedir, r"result\gt")
json_prefix = "show_"
test_prefix = "test_"

print(osp.join(jsondir, f"{json_prefix}*.json"))
assert len(glob.glob(osp.join(jsondir, f"{json_prefix}*.json"))), "Dataset files seems zero"
    
with open(r"D:\datasets\multi\time_anotation\result\csv_ver\test_laughter.csv", 'w',  newline='') as f:
    writer = csv.writer(f)
    for laughter_file in glob.iglob(osp.join(jsondir, f"{test_prefix+json_prefix}*.json")):
        with open(laughter_file, "r") as f:
            laughter = json.load(f)
        basename = osp.splitext(osp.basename(laughter_file))[0]
        if basename.startswith(test_prefix):
            basename = basename[len(test_prefix):]
        audio_path = osp.join(basedir, "audio", basename+".ogg")
        for k, v in laughter.items():
            if v["prob"] > .8:
                # 笑い出ない部分を学習済みモデルが笑いと判定していたら、その部分を笑いでないとして学習させる
                if ("not_a_laugh" in v) and v["not_a_laugh"]:
                    if v["end_sec"] - v["start_sec"] < 1:
                        continue
                    writer.writerow([
                        v["start_sec"],
                        v["end_sec"] - v["start_sec"],
                        v["start_sec"],
                        v["end_sec"] - v["start_sec"],
                        audio_path,
                        0,
                    ])
                else:
                    if v["start_sec"] >= 1.: # 時間が1秒以上でないとデータがおかしなる
                        writer.writerow([
                            v["start_sec"]-3 if v["start_sec"]-3 >= 0 else 0,
                            3 if v["start_sec"] > 3 else v["start_sec"],
                            v["start_sec"]-3 if v["start_sec"]-3 >= 0 else 0,
                            3 if v["start_sec"] > 3 else v["start_sec"],
                            audio_path,
                            0,
                        ])
                    if v["end_sec"] - v["start_sec"] >= 1.: # 時間が1秒以上でないとデータがおかしなる
                        writer.writerow([
                            v["start_sec"],
                            v["end_sec"] - v["start_sec"],
                            v["start_sec"],
                            v["end_sec"] - v["start_sec"],
                            audio_path,
                            1,
                        ])

