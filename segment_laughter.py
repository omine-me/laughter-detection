# Example usage:
# python segment_laughter.py --input_audio_file=tst_wave.wav --output_dir=./tst_wave --save_to_textgrid=False --save_to_audio_files=True --min_length=0.2 --threshold=0.5

import os, sys, pickle, time, librosa, argparse, torch, numpy as np, pandas as pd, scipy
import json
from tqdm import tqdm
import tgt
sys.path.append('./laughter_detection/utils/')
from . import laugh_segmenter
from . import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils
from tqdm import tqdm
from torch import optim, nn
from functools import partial
from distutils.util import strtobool
# from accelerate import Accelerator

model = None
feature_fn = None
device = None

def segment_laughter(input_audio_file="",
                     output_dir="",
                     threshold_high="0.5",
                     threshold_low="0.15",
                     min_length="0.2",
                     min_speaking_length="5.0",
                     save_to_audio_files="True",
                     offset=0.0,
                     duration=None,
                     caution_log=None,
                     is_last=True,
                     split_minutes=0,
                     device=None):
    global model, feature_fn#, device
    sample_rate = 8000

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='./laughter-detection/checkpoints/in_use/resnet_with_augmentation')
    parser.add_argument('--config', type=str, default='resnet_with_augmentation')
    parser.add_argument('--threshold_high', type=str, default=threshold_high)
    parser.add_argument('--threshold_low', type=str, default=threshold_low)
    parser.add_argument('--min_speaking_length', type=str, default=min_speaking_length)
    parser.add_argument('--min_length', type=str, default=min_length)
    parser.add_argument('--input_audio_file', type=str, default=input_audio_file)
    parser.add_argument('--output_dir', type=str, default=output_dir)
    parser.add_argument('--save_to_audio_files', type=str, default=save_to_audio_files)
    parser.add_argument('--save_to_textgrid', type=str, default='True')

    args = parser.parse_args()

    model_path = args.model_path
    config = configs.CONFIG_MAP[args.config]
    audio_path = args.input_audio_file
    threshold_high = float(args.threshold_high)
    threshold_low = float(args.threshold_low)
    min_length = float(args.min_length)
    save_to_audio_files = bool(strtobool(args.save_to_audio_files))
    save_to_textgrid = bool(strtobool(args.save_to_textgrid))
    output_dir = args.output_dir

    # accelerator = Accelerator()
    # device = accelerator.device

    if not model:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device {device}")

        ##### Load the Model

        model = config['model'](dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
        feature_fn = config['feature_fn']
        model.set_device(device)

        if os.path.exists(model_path):
            torch_utils.load_checkpoint(model_path+'/best.pth.tar', model)
            model.eval()
        else:
            raise Exception(f"Model checkpoint not found at {model_path}")
        
        # model = accelerator.prepare(model)
    
    ##### Load the audio file and features
        
    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=audio_path, feature_fn=feature_fn, sr=sample_rate, offset=offset, duration=duration)

    collate_fn=partial(audio_utils.pad_sequences_with_labels,
                            expand_channel_dim=config['expand_channel_dim'])

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset, num_workers=6, batch_size=20, shuffle=False, collate_fn=collate_fn)
    
    # inference_generator = accelerator.prepare(inference_generator)

    ##### Make Predictions

    probs = []
    for model_inputs, _ in tqdm(inference_generator):
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        # print(x.shape, preds.shape)
        if len(preds.shape)==0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    file_length = duration # return sec
    print("1",file_length)
    # if file_length - offset < duration:
    #     # file_length = file_length/60. - offset
    #     file_length = file_length - offset
    # print("2",file_length)
    fps = len(probs)/float(file_length)
    # print("fps",fps)

    probs = laugh_segmenter.lowpass(probs)
    instances = laugh_segmenter.get_laughter_instances(probs,
                                                       thres_high=threshold_high,
                                                       thres_low=threshold_low,
                                                       min_laughter_len=float(args.min_length),
                                                       min_speaking_len=float(args.min_speaking_length),
                                                       fps=fps)

    print("found %d laughs." % (len (instances)))

    if len(instances) > 0:
        if save_to_audio_files:
            full_res_y, full_res_sr = librosa.load(audio_path,sr=44100)
            wav_paths = []
            maxv = np.iinfo(np.int16).max

            if output_dir is None:
                raise Exception("Need to specify an output directory to save audio files")
            else:
                # os.system(f"mkdir -p {output_dir}")
                for index, instance in enumerate(instances):
                    laughs = laugh_segmenter.cut_laughter_segments([instance],full_res_y,full_res_sr)
                    wav_path = output_dir + "/laugh_" + str(index) + ".wav"
                    scipy.io.wavfile.write(wav_path, full_res_sr, (laughs * maxv).astype(np.int16))
                    wav_paths.append(wav_path)
                print(laugh_segmenter.format_outputs(instances, wav_paths))
        
        if save_to_textgrid:
            out_path = output_dir+'.json'

            if os.path.exists(out_path):
                with open(out_path, "r") as f:
                    laughter_attributes = json.load(f)
                start = len(laughter_attributes)
                # print(instances[0][0], offset, str(len(laughter_attributes)-1), laughter_attributes[str(len(laughter_attributes)-1)]["end_sec"])
                assert instances[0][0]+offset >= laughter_attributes[str(len(laughter_attributes)-1)]["end_sec"], "Previous prediction seems to exist."
            else:
                laughter_attributes = {}
                start = 0
            
            for idx, inst in enumerate(instances, start):
                laughter_attributes[idx] = {"start_sec": offset+inst[0],
                                            "end_sec": offset+inst[1],
                                            "prob": inst[2]}
            if not os.path.exists(os.path.abspath(os.path.join(output_dir, os.pardir))):
                os.makedirs(os.path.abspath(os.path.join(output_dir, os.pardir)))
            with open(out_path, mode='w', encoding="utf-8") as f:
                json.dump(laughter_attributes, f)
            
            # log caution when laughter location is too near start or near end
            # because it may be a separated laughter.
            if caution_log:
                if (instances[0][0] < min_length and offset!=0.) or\
                    instances[-1][1] > file_length-min_length:
                    with open(caution_log, mode='a') as f:
                        f.write(out_path+"\n")
    
    if is_last and save_to_textgrid:
        if os.path.exists(out_path):
            concat_splitted_segment(out_path, split_minutes, min_length, (threshold_high+threshold_low)/2.)

def concat_splitted_segment(laughter_path, split_minutes, range_sec, laugh_normal_thre):
    with open(laughter_path, "r") as f:
        laughter = json.load(f)
    
    split_sec = split_minutes*60.
    range_s = split_sec - range_sec
    range_e = split_sec + range_sec
    laugh_count = len(laughter)
    for idx in range(laugh_count-1):
        n_idx=str(idx+1)
        idx=str(idx)
        if not idx in laughter:
            continue
        if split_sec <= laughter[idx]["start_sec"]:
            print(idx)
            split_sec = ((laughter[idx]["start_sec"]//(split_minutes*60.))+1)*split_minutes*60.
            range_s = split_sec - range_sec
            range_e = split_sec + range_sec
        if (range_s <= laughter[idx]["end_sec"] <= split_sec) and (split_sec <= laughter[n_idx]["start_sec"] <= range_e):
            print(idx)
            if (laughter[idx]["prob"] < laugh_normal_thre and laughter[n_idx]["prob"] < laugh_normal_thre) or \
                (laughter[idx]["prob"] > laugh_normal_thre and laughter[n_idx]["prob"] > laugh_normal_thre):
                print(" "+idx)
                laughter[idx]["end_sec"] = laughter[n_idx]["end_sec"]
                laughter[idx]["prob"] = (laughter[idx]["prob"] + laughter[n_idx]["prob"])/2.
                del laughter[n_idx]
    # return laughter
    laughter_new = {}
    for idx, inst in enumerate(laughter.values()):
        laughter_new[idx] = {**inst}
    # return laughter_new
    with open(laughter_path, mode='w', encoding="utf-8") as f:
        json.dump(laughter_new, f)

if __name__ == '__main__':
    segment_laughter()
