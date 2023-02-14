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
model = None
feature_fn = None
device = None

def segment_laughter(input_audio_file="", output_dir="", threshold="0.5", min_length="0.2", save_to_audio_files="True", offset=0.0, duration=None, caution_log=None):
    global model, feature_fn, device
    sample_rate = 8000

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='./laughter_detection/checkpoints/in_use/resnet_with_augmentation')
    parser.add_argument('--config', type=str, default='resnet_with_augmentation')
    parser.add_argument('--threshold', type=str, default=threshold)
    parser.add_argument('--min_length', type=str, default=min_length)
    parser.add_argument('--input_audio_file', type=str, default=input_audio_file)
    parser.add_argument('--output_dir', type=str, default=output_dir)
    parser.add_argument('--save_to_audio_files', type=str, default=save_to_audio_files)
    parser.add_argument('--save_to_textgrid', type=str, default='True')

    args = parser.parse_args()

    model_path = args.model_path
    config = configs.CONFIG_MAP[args.config]
    audio_path = args.input_audio_file
    threshold = float(args.threshold)
    min_length = float(args.min_length)
    save_to_audio_files = bool(strtobool(args.save_to_audio_files))
    save_to_textgrid = bool(strtobool(args.save_to_textgrid))
    output_dir = args.output_dir

    if not model:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    ##### Load the audio file and features
        
    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=audio_path, feature_fn=feature_fn, sr=sample_rate, offset=offset, duration=duration)

    collate_fn=partial(audio_utils.pad_sequences_with_labels,
                            expand_channel_dim=config['expand_channel_dim'])

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset, num_workers=4, batch_size=8, shuffle=False, collate_fn=collate_fn)


    ##### Make Predictions

    probs = []
    for model_inputs, _ in tqdm(inference_generator):
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        print(x.shape, preds.shape)
        if len(preds.shape)==0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    file_length = audio_utils.get_audio_length(audio_path)
    print(probs.shape)
    print("1",file_length)
    if file_length/60. - offset < duration:
        # file_length = file_length/60. - offset
        file_length = file_length - offset*60.
    else:
        # file_length = duration
        file_length = duration*60.
    print("2",file_length)
    fps = len(probs)/float(file_length)
    print("fps",fps)

    probs = laugh_segmenter.lowpass(probs)
    instances = laugh_segmenter.get_laughter_instances(probs, threshold=threshold, min_length=float(args.min_length), fps=fps)

    print(); print("found %d laughs." % (len (instances)))

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
            out_path = output_dir+'_laughter.json'

            if os.path.exists(out_path):
                with open(out_path, "r") as f:
                    laughter_attributes = json.load(f)
                start = len(laughter_attributes)
                assert instances[0][0] >= laughter_attributes[str(len(laughter_attributes)-1)]["end_sec"], "Previous prediction seems to exist."
            else:
                laughter_attributes = {}
                start = 0
            
            for idx, inst in enumerate(instances, start):
                laughter_attributes[idx] = {"start_sec": offset*60+inst[0],
                                            "end_sec": offset*60+inst[1],
                                            "prob": inst[2]}
            if not os.path.exists(os.path.abspath(os.path.join(output_dir, os.pardir))):
                os.makedirs(os.path.abspath(os.path.join(output_dir, os.pardir)))
            with open(out_path, mode='w', encoding="utf-8") as f:
                json.dump(laughter_attributes, f)
            
            # log caution when laughter location is too near start or near end
            # because it may be a separated laughter.
            if caution_log:
                if instances[0][0] < min_length or\
                    instances[-1][1] > file_length-min_length:
                    with open(caution_log, mode='a') as f:
                        f.write(input_audio_file)

if __name__ == '__main__':
    segment_laughter()