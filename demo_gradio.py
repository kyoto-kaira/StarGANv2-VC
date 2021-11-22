# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import audiofile
import shutil
import os

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from models import Generator, StyleEncoder
from parallel_wavegan.utils import load_model

import gradio as gr



F0_model = None
vocoder = None
starganv2 = None


def numpy2tensor(wave):
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mean, std = -4, 4

    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     style_encoder=style_encoder)

    return nets_ema


def compute_style(reference_path):
    wave, sr = librosa.load(reference_path, sr=24000)
    if sr != 24000:
        wave = librosa.resample(wave, sr, 24000)
    mel_tensor = numpy2tensor(wave).to('cuda')

    with torch.no_grad():
        ref = starganv2.style_encoder(mel_tensor.unsqueeze(1))

    return ref


def inference(source_path, reference_path):
    # load source wave
    audio, source_sr = librosa.load(source_path, sr=24000)
    audio /= np.max(np.abs(audio))
    source = numpy2tensor(audio).to('cuda')

    # load reference wave
    reference = compute_style(reference_path)

    with torch.no_grad():
        f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
        out = starganv2.generator(source.unsqueeze(1), reference, F0=f0_feat)
        mel = out.transpose(-1, -2).squeeze()
        converted = vocoder.inference(mel)
        converted = converted.view(-1).to('cpu').detach().numpy()

    return converted


def main(src1, src2, ref1, ref2):
    if src1:
        source = src1
    elif src2:
        source = src2
    else:
        return "ERROR : [SOURCE]を入力してください", None

    if ref1:
        reference = ref1
    elif ref2:
        reference = ref2
    else:
        return "ERROR : [REFERENCE]を入力してください", None

    source_name = os.path.basename(source)
    source_name, ext = os.path.splitext(source_name)
    if ext != ".wav":
        return "ERROR : [SOURCE]がwavファイルではありません", None

    reference_name = os.path.basename(reference)
    reference_name, ext = os.path.splitext(reference_name)
    if ext != ".wav":
        return "ERROR : [REFERENCE]がwavファイルではありません", None
            

    converted = inference(source, reference)
    
    result_name = source_name + "_" + reference_name + ".wav"
    audiofile.write(f"/tmp/{result_name}", converted, 24000)

    return "SUCCESS : 変換が完了しました", f"/tmp/{result_name}"




if __name__ == "__main__":
    log_dir = "Models/JVS100_integration"
    
    # load F0 model
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load("Utils/JDC/bst.t7")['net']
    F0_model.load_state_dict(params)
    F0_model.eval().to('cuda')


    # load vocoder
    vocoder = load_model("Vocoder/checkpoint-400000steps.pkl").to('cuda').eval()
    vocoder.remove_weight_norm()
    vocoder.eval()


    # load starganv2
    with open(os.path.join(log_dir, 'config.yml')) as f:
        starganv2_config = yaml.safe_load(f)
    
    model_path = os.path.join(log_dir, f'epoch_{starganv2_config["epochs"]:05}.pth')

    starganv2 = build_model(model_params=starganv2_config["model_params"])
    params = torch.load(model_path, map_location='cpu')['model_ema']

    for key in starganv2:
        starganv2[key].load_state_dict(params[key])
        starganv2[key].eval()

    starganv2.style_encoder = starganv2.style_encoder.to('cuda')
    starganv2.generator = starganv2.generator.to('cuda')


    num = np.random.permutation(13) + 1
    idx = np.random.permutation(100) + 1
    examples = [[None, None, None, None] for i in range(5)]
    for i in range(10):
        path = f"./Data/jvs_ver1/jvs{idx[i]:03}/parallel100/wav24kHz16bit/VOICEACTRESS100_{num[i]:03}.wav"
        shutil.copy(path, f"sample/sample{i+1:02}.wav")
        
        examples[i//2][2*(i%2)] = f"sample/sample{i+1:02}.wav"
        examples[i//2][2*(i%2)+1] = f"sample/sample{i+1:02}.wav"


    input_src1 = gr.inputs.Audio(source="upload", type="filepath", label="[SOURCE]ファイルをアップロード", optional=True)
    input_src2 = gr.inputs.Audio(source="microphone", type="filepath", label="[SOURCE]マイクから音声を入力", optional=True)

    input_ref1 = gr.inputs.Audio(source="upload", type="filepath", label="[REFERENCE]ファイルをアップロード", optional=True)
    input_ref2 = gr.inputs.Audio(source="microphone", type="filepath", label="[REFERENCE]マイクから音声を入力", optional=True)

    output_log = gr.outputs.Textbox(type="str", label="ログ")
    output_audio = gr.outputs.Audio(type="file", label="変換結果")

    with open("demo_readme.md", "r") as f:
        readme = f.read()

    iface = gr.Interface(fn=main, 
                            inputs=[input_src1, input_src2, input_ref1, input_ref2],
                            outputs=[output_log, output_audio],
                            title="[KaiRA]声質変換AIデモプログラム",
                            theme="grass",
                            verbose=True,
                            examples=examples,
                            article=readme,
                            allow_flagging=False,
                            allow_screenshot=False)
    iface.launch(share=True)