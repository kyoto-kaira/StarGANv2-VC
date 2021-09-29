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
import soundfile
import argparse
import shutil
import os

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from models import Generator, MappingNetwork, StyleEncoder


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
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema


def compute_style(reference_path, speaker_id):
    if reference_path is not None:
        wave, sr = librosa.load(reference_path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            wave = librosa.resample(wave, sr, 24000)
        mel_tensor = numpy2tensor(wave).to('cuda')

        with torch.no_grad():
            # label = torch.LongTensor([speaker_id])
            ref = starganv2.style_encoder(mel_tensor.unsqueeze(1))

    else:
        # label = torch.LongTensor([speaker_id]).to('cuda')
        latent_dim = starganv2.mapping_network.shared[0].in_features
        ref = starganv2.mapping_network(torch.randn(1, latent_dim).to('cuda'))

    return ref


def inference(f0_model, vocoder, starganv2, source_path, reference_path, reference_id):
    # load source wave
    audio, source_sr = librosa.load(source_path, sr=24000)
    audio /= np.max(np.abs(audio))
    source = numpy2tensor(audio).to('cuda')

    # load reference wave
    reference = compute_style(reference_path, reference_id)

    with torch.no_grad():
        f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
        out = starganv2.generator(source.unsqueeze(1), reference, F0=f0_feat)
        
        mel = out.transpose(-1, -2).squeeze().to('cuda')
        converted = vocoder.inference(mel)
        converted = converted.view(-1).to('cpu').detach().numpy()

    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="Models/JVS10_normal", help="where models were saved")
    parser.add_argument("--source", type=str, default=None, help="source audio file path")
    parser.add_argument("--reference", type=str, default=None, help="reference audio file path")
    parser.add_argument("--reference_id", type=int, default=None, help="reference speaker ID")

    args = parser.parse_args()


    # load F0 model
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load("Utils/JDC/bst.t7")['net']
    F0_model.load_state_dict(params)
    F0_model.eval().to('cuda')


    # load vocoder
    from parallel_wavegan.utils import load_model
    vocoder = load_model("Vocoder/checkpoint-400000steps.pkl").to('cuda').eval()
    vocoder.remove_weight_norm()
    vocoder.eval()


    # load starganv2
    with open(os.path.join(args.log_dir, 'config.yml')) as f:
        starganv2_config = yaml.safe_load(f)
    
    model_path = os.path.join(args.log_dir, f'epoch_{starganv2_config["epochs"]:05}.pth')

    starganv2 = build_model(model_params=starganv2_config["model_params"])
    params = torch.load(model_path, map_location='cpu')['model_ema']

    for key in starganv2:
        starganv2[key].load_state_dict(params[key])
        starganv2[key].eval()

    starganv2.style_encoder = starganv2.style_encoder.to('cuda')
    starganv2.mapping_network = starganv2.mapping_network.to('cuda')
    starganv2.generator = starganv2.generator.to('cuda')


    # load reference audio
    if args.reference is None:
        with open('Data/val_list.txt', 'r') as f:
            val_list = f.read().split('\n')[:-1]

        reference_data = random.choice(val_list)
        args.reference, args.reference_id = reference_data.split('|')
        args.reference_id = int(args.reference_id)
    elif args.reference_id is None:
        args.reference_id = random.randint(0, starganv2_config["model_params"]["num_domains"]-1)


    converted = inference(F0_model, vocoder, starganv2, args.source, args.reference, args.reference_id)

    # save results
    # shutil.copy(args.source, "source.wav")
    shutil.copy(args.reference, "reference.wav")
    soundfile.write("converted.wav", converted, 24000)
    