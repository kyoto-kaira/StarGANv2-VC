#coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

from munch import Munch
from tqdm import tqdm
import numpy as np
import argparse
import librosa
import yaml
import os


from losses import log_norm, f0_loss
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from transforms import build_transforms
from models import Generator, MappingNetwork, StyleEncoder

from apex import amp


MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, sr=24000):
        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [path for path, label in _data_list]

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.max_mel_length = 192

    def __len__(self):
        return len(self.data_list)**2

    def __getitem__(self, idx):
        idx1 = idx // len(self.data_list)
        idx2 = idx  % len(self.data_list)

        data = self.data_list[idx1]
        mel_tensor = self._load_data(data)

        ref_data = self.data_list[idx2]
        ref_mel_tensor = self._load_data(ref_data)

        return mel_tensor.unsqueeze(0), ref_mel_tensor.unsqueeze(0)
    
    def _load_data(self, path):
        wave_tensor = self._load_tensor(path)
    
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor

    def _preprocess(self, wave_tensor, ):
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def _load_tensor(self, data):
        wave_path = data
        wave, sr = librosa.load(wave_path, sr=self.sr)
        if sr != self.sr:
            wave = librosa.resample(wave, sr, self.sr)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor


def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema


def calc_loss(nets, x_real, s_trg, x_fake):
    s_fake = nets.style_encoder(x_fake)

    # compute ASR/F0 features (real)
    F0_real, GAN_F0_real, cyc_F0_real = nets.f0_model(x_real)
    ASR_real = nets.asr_model.get_feature(x_real)

    # compute ASR/F0 features (fake)
    F0_fake, GAN_F0_fake, _ = nets.f0_model(x_fake)
    ASR_fake = nets.asr_model.get_feature(x_fake)
    
    # norm consistency loss
    x_fake_norm = log_norm(x_fake)
    x_real_norm = log_norm(x_real)
    norm_bias = 0.5
    loss_norm = ((torch.nn.ReLU()(torch.abs(x_fake_norm - x_real_norm) - norm_bias))**2).mean()
    
    # F0 loss
    loss_f0 = f0_loss(F0_fake, F0_real)

    # ASR loss
    loss_asr = F.smooth_l1_loss(ASR_fake, ASR_real)
    
    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))
    
    return loss_norm.item(), loss_f0.item(), loss_asr.item(), loss_sty.item()


def get_nets(args):
    with open(os.path.join(args.log_dir, 'config.yml')) as f:
        config = yaml.safe_load(f)
    

    # load ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    with open(ASR_config) as f:
            ASR_config = yaml.safe_load(f)
    ASR_model_config = ASR_config['model_params']
    ASR_model = ASRCNN(**ASR_model_config)
    params = torch.load(ASR_path, map_location='cpu')['model']
    ASR_model.load_state_dict(params)
    ASR_model.eval().to('cuda')
    

    # load F0 model
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load("Utils/JDC/bst.t7")['net']
    F0_model.load_state_dict(params)
    F0_model.eval().to('cuda')
    

    # load starganv2
    model_path = os.path.join(args.log_dir, f'epoch_{config["epochs"]:05}.pth')

    starganv2 = build_model(model_params=config["model_params"])
    params = torch.load(model_path, map_location='cpu')['model_ema']

    for key in starganv2:
        starganv2[key].load_state_dict(params[key])
        starganv2[key].eval()

    starganv2.style_encoder = starganv2.style_encoder.to('cuda')
    starganv2.mapping_network = starganv2.mapping_network.to('cuda')
    starganv2.generator = starganv2.generator.to('cuda')


    # set AMP
    opt_params = list(F0_model.parameters()) + \
                    list(starganv2.style_encoder.parameters()) + \
                    list(starganv2.mapping_network.parameters()) + \
                    list(starganv2.generator.parameters()) + \
                    list(ASR_model.parameters())
    optimizer = torch.optim.Adam(opt_params)

    [F0_model, starganv2.style_encoder, starganv2.mapping_network, starganv2.generator, ASR_model], optimizer = \
            amp.initialize([F0_model, starganv2.style_encoder, starganv2.mapping_network, starganv2.generator, ASR_model], optimizer, opt_level="O1")
    

    # set models in nets
    nets = Munch(f0_model=F0_model,
                 style_encoder=starganv2.style_encoder,
                 mapping_network=starganv2.mapping_network,
                 generator=starganv2.generator,
                 asr_model=ASR_model
            )

    return nets


def main(args):
    nets = get_nets(args)
    
    # set DataLoader
    with open("Data/test_list.txt", "r") as f:
        test_list = f.readlines()
    
    dataset = TestDataset(test_list)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch,
                                              shuffle=False,
                                              drop_last=True,
                                              num_workers=4)

    losses = {"norm":0, "f0":0, "asr":0, "sty":0}

    iter = 0

    for x_src, x_ref in tqdm(data_loader):
        with torch.no_grad():
            x_src = x_src.to('cuda')
            x_ref = x_ref.to('cuda')

            s_ref = nets.style_encoder(x_ref)
            f0_feat = nets.f0_model.get_feature_GAN(x_src)
            x_converted = nets.generator(x_src, s_ref, F0=f0_feat)

            loss_norm, loss_f0, loss_asr, loss_sty = calc_loss(nets, x_src, s_ref, x_converted)
            losses["norm"] += loss_norm
            losses["f0"] += loss_f0
            losses["asr"] += loss_asr
            losses["sty"] += loss_sty

            iter += 1
            if iter == 1000:
                break

    loss_total = 0
    losses_lambda = {"norm":1, "f0":5, "asr":10, "sty":1}

    for k in losses.keys():
        losses[k] /= iter #len(data_loader)
        loss_total += losses_lambda[k] * losses[k]

    print("Model Name    ", args.log_dir)
    print(f"loss_norm  = {losses['norm']:.05}")
    print(f"loss_f0    = {losses['f0']:.05}")
    print(f"loss_asr   = {losses['asr']:.05}")
    print(f"loss_sty   = {losses['sty']:.05}")
    print(f"loss_total = {loss_total:.05}")



if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="Models/JVS10_normal", 
                        help="where models were saved")
    parser.add_argument("--batch", type=int, default=4, 
                        help="batch size")

    args = parser.parse_args()

    main(args)
