from glob import glob
import os
from model import Generator
from signal_processing import get_spec_and_phase, transform_spec_to_wav
import argparse
import torch
import torchaudio
from tqdm import tqdm
import soundfile as sf

SAMPLE_RATE = 16000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weight", type=str, default="./1_PE_CS_Table2.pth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--causal", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    print(args)

    G = Generator(causal=args.causal).to(args.device)
    checkpoint = torch.load(args.weight, map_location=args.device)
    G.load_state_dict(checkpoint['generator'], strict=True)
    G.eval()
    print(f"Total parameters:{sum(p.numel() for p in G.parameters())/10**6:.3f}M")

    input_files = glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True)
    print(f"Total input files: {len(input_files)}")

    for input_file in tqdm(input_files):
        with torch.no_grad():
            noise_wav, _ = torchaudio.load(input_file)

            noise_mag, noise_phase = get_spec_and_phase(noise_wav.to(args.device))
            assert noise_mag.size(2) == 257, 'eval'
            assert noise_phase.size(2) == 257, 'eval'

            mask = G(noise_mag)
            mask = mask.clamp(min=0.05)

            enh_mag = torch.mul(mask, noise_mag)
            enh_wav = transform_spec_to_wav(torch.expm1(enh_mag), noise_phase, signal_length=noise_wav.size(1)).detach().cpu().numpy().squeeze()

            output_file = os.path.join(args.output_dir, os.path.relpath(input_file, args.input_dir))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            sf.write(output_file, enh_wav, SAMPLE_RATE)
