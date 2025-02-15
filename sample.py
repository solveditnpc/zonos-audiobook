import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

# Use the hybrid with "Zyphra/Zonos-v0.1-hybrid"
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")

wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
speaker = model.make_speaker_embedding(wav, sampling_rate)

torch.manual_seed(421)

cond_dict = make_cond_dict(text="Hello, world!", speaker=speaker, language="en-us")
conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
