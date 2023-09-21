from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch


import numpy as np
from scipy.io.wavfile import write

import wave

model = musicgen.MusicGen.get_pretrained('medium', device='cpu')
model.set_generation_params(duration=15)

prompt = input('Enter the prompt')
res = model.generate([prompt
                      ],
                     progress=True)

audio_data = res.numpy()[0, 0]

sample_rate = 44100

write(f'{prompt}.wav', sample_rate,audio_data)