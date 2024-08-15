# Raine
AI Vtuber shenanigans (work-in-progress)

## Vtube Studio API Connection:
![Screenshot 2024-08-15 094513](https://github.com/user-attachments/assets/a8b1ff19-a44e-4852-b90e-f474558b13c3)

Currently using `pyvts` to interface with Vtube Studio, still working on this though

## Text-To-Speech
https://github.com/user-attachments/assets/e5a235ef-95db-4675-b539-805f2ccbcaef

Using GPT-SoVITS for tts right now, but I might change to Parler TTS or CosyVoice later on.

### TTS Usage Instructions

Simply git clone this repo, cd into the tts folder and run `pip install -r requirements.txt`. An example is given below:

```python
from tts import TTS

tts_model = TTS(
    "gpt_sovits/GPT_weights/gpt-weights-file.ckpt",
    "gpt_sovits/SoVITS_weights/sovits-weights-file.pth",
    {"neutral": "tts/ref_audio/..."}
)

for chunk in tts_model.infer("hello world", "neutral"):
    tts_model.play(chunk)
```

## LLM
Using Exllamav2 for testing currently, but I plan to switch to using OctoAI later.
