# Persian-tts-coqui [![Hugging Face Spaces](https://cdn.statically.io/gh/karim23657/blogmaterials/main/assets/hf.svg)](https://huggingface.co/spaces/Kamtera/Persian-tts-CoquiTTS)
Persian/Farsi text to speech(TTS) training using coqui tts (Online demo : [![Hugging Face Spaces](https://cdn.statically.io/gh/karim23657/blogmaterials/main/assets/hf.svg)](https://huggingface.co/spaces/Kamtera/Persian-tts-CoquiTTS))<br>
This repository contains sample codes for training text to speech models <br>
Feel free to ask your questions [issues](https://github.com/karim23657/Persian-tts-coqui/issues)
<div dir="rtl">
این مخزن شامل نمونه کدهای لازم برای آموزش مدل های متن به صوت فارسی است
سوالاتتان را در <a href="https://github.com/karim23657/Persian-tts-coqui/issues">issues</a> مطرح کنید 
</div>


# How to train ?
Sample codes and notebooks are available at [recepies](https://github.com/karim23657/Persian-tts-coqui/tree/main/recepies) folder
<div dir="rtl">
نمونه کد ها و نوت بوک ها در پوشه <a href="https://github.com/karim23657/Persian-tts-coqui/tree/main/recepies">recepies</a> موجود هستند
</div>

# Pretrained models
These are models you can use to test or finetune<br>
<div dir="rtl">
مدل هایی که می توانید برای امتحان کردن یا finetune کردن از آنها استفاده کنید
</div>

|Model|Dataset|
|----|------|
|[vits female (best)](https://huggingface.co/Kamtera/persian-tts-female-vits)|[persian-tts-dataset-famale](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale)|
|[vits male1 (best)](https://huggingface.co/Kamtera/persian-tts-male1-vits)|[persian-tts-dataset-male](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-male)|
|[vits female1](https://huggingface.co/Kamtera/persian-tts-female1-vits)|[ParsiGoo](https://github.com/karim23657/ParsiGoo)|
|[vits male](https://huggingface.co/Kamtera/persian-tts-male-vits)|[persian-tts-dataset](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset)|
|[glowtts female](https://huggingface.co/Kamtera/persian-tts-female-glow_tts)|[persian-tts-dataset-famale](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale)|
|[glowtts male](https://huggingface.co/Kamtera/persian-tts-male-glow_tts)|[persian-tts-dataset](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset)|
|[tacotron2 female](https://huggingface.co/Kamtera/persian-tts-female-tacotron2)|[persian-tts-dataset-famale](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale)|
|[Hifigan](https://huggingface.co/Kamtera/persian-tts-female-Hifigan)|[persian-tts-dataset-famale](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale)|
|[Wavernn](https://huggingface.co/Kamtera/persian-female-Wavernn)|[persian-tts-dataset-famale](https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale)|

* Share your trained models [here](https://github.com/karim23657/Persian-tts-coqui/issues/1)

- :hugs: huggingface Demo https://huggingface.co/spaces/Kamtera/Persian-tts-CoquiTTS


# Datasets
Models trained on these datasets : 
- https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset
- https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale
- https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-male
- https://github.com/karim23657/ParsiGoo

- If you'v created a dataset or found any good datasets on the web you can share with us [here](https://github.com/karim23657/Persian-tts-coqui/issues/2).

## Test trained model
* install TTS and Espeak
```
pip install -q TTS
sudo apt-get -y install espeak-ng
```
* predict one text from commandline
```
tts --text "شیش سیخ جیگر" --model_path "best_model.ckpt" --config_path "config.json"
```
* From python API
```python
from TTS.api import TTS
tts=TTS(model_path="https://huggingface.co/Kamtera/persian-tts-male1-vits/resolve/main/checkpoint_88000.pth",
        config_path="https://huggingface.co/Kamtera/persian-tts-male1-vits/resolve/main/config.json")
tts.tts_to_file(".زندگی فقط یک بار است؛ از آن به خوبی استفاده کن",file_path='output.wav')
```

* Another way :
```python
from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


model_path ="config.json"  # Absolute path to the model checkpoint.pth
config_path ="best_model.pth" # Absolute path to the model config.json

text=".زندگی فقط یک بار است؛ از آن به خوبی استفاده کن"

synthesizer = Synthesizer(
    model_path, config_path
)
wavs = synthesizer.tts(text)
synthesizer.save_wav(wavs, 'sp.wav')

```


## usefull links 

- https://github.com/coqui-ai/TTS
