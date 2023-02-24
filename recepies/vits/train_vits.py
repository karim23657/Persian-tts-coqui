import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig , CharactersConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.downloaders import download_thorsten_de

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    formatter="mozilla", meta_file_train="metadata.csv", path="/kaggle/input/persian-tts-dataset-famale" 
)



audio_config = BaseAudioConfig(
    sample_rate=24000,
    do_trim_silence=True,
    resample=False,
    mel_fmin=0,
    mel_fmax=None 
)
character_config=CharactersConfig(
  characters='ءابتثجحخدذرزسشصضطظعغفقلمنهويِپچژکگیآأؤإئًَُّ',
  punctuations='!(),-.:;? ̠،؛؟‌<>',
  phonemes='ˈˌːˑpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟaegiouwyɪʊ̩æɑɔəɚɛɝɨ̃ʉʌʍ0123456789"#$%*+/=ABCDEFGHIJKLMNOPRSTUVWXYZ[]^_{}',
  pad="<PAD>",
  eos="<EOS>",
  bos="<BOS>",
  blank="<BLNK>",
  characters_class="TTS.tts.utils.text.characters.IPAPhonemes",
  )
config = VitsConfig(
    audio=audio_config,
    run_name="vits_fa_female",
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=0,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    save_step=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=True,
    phoneme_language="fa",
    characters=character_config,
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    test_sentences=[
        ["سلطان محمود در زمستانی سخت به طلخک گفت که: با این جامه ی یک لا در این سرما چه می کنی "],
        ["مردی نزد بقالی آمد و گفت پیاز هم ده تا دهان بدان خو شبوی سازم."],
        ["از مال خود پاره ای گوشت بستان و زیره بایی معطّر بساز"],
        ["یک بار هم از جهنم بگویید."],
        ["یکی اسبی به عاریت خواست"]
    ],
    output_path=output_path,
    datasets=[dataset_config],
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()