# check_adv.py
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
w2v2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
hubert = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device)

save_dir = "C:/Projects/bitblind/asr/attacks/pgd_untargeted_conformer/hubert-large-960h-wav2vec2-large-960h-lv60-self-conformer-mini-101/1002/save"

for name in ["obama_test_video_short_input_nat.wav", "obama_test_video_short_input_adv.wav"]:
    audio, sr = sf.read(f"{save_dir}/{name}")
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        t1 = processor.batch_decode(torch.argmax(w2v2(inputs).logits, dim=-1))[0]
        t2 = processor.batch_decode(torch.argmax(hubert(inputs).logits, dim=-1))[0]
    print(f"\n{name}")
    print(f"  Wav2Vec2: {t1}")
    print(f"  HuBERT:   {t2}")