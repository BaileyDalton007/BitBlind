# make_csv.py
import sys
import os
import csv
import soundfile as sf
import librosa
import numpy as np

audio_path = sys.argv[1]
output_csv = sys.argv[2]
TARGET_SR  = 16000

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Handle mp4
if audio_path.lower().endswith(".mp4"):
    from moviepy.editor import VideoFileClip
    temp_wav = audio_path.replace(".mp4", "_input.wav")
    clip = VideoFileClip(audio_path)
    clip.audio.write_audiofile(temp_wav, fps=TARGET_SR, nbytes=2, codec='pcm_s16le', logger=None)
    clip.close()
    audio_path = temp_wav

audio, sr = sf.read(audio_path)
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)
if sr != TARGET_SR:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    sr = TARGET_SR
    sf.write(audio_path, audio, sr)

duration = len(audio) / TARGET_SR
abs_path  = os.path.abspath(audio_path).replace("\\", "/")
filename  = os.path.splitext(os.path.basename(audio_path))[0]

print(f"File: {abs_path}")
print(f"Duration: {duration:.2f}s")

if duration > 14.0:
    print("WARNING: file is longer than 14s, trimming...")
    audio = audio[:TARGET_SR * 14]
    sf.write(audio_path, audio, sr)
    duration = 14.0

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["ID","duration","wav","spk_id","wrd"])
    writer.writeheader()
    writer.writerow({
        "ID":       filename,
        "duration": f"{duration:.2f}",
        "wav":      abs_path,
        "spk_id":   "custom",
        "wrd":      "UNKNOWN"
    })

print(f"CSV saved to: {output_csv}")