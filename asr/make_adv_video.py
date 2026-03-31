# make_adv_video.py
import sys
from moviepy.editor import VideoFileClip, AudioFileClip

input_video = sys.argv[1] if len(sys.argv) > 1 else "obama_test_video_short.mp4"
adv_audio   = sys.argv[2] if len(sys.argv) > 2 else "C:/Projects/bitblind/asr/attacks/pgd_untargeted_conformer/hubert-large-960h-wav2vec2-large-960h-lv60-self-conformer-mini-101/1002/save/obama_test_video_short_input_adv.wav"
output_video = sys.argv[3] if len(sys.argv) > 3 else "output_adv.mp4"

print(f"Input video: {input_video}")
print(f"Adv audio:   {adv_audio}")
print(f"Output:      {output_video}")

video = VideoFileClip(input_video)
audio = AudioFileClip(adv_audio)

# Trim video to match audio length
audio_duration = audio.duration
video = video.subclip(0, min(audio_duration, video.duration))

output = video.set_audio(audio)
output.write_videofile(output_video, codec="libx264", audio_codec="aac")

video.close()
audio.close()

print(f"\nDone! Saved to: {output_video}")