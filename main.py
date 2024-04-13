import argparse
import os
import time

import torch
from better_profanity import profanity
from moviepy.editor import AudioFileClip, VideoFileClip
from pydub import AudioSegment
from pydub.generators import Sine
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

os.makedirs("tmp", exist_ok=True)


def extract_audio(input_video_path: str, output_dir: str, **kwargs) -> str:
    """
    Extract audio from video file

    Args:
    input_video_path (str): Path to the video file
    output_dir (str): Directory to save the extracted audio file
    """
    video = VideoFileClip(input_video_path)
    video_name = os.path.basename(input_video_path).split(".")[0]
    output_audio_path = os.path.join(output_dir, f"{video_name}.mp3")
    video.audio.write_audiofile(output_audio_path, codec="mp3")
    video.close()

    return output_audio_path


def transcribe_audio(input_audio_path: str, **kwargs) -> dict:
    """
    Transcribe audio file

    Args:
    input_audio_path (str): Path to the audio file
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    transcription_model_id = "openai/whisper-large-v3"
    transcription_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        transcription_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    transcription_model.to(device)
    processor = AutoProcessor.from_pretrained(transcription_model_id)

    transcription_pipeline = pipeline(
        "automatic-speech-recognition",
        model=transcription_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = transcription_pipeline(input_audio_path, return_timestamps="word")

    return result


def censor_transcription(transcription: dict, **kwargs) -> dict:
    """
    Censor the transcription

    Args:
    transcription (dict): Transcription of the audio file
    """

    for chunk in transcription["chunks"]:
        chunk["censored"] = (
            True if chunk["text"] != profanity.censor(chunk["text"]) else False
        )

    return transcription


def censor_audio(
    extracted_audio_path: str, censored_transcription: dict, **kwargs
) -> str:
    """
    Censor the audio file

    Args:
    extracted_audio_path (str): Path to the extracted audio
    censored_transcription (dict): Censored transcription of the audio file
    """

    # read the audio file
    audio = AudioSegment.from_file(extracted_audio_path)
    muted_sections = []

    for chunk in censored_transcription["chunks"]:
        if chunk["censored"]:
            muted_sections.append(chunk["timestamp"])

    beep = Sine(1000).to_audio_segment(duration=1000).apply_gain(-20)
    for start, end in muted_sections:
        start_ms, end_ms = start * 1000, end * 1000
        beep_segment = beep[: end_ms - start_ms]
        audio = audio[:start_ms] + beep_segment + audio[end_ms:]

    censored_audio_path = extracted_audio_path.replace(".mp3", "_censored.mp3")
    audio.export(censored_audio_path, format="mp3")

    return censored_audio_path


def attach_censored_audio_with_video(
    censored_audio_path: str, input_video_path: str, **kwargs
) -> None:
    """
    Attach censored audio with original video

    Args:
    censored_audio_path (str): Path to the censored audio
    input_video_path (str): Path to the original input video
    """

    video = VideoFileClip(input_video_path)
    output_video_path = input_video_path.replace(".mp4", "_censored.mp4")
    target_audio = AudioFileClip(censored_audio_path)
    output_video = video.set_audio(target_audio)
    output_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    video.close()
    target_audio.close()
    output_video.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, help="path to input file")
    args = parser.parse_args()

    extracted_audio_path = extract_audio(args.input_video, "tmp")
    audio_transcription = transcribe_audio(extracted_audio_path)
    censored_transcription = censor_transcription(audio_transcription)
    censored_audio_path = censor_audio(extracted_audio_path, censored_transcription)
    output_video_path = attach_censored_audio_with_video(
        censored_audio_path, args.input_video
    )

    print("Done!")
