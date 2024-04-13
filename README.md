# Profanity Censorship in Videos
This tool detects and censors profanity in videos. It handles everything from video loading, processing, and exporting. The higher level approach is as following:

- Extract audio from video file using `moviepy`.
- Transcribe the audio using OpenAI's `Whisper`.
- Detect and localize chunks with profanity using `better_profanity`.
- Beep out the mutable chunks using `Sine` in `pydub`.
- Export a new video with new audio using `moviepy`.

## Instructions
To use the tool, please use the following instructions:

```sh
# create virtual environment
python3 -m venv venv
source venv/bin/activate

# install requirements
pip install -r requirements.txt

# run the main script with args
python3 main.py --input_video=<path_to_your_video>

# example
python main.py --input_video=./input/input_video.mp4
```

## Example Output
<iframe width="560" height="315" src="https://www.youtube.com/embed/Vl0ZF66hFcE?si=fXRDqT9LCzZDHeBE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Limitations
It only works with a couple of video formats. You are better off using `.mp4` as the input format. It might miss a couple of things but the model is fairly accurate. Since we're using the larget Whisper model to highest accuracy, it may be slow. On an M2 Pro (12-Core), it takes about 6s per 1s of video. It hasn't been tested on CUDA yet.

## Contributions
I won't be working on this actively, but contributions are welcome.