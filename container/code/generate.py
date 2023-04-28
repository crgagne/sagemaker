import os
import argparse
import json
import sys

print(os.environ['HOME'])
print(sys.path)

from bark import SAMPLE_RATE, generate_audio, preload_models


def main(args):
     print(args)

     print('loading models')
     preload_models()

     print('generating audio')
     text_prompt = "Hello, my name is Suno"
     audio_array = generate_audio(text_prompt)
     print(audio_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--train",
            type=int,
            default=64,
        )

    # Data and model checkpoints directories
    parser.add_argument(
        "--test",
        type=int,
        default=64,
    )
    args, unknown = parser.parse_known_args()
    main(args)