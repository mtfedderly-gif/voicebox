#!/usr/bin/env python3
import argparse
from pathlib import Path

from kokoro_mlx import KokoroTTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local TTS with Kokoro-MLX.")
    parser.add_argument("text", help="Text to synthesize.")
    parser.add_argument("-o", "--output", default="speech.wav", help="Output WAV path.")
    parser.add_argument("-v", "--voice", default="af_heart", help="Voice name.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier.")
    parser.add_argument("--sample-rate", type=int, default=24000, choices=(24000, 48000))
    args = parser.parse_args()

    output = Path(args.output)
    tts = KokoroTTS.from_pretrained()
    tts.save(
        args.text,
        str(output),
        voice=args.voice,
        speed=args.speed,
        sample_rate=args.sample_rate,
    )
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
