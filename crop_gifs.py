"""Crop 1/4 off each side of every gif in a folder; save to a new folder."""

import argparse
import os

from PIL import Image, ImageSequence


def crop_gif(in_path, out_path):
    img = Image.open(in_path)
    w, h = img.size
    left, right = w // 4, w - w // 4
    top, bottom = h // 4, h - h // 4

    frames = []
    durations = []
    for frame in ImageSequence.Iterator(img):
        f = frame.convert("RGBA").crop((left, top, right, bottom))
        frames.append(f)
        durations.append(frame.info.get("duration", 33))

    if not frames:
        return False
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=img.info.get("loop", 0),
        disposal=2,
    )
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder containing gifs")
    p.add_argument("--dst", default=None,
                   help="output folder (default: <src>_cropped)")
    args = p.parse_args()

    dst = args.dst or args.src.rstrip("/\\") + "_cropped"
    os.makedirs(dst, exist_ok=True)

    gifs = [f for f in os.listdir(args.src) if f.lower().endswith(".gif")]
    if not gifs:
        print(f"no .gif files in {args.src}")
        return

    for name in sorted(gifs):
        in_path = os.path.join(args.src, name)
        out_path = os.path.join(dst, name)
        try:
            if crop_gif(in_path, out_path):
                print(f"[ok]   {name}")
            else:
                print(f"[skip] {name} (no frames)")
        except Exception as e:
            print(f"[err]  {name}: {e}")

    print(f"\nDone. Cropped gifs in: {dst}")


if __name__ == "__main__":
    main()
