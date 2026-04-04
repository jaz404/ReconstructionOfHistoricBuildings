import ffmpeg
import os

out_dir = "test"
os.makedirs(out_dir, exist_ok=True)

(
    ffmpeg
    .input("csc_frontface.mp4")
    .filter("fps", fps=2)
    .filter("scale", 1920, -1)
    .output(f"{out_dir}/frame_%05d.png")
    .run()
)