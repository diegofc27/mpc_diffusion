import glob
import os
from PIL import Image
def make_gif(frame_folder):
    frames = []
    for i in range(200):
        name = f"{frame_folder}/{i}.png"
        frames.append(Image.open(name))
    for i in range(50):
        frames.append(Image.open(f"{frame_folder}/199.png"))
    frame_one = frames[0]
    frame_one.save("/home/fernandi/projects/decision-diffuser/code/skills/analysis/reset_50/gifs/up_.gif", format="GIF", append_images=frames,
               save_all=True, duration=120, loop=0)
    #finishing the gif in the last frame    
    
if __name__ == "__main__":
    make_gif("/home/fernandi/projects/decision-diffuser/code/skills/analysis/reset_50/steps/up")