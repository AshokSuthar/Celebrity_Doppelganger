
#!/usr/bin/python
from PIL import Image
import os, sys

DIRECTORY_PATH = "temp_data/known2/"

def resize(PATH):
    for item in os.listdir(PATH):
        if os.path.isfile(PATH+item):
            im = Image.open(PATH+item)
            f, e = os.path.splitext(PATH+item)
            print(f)
            imResize = im.resize((250,250), Image.ANTIALIAS)
            rgb_im = imResize.convert('RGB')
            rgb_im.save(f+e, quality=90)
            print("converting.")
            print("converting..")
            print("converting...")
if __name__ == '__main__':
    for folder in os.listdir(DIRECTORY_PATH):
        folder = folder + "/"
        resize(DIRECTORY_PATH+folder)