import os
from os import listdir
from PIL import Image

DIRECTORY_PATH = "temp_data/known2/"

count=0
for folder in os.listdir(DIRECTORY_PATH):
    folder = folder+"/"
    for filename in os.listdir(DIRECTORY_PATH+folder):
        try:
            img=Image.open(DIRECTORY_PATH+folder+filename)
            img.verify()
        except (IOError,SyntaxError) as e:
            print('Bad file  :  '+filename)
            count=count+1
            print(count)
            os.remove(DIRECTORY_PATH+folder+filename)