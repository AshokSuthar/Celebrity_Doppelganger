# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

# source: https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_svm.py

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

import face_recognition
import numpy as np
import pandas as pd
from sklearn import svm
import os
from pickle import dump, load
import os.path
from os import path

from mlxtend.plotting import plot_decision_regions

TRAIN_DIR = "temp_data/known2/"
TEST_DIR = "Data/unknown/"

def train():
    # Training the SVC classifier

    # # The training data would be all the face encodings from all the known images and the labels are their names
    # encodings = []
    # names = []
    #
    # # Training directory
    # train_dir = os.listdir(TRAIN_DIR)
    #
    # # Loop through each person in the training directory
    # count = 0
    # for person in train_dir:
    #     pix = os.listdir(TRAIN_DIR + person)
    #     # pix = [item for item in pix if not item.startswith('.') and os.path.isfile(os.path.join(root, item))]
    #
    #     # Loop through each training image for the current person
    #     for person_img in pix:
    #         print("processing..."+str(count), sep=" ", end="\r", flush=True)
    #         count += 1
    #         # Get the face encodings for the face in each image file
    #         # if person_img == ".DS_Store":
    #         #     continue
    #         face = face_recognition.load_image_file(
    #             TRAIN_DIR + person + "/" + person_img
    #         )
    #         face_bounding_boxes = face_recognition.face_locations(face)
    #
    #         # if there is atleast one or more faces
    #         if len(face_bounding_boxes) >= 1:
    #             face_enc = face_recognition.face_encodings(face)[0]
    #         else:
    #             # if no face is detected, adding high values for face embeddings
    #             face_enc = np.full((128,), 999).tolist()
    #         # Add face encoding for current image with corresponding label (name) to the training data
    #         encodings.append(face_enc)
    #         names.append(person)
    #
    # print("ENCODINGSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS",len(encodings))
    # df = pd.DataFrame(np.array(encodings))
    # df2 = pd.DataFrame(np.array(names))
    # df.to_csv("encodings.csv",index = None)
    # df2.to_csv("names.csv", index=None)
    encodings = pd.read_csv("encodings.csv", header=[0])
    names = pd.read_csv("names.csv",header=[0])
    print(names.columns)
    print(names.iloc[:,:].head())
    # Create and train the SVC classifier
    clf = svm.SVC(gamma="scale", probability=True)
    clf.fit(encodings, names['0'])

    with open("model.pkl", "wb") as f:
        dump(clf, f)
    return clf


def match(clf, filename):

    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file(filename)

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    print("\nNumber of faces detected: ", no)

    # Predict all the faces in the test image using the trained classifier
    firstname = None
    print("\n(☞ﾟヮﾟ)☞   ☜(ﾟヮﾟ☜)\n")
    print("Found:")
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        name = clf.predict([test_image_enc])
        probs = clf.predict_proba([test_image_enc])
        print(*name)
        firstname = (str(name[0]))
    return (firstname, no)


def main():
    # didn't have time to sort data to implement this, but it's not really needed
    # gender = input('Pick the gender of your celebrity doppleganger: (F)emale, (M)ale, or (B)oth').lower()[0]
    # while (gender not in ['f', 'm', 'b']):
    #     gender = input('Invalid input. \
    #         Pick the gender of your celebrity doppleganger: (F)emale, (M)ale, or (B)oth').lower()[0]

    # industry = input('Pick a film industry for your celebrity doppleganger (all works best!): (H)ollywood, (B)ollywood, (T)ollywood')
    # while (gender not in ['h', 'b', 't']):
    #     gender = input('Invalid input. \
    #         Pick a film industry for your celebrity doppleganger (all works best!): (H)ollywood, (B)ollywood, (T)ollywood').lower()[0]

    print("\n===================================================================")
    # print("Welcome to the Celebrity Doppleganger Finder!\n\n")
    print("\nWelcome to the")
    print("""
   ____     _      _          _ _
  / ___|___| | ___| |__  _ __(_) |_ _   _
 | |   / _ \ |/ _ \ '_ \| '__| | __| | | |
 | |__|  __/ |  __/ |_) | |  | | |_| |_| |
  \____\___|_|\___|_.__/|_|  |_|\__|\__, |
  ____                         _    |___/  _
 |  _ \  ___  _ __  _ __   ___| | __ _(_)_(_)_ __   __ _  ___ _ __
 | | | |/ _ \| '_ \| '_ \ / _ \ |/ _` |/ _` | '_ \ / _` |/ _ \ '__|
 | |_| | (_) | |_) | |_) |  __/ | (_| | (_| | | | | (_| |  __/ |
 |____/ \___/| .__/| .__/ \___|_|\__, |\__,_|_| |_|\__, |\___|_|
  _____ _    |_|   |_|           |___/             |___/
 |  ___(_)_ __   __| | ___ _ __
 | |_  | | '_ \ / _` |/ _ \ '__|
 |  _| | | | | | (_| |  __/ |
 |_|   |_|_| |_|\__,_|\___|_|
    """)

    # load classifier
    if path.exists('model.pkl'):
        with open("model.pkl", "rb") as f:
            clf = load(f)
            encodings = pd.read_csv("encodings.csv", header=[0])
            names = pd.read_csv("names.csv", header=[0])
            y = names['0'].values
            y = np.linspace(1, len(y), num=len(y))
            value = 1.5
            # Plot training sample with feature 3 = 1.5 +/- 0.75
            width = 0.75
            plot_decision_regions(X=encodings.values,
                                  y=y.astype(np.integer),
                                  filler_feature_values={2: value},
                                  filler_feature_ranges={2: width},
                                  clf=clf,
                                  legend=2)
    else:
        clf = train()
    print("The Machine Learning Model has been trained! 🧠\n\nNow, let's find your match.\n")

    while True:
        filename = ""
        while not os.path.isfile(filename):
            filename = input("Which file would you like to use? Press enter to default to test_image.jpg: ")
            if filename == "":
                filename = "test_image.jpg"
                break
            elif filename == 'q': # exit out of program
                return
            elif os.path.isfile(filename):
                break
            else:
                print("Invalid file name. Try again.\n")

        res, no = match(clf, filename)
        print("\n===================================================================\n")
        print(res)

main()