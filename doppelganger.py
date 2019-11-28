import face_recognition
from os import listdir
from os.path import isfile, join
from itertools import compress
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

KNOWN_DIRECTORY_PATH = "temp_data/known/"
UNKNOWN_DIRECTORY_PATH = "Data/unknown/"
EMBEDDING_LIST = "temp_data/known_embeddings.npy"
FACE_NOT_DETECTED = []

def embeddings_list(known_file_paths):
    try:
        return np.load(EMBEDDING_LIST).tolist()
    except (IOError, FileNotFoundError)  as e:
        print(len(known_file_paths))
        known_embeddings = []
        count = 0
        for file_path in known_file_paths:
            known_picture = face_recognition.load_image_file(KNOWN_DIRECTORY_PATH + file_path)
            print(file_path, "Files Processed: ", count)
            face_bounding_boxes = face_recognition.face_locations(known_picture)
            # checking that the known pic has only a single face in it. You can change it accordingly
            if len(face_bounding_boxes) >= 1:
                known_embeddings.append(face_recognition.face_encodings(known_picture)[0])
                # print(np.array((face_recognition.face_encodings(known_picture)[0])).shape)
            else:
                # if no face is detected, adding high values for face embeddings
                known_embeddings.append(np.full((128,), 999).tolist())
            count = count + 1
        # saving the known encodings list to increase performance
        np.save(EMBEDDING_LIST, known_embeddings)
        return known_embeddings


def print_doppelganger(known_file_paths, matched_indices_list, file_you, face_no):
    doppelganger_name = "Your Celebrity DoppleGanger is {0}".format(known_file_paths[matched_indices_list[0]].split("_")[0])
    # img = mpimg.imread(KNOWN_DIRECTORY_PATH + known_file_paths[matched_indices_list[0]])
    # image_doppelganger = Image.open(KNOWN_DIRECTORY_PATH + known_file_paths[matched_indices_list[0]])
    # image = Image.fromarray(np.hstack((np.array(image_doppelganger), np.array(image_you)))).show()
    # image.show()
    # figure size in inches optional
    rcParams['figure.figsize'] = 11, 8

    # read images
    image_doppelganger = mpimg.imread(KNOWN_DIRECTORY_PATH + known_file_paths[matched_indices_list[0]])
    image_you = mpimg.imread(UNKNOWN_DIRECTORY_PATH + file_path)
    # img_B = mpimg.imread('\path\to\img_B.png')

    print(file_you,doppelganger_name)
    # display images
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("Hey you!!")
    ax[0].imshow(image_you);
    ax[0].set_yticklabels([])
    ax[0].set_xticklabels([])
    ax[1].set_title(doppelganger_name)
    ax[1].imshow(image_doppelganger);
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    plt.savefig(file_you[:13]+"Doppos/"+file_you[13:]+"_Doppo_"+str(face_no)+".png")

def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))
    # print("distance", np.linalg.norm(face_encodings - face_to_compare, axis=1))
    # print(np.linalg.norm(face_encodings - face_to_compare, axis=1).argmin())
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    arr = list(face_distance(known_face_encodings, face_encoding_to_check))
    min_dist_index = np.array(arr).argmin()
    # print("distance with the celebrity face: ",arr[min_dist_index])
    # index_1, = np.where(arr == np.partition(arr, 2)[2])
    # return index_1
    return [np.array(arr).argmin()]


if __name__ == '__main__':
    known_file_paths = [f for f in listdir(KNOWN_DIRECTORY_PATH) if isfile(join(KNOWN_DIRECTORY_PATH, f))]
    unknown_file_paths = [f for f in listdir(UNKNOWN_DIRECTORY_PATH) if isfile(join(UNKNOWN_DIRECTORY_PATH, f))]
    count = 1
    for file_path in unknown_file_paths:
        print("Processing file "+ str(count)+ file_path)
        count += 1
        unknown_picture = face_recognition.load_image_file(UNKNOWN_DIRECTORY_PATH + file_path)
        # Find all the faces in the test image using the default HOG-based model
        face_locations = face_recognition.face_locations(unknown_picture)
        no_of_faces = len(face_locations)
        known_embeddings = embeddings_list(known_file_paths)
        print(len(known_embeddings), "LENGTHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        print("\nNumber of faces detected: ", no_of_faces)
        for face_no in range(no_of_faces):
            unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[face_no]
            # using own compare_faces() func instead of from face_recognition library
            results = compare_faces(known_embeddings, unknown_face_encoding, tolerance=0.6)
            matched_indices_list = results
            # matched_indices_list = list(compress(range(len(results)), results))
            # print(matched_indices_list)
            file_you = UNKNOWN_DIRECTORY_PATH + file_path
            print_doppelganger(known_file_paths, matched_indices_list, file_you, face_no)
