from sklearn.datasets import fetch_lfw_people
import numpy as np

def load_dataset():
    # Fetch the LFW dataset with at least 70 images per person, resizing to 0.4 for faster computation
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4, color=True)
    X = lfw_people.images  # Image data
    y = lfw_people.target  # Labels (person IDs)
    target_names = lfw_people.target_names  # Names of people corresponding to IDs
    n_samples, h, w, _ = lfw_people.images.shape  # Dataset dimensions
    return X, y, target_names, n_samples, h, w
