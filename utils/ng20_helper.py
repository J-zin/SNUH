import numpy as np

def extract_subject(text):
    subject = " ".join(text.split("\n")[1].split(' ')[1:]).strip()
    return subject