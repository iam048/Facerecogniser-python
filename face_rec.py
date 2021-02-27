#!/usr/bin/env python3

import face_recognition
import os, random, sys
import cv2
import pickle

from argparse import ArgumentParser as parse

parser = parse()

parser.add_argument("--extract-unknown",
	dest="EXTRACT",
	action="store_true",
	help="Extract the Unknown faces to the Local disk"
	)
parser.add_argument("--target-dir",
	dest="targetDir",
	help="Target Directory to Compare Faces / Generate HAS",
	required=True
	)
parser.add_argument("--gen-has",
	dest="GEN",
	help="Generates file of known faces",
	action="store_true"
	)
parser.add_argument("--known-has",
	dest="DAT",
	help="Holds a Known faces"
	)
ARG = parser.parse_args()

UNKNOWN = "unknown_faces"
TARGET = ARG.targetDir
TOL = 0.5
MODEL = "hog"

class faceRecogniser():
	
	def extract_faces(self,image):
		encodings = []
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		faces = face_recognition.face_locations(image, model=MODEL)

		if faces is not None:
			print("Processing Unknown faces")
			for (top, right, bottom, left) in faces:
				face = image[top:bottom, left:right]

				face_enc = face_recognition.face_encodings(face)
				result = self.compare(face_enc)
				if result is not True:
					cv2.imwrite(f"{UNKNOWN}/{random.random()}.jpg", face)

	def compare(self,encodings) -> bool:
		data = open(ARG.DAT, 'rb')
		data = pickle.load(data)

		names = []
		faces = []
		
		for key in data:
			names.append(key)
			faces.append(data[key])

		for face_encoding in encodings:
			match = face_recognition.compare_faces(faces, face_encoding, TOL)
			try:
				if True in match:
					name = names[match.index(True)]
					print(f"Ignoring Known Match: {name}")
					return True
				else:
					return False
			except IndexError:
				pass

	def gen_known_has(self):
		print("[+] Generating Data")
		faces = []
		names = []
		# list all files in directory
		for name in os.listdir(TARGET):
			for filename in os.listdir(f"{TARGET}/{name}"):
				image = face_recognition.load_image_file(f"{TARGET}/{name}/{filename}")
				encoding = face_recognition.face_encodings(image)[0]
				
				faces.append(encoding)
				names.append(name)

				has = {}
				for enc, name in zip(faces, names):
					has[name] = enc

				f = open("known_faces_hasscasdes.dat", 'wb')
				pickle.dump(has, f)
				f.close()

#Detecting Faces Unknown
try:
	recogniser = faceRecogniser()
	if ARG.EXTRACT:
		for filename in os.listdir(TARGET):
			print(filename)
			image = face_recognition.load_image_file(f"{TARGET}/{filename}")
			recogniser.extract_faces(image)

	if ARG.GEN:
		recogniser.gen_known_has()

except KeyboardInterrupt:
	print("Bye")
	exit()