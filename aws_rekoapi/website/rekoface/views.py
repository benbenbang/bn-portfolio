# standard library
import json
import os
from base64 import b64decode
from time import time

# pypi/conda library
import boto3
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render


def index(request):
    return render(request, "base.html")


bucket = "hello.rekognition"
region = "eu-west-1"
root_path = "tmp/"


def b64Decoding(b64text):
    # b64text read from somewhere
    b64String = b64text.split(",")[1]

    filename = str(int(time()))
    extension = ".jpg"
    pic = filename + extension
    pic_path = os.path.join(root_path, pic)

    with open(pic_path, "wb") as file:
        file.write(b64decode(b64String))

    return pic


def s3Upload(fileName):
    # Upload photo to S3
    try:
        s3 = boto3.resource("s3")
        file_path = os.path.join(root_path, fileName)
        s3.Object(bucket, fileName).put(Body=open(file_path, "rb"))
        print("Sucessfully upload to S3!")
    except Exception as e:
        print(e)


def detectFace(fileName):
    # Retrieve Data and Sent to Rekognition API
    try:
        client = boto3.client("rekognition", region)
        response = client.detect_faces(Image={"S3Object": {"Bucket": bucket, "Name": fileName}}, Attributes=["ALL"])

        print("Detected faces for " + fileName)
        for faceDetail in response["FaceDetails"]:
            print(
                "The detected face is between "
                + str(faceDetail["AgeRange"]["Low"])
                + " and "
                + str(faceDetail["AgeRange"]["High"])
                + " years old"
            )
            print("Here are the other attributes:")
            print(json.dumps(faceDetail, indent=4, sort_keys=True))
        response = retreiveData(response)
        return response
    except Exception as e:
        print("Something wrong:\n{0}".format(e))


def indexFace(fileName, faceID):
    try:
        client = boto3.client("rekognition", region)
        response = client.index_faces(
            CollectionId=collectionID,
            Image={"S3Object": {"Bucket": bucket, "Name": fileName}},
            ExternalImageId=faceID,
            DetectionAttributes=["ALL"],
        )
        response = retreiveData(response)
        return response
    except Exception as e:
        print("Something wrong:\n{0}".format(e))


def register(request):
    try:
        faceID, b64text = request.GET.get("faceID", "b64text", None)
    except:
        faceID = None
        b64text = request.GET.get("b64text", None)
    try:
        fileName = b64Decoding(b64text)
        s3Upload(fileName)
        client = boto3.client("rekognition", region)
        if faceID is None:
            print("Without giving faceID, going to process face detection only.")
            response = detectFace(fileName)
            return JsonResponse(response)
        elif faceID and type(faceID) is str:
            response = indexFace(fileName, faceID)
            print("Sucessfully register client's faceID")
            data = {"is_saved": response}
            return JsonResponse(data)
        else:
            raise ValueError("faceId must be string or you can just leave it as None")
    except Exception as e:
        print(e)


def rekognize(request):
    try:
        faceID, b64text = request.GET.get("faceID", "b64text", None)
        fileName = b64Decoding(b64text)
        s3Upload(fileName)
        client = boto3.client("rekognition", region)
        response = client.search_faces_by_image(
            CollectionId=collectionID,
            Image={"S3Object": {"Bucket": bucket, "Name": fileName}},
            MaxFaces=10,
            FaceMatchThreshold=0.7,
        )
        try:
            similarity = response["FaceMatches"][0]["Similarity"]
            faceAPIId = response["FaceMatches"][0]["Face"]["FaceId"]
            faceID = response["FaceMatches"][0]["Face"]["ExternalImageId"]
            imageid = response["FaceMatches"][0]["Face"]["ImageId"]
            knowIf = (
                "I know you! You're {0}".format(faceID)
                if float(similarity) > 0.7
                else "Take some more phote let me know you better :$"
            )
            response = {"response": response["FaceMatches"][0], "similarity": similarity, "bool": knowIf}
            print(
                "Found this person:\n\
	AmazonID: {0}\n\
	In photo {1}\n\
	Similarity {2:.4f}%\n\
	Registered by {3}".format(
                    faceAPIId, imageid, similarity, faceID
                )
            )
            return JsonResponse(response)
        except Exception as e:
            print(e)
            response = {"response": "Something wrong, cannot process", "bool": False}
            return JsonResponse(response)
    except Exception as e:
        print("Something wrong:\n{0}".format(e))


def retreiveData(data):
    faceDetails = data["FaceDetails"][0]
    Age = (
        "The detected face is between "
        + str(faceDetails["AgeRange"]["Low"])
        + " and "
        + str(faceDetails["AgeRange"]["High"])
        + " years old"
    )
    Gender = faceDetails["Gender"]["Value"]
    Emotions = [
        faceDetails["Emotions"][i]["Type"]
        for i in range(len(faceDetails["Emotions"]))
        if faceDetails["Emotions"][i]["Confidence"] >= 70
    ]
    Smile = faceDetails["Smile"]["Value"]
    Beard = faceDetails["Beard"]["Value"]
    Mustache = faceDetails["Mustache"]["Value"]
    Eyeglasses = faceDetails["Eyeglasses"]["Value"]
    Sunglasses = faceDetails["Sunglasses"]["Value"]
    EyesOpen = faceDetails["EyesOpen"]["Value"]
    MouthOpen = faceDetails["MouthOpen"]["Value"]
    Brightness = (
        "You seems like in a bright room"
        if faceDetails["Quality"]["Brightness"] > 0.5
        else "You seems like in a room a bit be sombre"
    )

    jsonResponse = {
        "Age": Age,
        "Gender": Gender,
        "Emotions": Emotions,
        "Smile": Smile,
        "Beard": Beard,
        "Mustache": Mustache,
        "Eyeglasses": Eyeglasses,
        "Sunglasses": Sunglasses,
        "EyesOpen": EyesOpen,
        "MouthOpen": MouthOpen,
        "Brightness": Brightness,
    }

    return jsonResponse
