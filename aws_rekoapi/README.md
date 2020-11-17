# Play with Amazon Rekognition API

## Features

I'll build a web application for face recognition by using Amazon Rekognition API. This application could be applied to some use cases like **attendance** or **auto-login**…etc. But be awared of **GDPR**, if you are processing data in / from clients who live in Europe.

## Requirement

1. For the dependency:

   `pip3 install -r reqirement.txt`

   Basically, you will only need install `boto3` and `django` from `pip`

2. For AWS Service:

   `pip3 install awscli --upgrade --user`

   after install, you might want to check

   ```bash
   $ aws --version
   aws-cli/1.12.1 Python/3.5.2 Darwin/16.7.0 botocore/1.8.1
   ```

   then you'll need to configure the aws cli in your mac / pc

   ```bash
   $ aws configure
   AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
   AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   Default region name [None]: us-west-2
   Default output format [None]: json
   # The Key ID and Secret are just examples :)
   ```

3. For the website:

   1. Download `webcamjs` from [the repo of **jhuckaby**](https://github.com/jhuckaby/webcamjs.git)
   2. Download Bootstrap V4 from [their official site](https://getbootstrap.com)

## Django Go!

From the beginning, we'll need to start a new project by using `django`

```bash
$ django-admin.py startproject website
$ cd website
```

The structure of the current project will be looked like:

```txt
website/
├── manage.py
└── website
    ├── __init__.py
    ├── settings.py
    ├── urls.py
    └── wsgi.py
```

move your css files from bootstrap and webcamjs into `static/css`, `static/fonts` and `static/js`

we also need to let our site know this folder should be included, so we'll need to go to *website/settings.py* and add the following codes:

```python
# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.11/howto/static-files/

STATIC_URL = '/static/'
TMP_URL = '/tmp'

STATICFILES_DIRS = [
	os.path.join(BASE_DIR, "static"),
	os.path.join(BASE_DIR, "tmp")
]
```

## Create the web application: rekognition

Let us use `startapp` to create our `rekogition` service:

```bash
python3 manage.py startapp rekoface
```

then `startapp` will create a folder named `rekognition` with this structure:

```txt
rekoface
├── __init__.py
├── admin.py
├── migrations
├── models.py
├── tests.py
└── views.py
```

#### Add Our new app into list

open *website/settings.py* again, and find [INSTALLED_APPS](https://docs.djangoproject.com/en/1.8/ref/settings/#std:setting-INSTALLED_APPS), then configure the file as following：

```python
INSTALLED_APPS = [
	'django.contrib.admin',
	'django.contrib.auth',
	'django.contrib.contenttypes',
	'django.contrib.sessions',
	'django.contrib.messages',
	'django.contrib.staticfiles',
	'rekoface'
]
```

also *website/urls.py* need to be configured:

```python
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^rekoface/', include("rekoface.urls", namespace='rekoface')),
]
```

## How to use Rekognition API

The mechanism of Amazon Rekognition API will like:

![mechanism](assets/mechanism.png)

Thus to implement this features, I'll simply upload our photos to S3 but if you're using `NodeJS` as your backend, or you can use `angularjs` to deal with your frontend data, then it will be easier to just transform the photo into `base64` format then send it to the Amazon Reko API. Finally, the result sent from API will be `json` format. So to easier handle these data, I'll recommend you to implement your front-end by `angularjs`…etc (whatever framework you like, in fact. Just don't use pure javascript...)

### Implementation

To implement this feature, we will realize these functions in `view.py`

1. First, for the html structure:

   ```python
   from django.http import HttpResponse, JsonResponse
   from django.shortcuts import render, get_object_or_404, redirect
   import boto3, json, os
   from base64 import b64decode
   from time import time

   def index(request):
   	return render(request, 'base.html')
   ```


2. Depends on which region of the API in Amazon you would like to use, at this time, they only offer Rekognition API in **US East (N. Virginia)**, **US West (Oregon)** and **EU (Ireland)** (Yes…only three)

   - us-east-1
   - us-west-2
   - eu-west-1

   Declare your bucket to save the uploaded photos, which region you want to use and where you are going to save the photos in local

   ```python
   bucket = 'hello.rekognition'
   region = 'eu-west-1'
   root_path = 'tmp/'
   # Noted that S3 and Reko API could be in different region, here I just want to simplify my work
   ```

3. To upload our photos to S3:

   ```python
   def s3Upload(fileName):
   	# Upload photo to S3
   	try:
   		s3 = boto3.resource('s3')
   		file_path = os.path.join(root_path, fileName)
   		s3.Object(bucket, fileName).put(Body=open(file_path, 'rb'))
   		print('Sucessfully upload to S3!')
   	except Exception as e:
   		print(e)
   ```

4. To implement face detection and index our face id and features from detection:

   ```python
   def detectFace(fileName):
   	# Retrieve Data and Sent to Rekognition API
   	try:
   		client=boto3.client('rekognition', region)
   		response = client.detect_faces(
   								Image={'S3Object':{'Bucket':bucket, 'Name':fileName}},
   								Attributes=['ALL']
   								)

   		print('Detected faces for ' + fileName)
   		for faceDetail in response['FaceDetails']:
   			print('The detected face is between ' + str(faceDetail['AgeRange']['Low'])
   				  + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')
   			print('Here are the other attributes:')
   			print(json.dumps(faceDetail, indent=4, sort_keys=True))
   		response = retreiveData(response)
   		return response
   	except Exception as e:
   		print('Something wrong:\n{0}'.format(e))

   def indexFace(fileName, faceID):
   	try:
   		client=boto3.client('rekognition', region)
   		response = client.index_faces(
   			CollectionId=collectionID,
   			Image={'S3Object': {'Bucket': bucket, 'Name': fileName}},
   			ExternalImageId=faceID,
   			DetectionAttributes=['ALL']
   		)
   		response = retreiveData(response)
   		return response
   	except Exception as e:
   		print('Something wrong:\n{0}'.format(e))
   ```

   5. Now we need to combine the functions in step4. For the user who won't like to be registered, we will only send their data by using `detectFace`. And for those who want to register their features with their userid they created, we should use `indexFace`

      ```python
      def register(request):
      	try:
      		faceID, b64text = request.GET.get('faceID', 'b64text', None)
      	except:
      		faceID = None
      		b64text = request.GET.get('b64text', None)
      	try:
      		fileName = b64Decoding(b64text)
      		s3Upload(fileName)
      		client=boto3.client('rekognition', region)
      		if faceID is None:
      			print('Without giving faceID, going to process face detection only.')
      			response = detectFace(fileName)
      			return(JsonResponse(response))
      		elif faceID and type(faceID) is str:
      			response = indexFace(fileName, faceID)
      			print("Sucessfully register client's faceID")
      			data = {
      				'is_saved': response
      			}
      			return(JsonResponse(data))
      		else:
      			raise ValueError("faceId must be string or you can just leave it as None")
      	except Exception as e:
      		print(e)
      ```

   6. After registered, when user want to **log in**, or say to be **recognized**, we will use this function to check whether his or her face has already been saved and could be recognized.

      ```python
      def rekognize(request):
      	try:
      		faceID, b64text = request.GET.get('faceID', 'b64text', None)
      		fileName = b64Decoding(b64text)
      		s3Upload(fileName)
      		client=boto3.client('rekognition', region)
      		response = client.search_faces_by_image(
          		CollectionId=collectionID, Image={'S3Object': {'Bucket': bucket, 'Name': fileName}},
          		MaxFaces=10,
          		FaceMatchThreshold=0.7
      		)
      		try:
      			similarity = response['FaceMatches'][0]['Similarity']
      			faceAPIId = response['FaceMatches'][0]['Face']['FaceId']
      			faceID = response['FaceMatches'][0]['Face']['ExternalImageId']
      			imageid = response['FaceMatches'][0]['Face']['ImageId']
      			knowIf = "I know you! You're {0}".format(faceID) if float(similarity) > 0.7 else "Take some more phote let me know you better :$"
      			response = {
      				'response': response['FaceMatches'][0],
      				'similarity': similarity,
      				'bool': knowIf
      				}
      			print("Found this person:\n\
      	AmazonID: {0}\n\
      	In photo {1}\n\
      	Similarity {2:.4f}%\n\
      	Registered by {3}".format(faceAPIId, imageid, similarity, faceID))
      			return(JsonResponse(response))
      		except Exception as e:
      			print(e)
      			response = {
      				'response': 'Something wrong, cannot process',
      				'bool': False
      			}
      			return(JsonResponse(response))
      	except Exception as e:
      		print('Something wrong:\n{0}'.format(e))
      ```
