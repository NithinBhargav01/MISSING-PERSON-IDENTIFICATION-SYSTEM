from django.shortcuts import render,redirect
from .models import* 
from django.contrib import messages
from django.core.mail import send_mail
from django.template.loader import render_to_string
from datetime import datetime
# import face_recognition
import cv2
import numpy as np
from twilio.rest import Client
from django.shortcuts import get_object_or_404, redirect
from django.http import HttpResponse
import os

#Add yourr own credentials
# account_sid = 'xxxxxxxxxxxxxxxxxxxxxxxxxxx'
# auth_token = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
# twilio_whatsapp_number = '+14155238886'


# Create your views here.
def home(request):
    return render(request,"index.html")

# def send_whatsapp_message(to,context):
#     client = Client(account_sid, auth_token)
#     whatsapp_message = (
#     f"Dear {context['fathers_name']},\n\n"
#     f"We are pleased to inform you that the missing person missing from {context['missing_from']} and you were concerned about has been found. "
#     "The person was located in a camera footage, and we have identified their whereabouts.\n\n"
#     "Here are the details:\n"
#     f" - Name: {context['first_name']} {context['last_name']}\n"
#     f" - Date and Time of Sighting: {context['date_time']}\n"
#     f" - Location: {context['location']}\n"
#     f" - Aadhar Number: {context['aadhar_number']}\n\n"
#     #"We understand the relief this news must bring to you. If you have any further questions or require more information, please do not hesitate to reach out to us.\n\n"
#     "Thank you for your cooperation and concern in this matter.\n\n"
#     "Sincerely,\n\n"
#     "Team Bharatiya Rescue ")
#     message = client.messages.create(
#         body=whatsapp_message,
#         from_='whatsapp:' + twilio_whatsapp_number,
#         to='whatsapp:' + to
#     )
#     print(f"WhatsApp message sent: {message.sid}")
 
def detect(request):
    # Load all missing person images from database
    missing_persons = MissingPerson.objects.all()
    
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Prepare face encodings for all missing persons
    known_face_encodings = []
    known_face_names = []
    
    # Load and preprocess all missing person images
    for person in missing_persons:
        if person.image:
            try:
                # Read the stored image
                img = cv2.imread(person.image.path)
                if img is not None:
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Detect face in the image
                    face_rects = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
                    
                    for (x, y, w, h) in face_rects:
                        face_roi = gray[y:y+h, x:x+w]
                        # Resize to a standard size for comparison
                        face_roi = cv2.resize(face_roi, (100, 100))
                        known_face_encodings.append(face_roi)
                        known_face_names.append(f"{person.first_name} {person.last_name}")
            except Exception as e:
                print(f"Error processing image for {person.first_name}: {str(e)}")
                continue
    
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        messages.error(request, 'Webcam could not be started. Please check your camera connection.')
        return render(request, "surveillance.html")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            messages.error(request, 'Failed to grab frame from webcam.')
            break
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            # Resize to match known faces
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Compare with known faces
            best_match = None
            best_match_value = float('inf')
            
            for i, known_face in enumerate(known_face_encodings):
                # Calculate difference between faces
                diff = cv2.absdiff(face_roi, known_face)
                match_value = np.mean(diff)
                
                if match_value < best_match_value:
                    best_match_value = match_value
                    best_match = (known_face_names[i], match_value)
            
            # Display match information
            if best_match and best_match_value < 50:  # Threshold for recognition
                name, match_value = best_match
                # Display match information
                cv2.putText(frame, f'Match: {name}', 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f'Confidence: {100 - (match_value/50)*100:.1f}%', 
                          (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Match Found', 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Display the webcam feed with face detection and matching
        cv2.imshow('Missing Person Detection', frame)
        
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    messages.success(request, 'Face detection and matching completed!')
    return render(request, "surveillance.html")

def surveillance(request):
    return render(request,"surveillance.html")


def register(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        father_name = request.POST.get('fathers_name')
        date_of_birth = request.POST.get('dob')
        address = request.POST.get('address')
        phone_number = request.POST.get('phonenum')
        aadhar_number = request.POST.get('aadhar_number')
        missing_from = request.POST.get('missing_date')
        email = request.POST.get('email')
        image = request.FILES.get('image')
        gender = request.POST.get('gender')
        aadhar = MissingPerson.objects.filter(aadhar_number=aadhar_number)
        if aadhar.exists():
            messages.info(request, 'Aadhar Number already exists')
            return redirect('/register')
        person = MissingPerson.objects.create(
            first_name = first_name,
            last_name = last_name,
            father_name = father_name,
            date_of_birth = date_of_birth,
            address = address,
            phone_number = phone_number,
            aadhar_number = aadhar_number,
            missing_from = missing_from,
            email = email,
            image = image,
            gender = gender,
        )
        person.save()
        messages.success(request,'Case Registered Successfully')
        current_time = datetime.now().strftime('%d-%m-%Y %H:%M')
        subject = 'Case Registered Successfully'
        from_email = 'pptodo01@gmail'
        recipientmail = person.email
        context = {"first_name":person.first_name,"last_name":person.last_name,
                    'fathers_name':person.father_name,"aadhar_number":person.aadhar_number,
                    "missing_from":person.missing_from,"date_time":current_time}
        html_message = render_to_string('regmail.html',context = context)
        # Send the email
        send_mail(subject,'', from_email, [recipientmail], fail_silently=False, html_message=html_message)

    return render(request,"register.html")


def  missing(request):
    queryset = MissingPerson.objects.all()
    search_query = request.GET.get('search', '')
    if search_query:
        queryset = queryset.filter(aadhar_number__icontains=search_query)
    
    context = {'missingperson': queryset}
    return render(request,"missing.html",context)

def delete_person(request, person_id):
    person = get_object_or_404(MissingPerson, id=person_id)
    person.delete()
    return redirect('missing')  # Redirect to the missing view after deleting


def update_person(request, person_id):
    person = get_object_or_404(MissingPerson, id=person_id)

    if request.method == 'POST':
        # Retrieve data from the form
        first_name = request.POST.get('first_name', person.first_name)
        last_name = request.POST.get('last_name', person.last_name)
        fathers_name = request.POST.get('fathers_name', person.fathers_name)
        dob = request.POST.get('dob', person.dob)
        address = request.POST.get('address', person.address)
        email = request.POST.get('email', person.email)
        phonenum = request.POST.get('phonenum', person.phonenum)
        aadhar_number = request.POST.get('aadhar_number', person.aadhar_number)
        missing_date = request.POST.get('missing_date', person.missing_date)
        gender = request.POST.get('gender', person.gender)

        # Check if a new image is provided
        new_image = request.FILES.get('image')
        if new_image:
            person.image = new_image

        # Update the person instance
        person.first_name = first_name
        person.last_name = last_name
        person.fathers_name = fathers_name
        person.dob = dob
        person.address = address
        person.email = email
        person.phonenum = phonenum
        person.aadhar_number = aadhar_number
        person.missing_date = missing_date
        person.gender = gender

        # Save the changes
        person.save()

        return redirect('missing')  # Redirect to the missing view after editing

    return render(request, 'edit.html', {'person': person})

def locations(request):
    locations = Location.objects.all()
    return render(request, "locations.html", {"locations": locations})
