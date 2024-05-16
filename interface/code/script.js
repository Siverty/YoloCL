// Get access to the webcam video element
const webcamElement = document.getElementById('webcam');

// Access the user's webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        // Set the video source to the stream from the webcam
        webcamElement.srcObject = stream;
    })
    .catch(error => {
        console.error('Error accessing webcam:', error);
    });

async function detectObjects() {
    const canvas = document.createElement('canvas');
    canvas.width = webcamElement.videoWidth;
    canvas.height = webcamElement.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(webcamElement, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg');
    const blob = await (await fetch(dataUrl)).blob();
    const formData = new FormData();
    formData.append('image', blob, 'frame.jpg');

    const response = await fetch('http://localhost:5000/detect', {
        method: 'POST',
        body: formData,
    });
    const detections = await response.json();
    displayDetections(detections);
}

function displayDetections(detections) {
    const canvas = document.getElementById('overlay');
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(detection => {
        const [x1, y1, x2, y2, confidence, classId] = detection;
        context.strokeStyle = 'red';
        context.lineWidth = 2;
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
}

setInterval(detectObjects, 250);
