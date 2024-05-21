// Get access to the webcam video element
const webcamElement = document.getElementById('webcam');
const overlayCanvas = document.getElementById('overlay');
const context = overlayCanvas.getContext('2d');

// Create a reusable canvas element for capturing frames
const captureCanvas = document.createElement('canvas');
const captureContext = captureCanvas.getContext('2d');

// Access the user's webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        // Set the video source to the stream from the webcam
        webcamElement.srcObject = stream;
    })
    .catch(error => {
        console.error('Error accessing webcam:', error);
    });

const worker = new Worker('worker.js');
let detectionInterval = 80;

worker.onmessage = function (e) {
    const { detections } = e.data;
    displayDetections(detections);
};

async function detectObjects() {
    const startTime = performance.now();

    // Ensure the webcam video is ready
    if (webcamElement.readyState !== webcamElement.HAVE_ENOUGH_DATA) {
        console.warn('Webcam video not ready');
        return;
    }

    captureCanvas.width = webcamElement.videoWidth;
    captureCanvas.height = webcamElement.videoHeight;
    captureContext.drawImage(webcamElement, 0, 0, captureCanvas.width, captureCanvas.height);

    const imageBitmap = await createImageBitmap(captureCanvas);

    worker.postMessage({ imageBitmap }, [imageBitmap]);

    const endTime = performance.now();
    console.log(`Detection took ${endTime - startTime} milliseconds`);

    detectionInterval = Math.max(100, endTime - startTime);
    clearInterval(intervalId);
    intervalId = setInterval(detectObjects, detectionInterval);
}

function displayDetections(detections) {
    context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection;
        context.strokeStyle = 'red';
        context.lineWidth = 2;
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
}

let intervalId = setInterval(detectObjects, detectionInterval);
