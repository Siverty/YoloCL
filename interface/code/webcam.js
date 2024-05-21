// Wait until classNames is populated before starting detection
const checkClassNamesInterval = setInterval(() => {
    if (classNames.length > 0) {
        console.log('classNames loaded:', classNames);
        clearInterval(checkClassNamesInterval);
        startWebcamDetection();
    } else {
        console.log('Waiting for classNames to load...');
    }
}, 100);

function startWebcamDetection() {
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

    // Load the class names for the detected objects
    const worker = new Worker('worker.js');
    let detectionInterval = 100;

    // Receive the detected objects from the worker
    worker.onmessage = function (e) {
        const { detections } = e.data;
        console.log('Detections received from worker:', detections);
        displayDetections(detections);
        updatePredictions(detections);  // Ensure predictions list is updated
    };

    // Function to detect objects in the webcam video stream
    async function detectObjects() {
        const startTime = performance.now();

        // Ensure the webcam video is ready
        if (webcamElement.readyState !== webcamElement.HAVE_ENOUGH_DATA) {
            console.warn('Webcam video not ready');
            return;
        }

        // Match overlay canvas size with the video element size
        overlayCanvas.width = webcamElement.videoWidth;
        overlayCanvas.height = webcamElement.videoHeight;

        // Match capture canvas size with the video element size
        captureCanvas.width = webcamElement.videoWidth;
        captureCanvas.height = webcamElement.videoHeight;
        captureContext.drawImage(webcamElement, 0, 0, captureCanvas.width, captureCanvas.height);

        // Convert the capture canvas to an ImageBitmap
        const imageBitmap = await createImageBitmap(captureCanvas);

        // Send the ImageBitmap to the worker for object detection
        worker.postMessage({ imageBitmap }, [imageBitmap]);

        const endTime = performance.now();
        const processingTime = endTime - startTime;
        console.log(`Detection processing time: ${processingTime.toFixed(2)} ms`);
        clearInterval(intervalId);
        intervalId = setInterval(detectObjects, detectionInterval);
    }

    // Function to display the detected objects on the overlay canvas
    function displayDetections(detections) {
        context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        // Get the dimensions of the overlay canvas
        const canvasWidth = overlayCanvas.width;
        const canvasHeight = overlayCanvas.height;

        // Loop through the detected objects and draw boxes around them with the class names inside
        detections.forEach(detection => {
            let [x1, y1, x2, y2, confidence, classId] = detection;

            // Scale the coordinates based on the canvas size
            x1 = x1 / 640 * canvasWidth;
            y1 = y1 / 640 * canvasHeight;
            x2 = x2 / 640 * canvasWidth;
            y2 = y2 / 640 * canvasHeight;

            // Get the class name for the detected object from the classNames array in displayNames.js
            const className = classNames[classId];

            // Draw the bounding box around the detected object
            console.log(`Drawing box: x1=${x1}, y1=${y1}, x2=${x2}, y2=${y2}, confidence=${confidence}, classId=${classId}`);
            context.strokeStyle = 'red';
            context.lineWidth = 5;
            context.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Measure the width of the text to scale the background dynamically
            context.font = '52px Arial';
            const textWidth = context.measureText(className).width;
            const textHeight = 53; // Height of the text background

            // Draw black rectangle for text background at the bottom left of the bounding box
            context.fillStyle = 'black';
            context.fillRect(x1, y2 - textHeight, textWidth + 10, textHeight);

            // Draw the class name inside the bounding box
            context.fillStyle = 'white';
            context.fillText(className, x1 + 5, y2 - 5);
        });
    }

    // Start detecting objects
    let intervalId = setInterval(detectObjects, detectionInterval);
}
