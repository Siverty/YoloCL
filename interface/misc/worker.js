// Description: Web worker script that sends video frames to the server for object detection.

// Receive messages from webcam.js
self.onmessage = async function (e) {
    const { imageBitmap } = e.data;

    // Create an offscreen canvas to capture video frames
    const captureCanvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
    const captureContext = captureCanvas.getContext('2d');

    // Draw the video frame onto the canvas
    captureContext.drawImage(imageBitmap, 0, 0);

    // Convert the canvas to a blob (JPEG format with reduced quality)
    const blob = await captureCanvas.convertToBlob({ type: 'image/jpeg', quality: 0.8 });

    // Send the blob to the server for object detection
    if (blob) {
        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');

        // Send the blob to the server for object detection
        try {
            const response = await fetch('http://localhost:5000/detect', {
                method: 'POST',
                body: formData,
            });

            // Receive the detected objects from the server
            if (response.ok) {
                const detections = await response.json();
                self.postMessage({ detections });
            } else {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        } catch (error) {
            console.error('Error during detection: ', error);
        }
    } else {
        console.error('Failed to create blob from canvas');
    }
};
