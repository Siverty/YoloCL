// Description: Display names from the projects data.yaml on the interface.

let classNames = [];

// Fetch the project name from the server
async function fetchProjectName() {
    try {
        const response = await fetch('/get_project_name');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        return data.project_name;
    } catch (error) {
        console.error('Error fetching project name:', error);
        return null;
    }
}

// Fetch the data.yaml file and extract the class names
async function fetchNames() {
    try {
        const projectName = await fetchProjectName();
        if (!projectName) {
            throw new Error('Project name not found');
        }

        const response = await fetch(`/data/${projectName}/yaml-files/data.yaml`);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const yamlText = await response.text();
        const yamlData = jsyaml.load(yamlText);
        console.log('Fetched names:', yamlData.names);
        classNames = yamlData.names;
        return yamlData.names;
    } catch (error) {
        console.error('Error fetching or parsing data.yaml:', error);
        return [];
    }
}

// Initialize the predictions list with all class names set to 0% confidence
function initializePredictionsList() {
    const predictionList = document.getElementById('prediction-list');
    predictionList.innerHTML = '';

    classNames.forEach(name => {
        const listItem = document.createElement('li');
        listItem.id = `prediction-${name}`;
        listItem.textContent = `${name}: 0%`;
        predictionList.appendChild(listItem);
    });
}

// Function to update the predictions list with new detections
function updatePredictions(detections) {
    // Reset all confidences to 0%
    classNames.forEach(name => {
        const listItem = document.getElementById(`prediction-${name}`);
        listItem.textContent = `${name}: 0%`;
    });

    // Update the list with the new detection confidences
    detections.forEach(detection => {
        const [x1, y1, x2, y2, confidence, classId] = detection;
        const name = classNames[classId];
        const listItem = document.getElementById(`prediction-${name}`);
        listItem.textContent = `${name}: ${(confidence * 100).toFixed(2)}%`;
    });
}

// Function to initialize the display names
async function initializeDisplayNames() {
    const names = await fetchNames();
    if (!names.length) {
        console.error('No names found in data.yaml');
        return;
    }

    // Initialize the predictions list
    initializePredictionsList();

    // Initialize the worker to listen for detections
    const worker = new Worker('worker.js');
    worker.onmessage = function (e) {
        const { detections } = e.data;
        console.log('Detections received from worker:', detections);
        updatePredictions(detections);
    };
}

// Initialize the display names when the DOM content is loaded
document.addEventListener('DOMContentLoaded', initializeDisplayNames);
