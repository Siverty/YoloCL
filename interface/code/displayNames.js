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

// Display the class names on the interface in a list
async function displayNames() {
    const names = await fetchNames();
    if (!names.length) {
        console.error('No names found in data.yaml');
        return;
    }

    const predictionList = document.getElementById('prediction-list');
    predictionList.innerHTML = '';

    const maxItemsPerRow = 10;
    let currentRow = document.createElement('div');
    currentRow.classList.add('prediction-row');
    predictionList.appendChild(currentRow);

    names.forEach((name, index) => {
        if (index > 0 && index % maxItemsPerRow === 0) {
            currentRow = document.createElement('div');
            currentRow.classList.add('prediction-row');
            predictionList.appendChild(currentRow);
        }

        const listItem = document.createElement('li');
        listItem.textContent = `${name}: xx%`; // Placeholder for confidence
        currentRow.appendChild(listItem);
    });
}

// Display the class names when the DOM content is loaded
document.addEventListener('DOMContentLoaded', displayNames);
