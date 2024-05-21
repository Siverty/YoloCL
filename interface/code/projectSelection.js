// projectSelection.js

// Function to fetch projects from config.json
async function fetchProjects() {
    try {
        const response = await fetch('/data/config.json'); // Adjust the path if needed
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        return data.projects;
    } catch (error) {
        console.error('Error fetching projects:', error);
        return {};
    }
}

// Function to populate the project list in the modal
async function populateProjectList() {
    const projects = await fetchProjects();
    const projectList = document.getElementById('project-list');
    projectList.innerHTML = '';

    for (const projectName in projects) {
        const button = document.createElement('button');
        button.textContent = projectName;
        button.addEventListener('click', () => selectProject(projectName, projects[projectName]));
        projectList.appendChild(button);
    }
}

// Function to handle project selection
function selectProject(projectName, projectConfig) {
    console.log(`Selected project: ${projectName}`, projectConfig);
    // Close the modal
    document.getElementById('project-modal').style.display = 'none';
    // Here you can add code to load the selected project's configuration
}

// Modal handling
document.getElementById('project-button').addEventListener('click', () => {
    document.getElementById('project-modal').style.display = 'block';
    populateProjectList();
});

document.getElementsByClassName('close-button')[0].addEventListener('click', () => {
    document.getElementById('project-modal').style.display = 'none';
});

window.addEventListener('click', (event) => {
    if (event.target === document.getElementById('project-modal')) {
        document.getElementById('project-modal').style.display = 'none';
    }
});
