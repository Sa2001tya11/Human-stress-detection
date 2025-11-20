// Global variables
let modelMetrics = null;
let confusionMatrices = null;
let correlationData = null;
let stressDistribution = null;

// Stress level labels
const stressLabels = {
    0: 'Low Stress',
    1: 'Normal',
    2: 'Medium Stress',
    3: 'High Stress',
    4: 'Very High Stress'
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', async function() {
    await loadModelData();
    initializeCharts();
});

// Load model data from backend
// Load model data from backend
async function loadModelData() {
    try {
        // Load model metrics
        const metricsResponse = await fetch('/model-metrics');
        const metricsData = await metricsResponse.json();
        if (metricsData.success) {
            modelMetrics = metricsData.metrics;
        }

        // Load confusion matrices
        const confusionResponse = await fetch('/confusion-matrix');
        const confusionData = await confusionResponse.json();
        if (confusionData.success) {
            confusionMatrices = confusionData.confusion_matrices;
        }

        // Load correlation matrix
        const correlationResponse = await fetch('/correlation-matrix');
        const corrData = await correlationResponse.json();
        if (corrData.success) {
            correlationData = corrData;
        }

        // Load stress distribution
        const distributionResponse = await fetch('/stress-distribution');
        const distData = await distributionResponse.json();
        if (distData.success) {
            stressDistribution = distData.distribution;
        }

        console.log('Model data loaded successfully');
    } catch (error) {
        console.error('Error loading model data:', error);
    }
}

// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form data
    const formData = {
        snoring: document.getElementById('snoring').value,
        respiration: document.getElementById('respiration').value,
        temperature: document.getElementById('temperature').value,
        limb: document.getElementById('limb').value,
        oxygen: document.getElementById('oxygen').value,
        eye: document.getElementById('eye').value,
        hours: document.getElementById('hours').value,
        heart: document.getElementById('heart').value
    };
    
    try {
        // Send prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Error making prediction: ' + error.message);
    }
});

// Display prediction results
function displayResults(result) {
    // Show results section
    document.getElementById('results').style.display = 'block';
    
    // Update predictions
    document.getElementById('dtResult').textContent = result.decisionTree;
    document.getElementById('dtLabel').textContent = stressLabels[result.decisionTree];
    
    document.getElementById('rfResult').textContent = result.randomForest;
    document.getElementById('rfLabel').textContent = stressLabels[result.randomForest];
    
    document.getElementById('avgResult').textContent = result.average;
    document.getElementById('avgLabel').textContent = stressLabels[result.average];
    
    // Update confidence
    const confidencePercent = (result.confidence * 100).toFixed(1);
    document.getElementById('confidence').textContent = confidencePercent + '%';
    document.getElementById('confidenceFill').style.width = confidencePercent + '%';
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    // Remove active class from all buttons
    const buttons = document.querySelectorAll('.tab-button');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// Initialize all charts
function initializeCharts() {
    if (modelMetrics) {
        createMetricsCharts();
    }
    if (confusionMatrices) {
        createConfusionMatrices();
    }
    if (correlationData) {
        createCorrelationHeatmap();
    }
    if (stressDistribution) {
        createDistributionChart();
    }
}

// Create metrics charts
function createMetricsCharts() {
    // Decision Tree Metrics
    const dtCtx = document.getElementById('dtMetricsChart').getContext('2d');
    new Chart(dtCtx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [{
                label: 'Decision Tree',
                data: [
                    modelMetrics.dt.accuracy,
                    modelMetrics.dt.precision,
                    modelMetrics.dt.recall,
                    modelMetrics.dt.f1Score
                ],
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(102, 126, 234, 1)'
            }]
        },
        options: {
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
    
    // Random Forest Metrics
    const rfCtx = document.getElementById('rfMetricsChart').getContext('2d');
    new Chart(rfCtx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [{
                label: 'Random Forest',
                data: [
                    modelMetrics.rf.accuracy,
                    modelMetrics.rf.precision,
                    modelMetrics.rf.recall,
                    modelMetrics.rf.f1Score
                ],
                backgroundColor: 'rgba(118, 75, 162, 0.2)',
                borderColor: 'rgba(118, 75, 162, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(118, 75, 162, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(118, 75, 162, 1)'
            }]
        },
        options: {
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Create confusion matrices
function createConfusionMatrices() {
    // Create DT confusion matrix
    createHeatmap('dtConfusionMatrix', confusionMatrices.dt, 'Decision Tree Confusion Matrix');
    
    // Create RF confusion matrix
    createHeatmap('rfConfusionMatrix', confusionMatrices.rf, 'Random Forest Confusion Matrix');
}

// Create heatmap helper function
function createHeatmap(elementId, data, title) {
    const labels = ['Low', 'Normal', 'Medium', 'High', 'Very High'];
    
    const trace = {
        z: data,
        x: labels,
        y: labels,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: true,
        text: data.map(row => row.map(val => val.toString())),
        texttemplate: '%{text}',
        textfont: {
            size: 14,
            color: 'white'
        },
        hovertemplate: 'True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    };
    
    const layout = {
        xaxis: {
            title: 'Predicted Label',
            side: 'bottom'
        },
        yaxis: {
            title: 'True Label',
            autorange: 'reversed'
        },
        margin: {
            l: 100,
            r: 50,
            t: 50,
            b: 100
        },
        height: 400
    };
    
    Plotly.newPlot(elementId, [trace], layout, {responsive: true});
}

// Create correlation heatmap
function createCorrelationHeatmap() {
    if (!correlationData || !correlationData.correlation_matrix) {
        console.error('Correlation data not available');
        return;
    }

    const features = correlationData.features.map(f => {
        // Shorten feature names for display
        const shortNames = {
            'snoring_rate': 'Snoring',
            'respiration_rate': 'Respiration',
            'body_temperature': 'Temperature',
            'limb_movement': 'Limb Move',
            'blood_oxygen': 'Blood O2',
            'eye_movement': 'Eye Move',
            'sleeping_hours': 'Sleep Hours',
            'heart_rate': 'Heart Rate'
        };
        return shortNames[f] || f;
    });
    
    const trace = {
        z: correlationData.correlation_matrix,
        x: features,
        y: features,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        showscale: true,
        text: correlationData.correlation_matrix.map(row => 
            row.map(val => val.toFixed(2))
        ),
        texttemplate: '%{text}',
        textfont: {
            size: 12
        },
        hovertemplate: '%{y} vs %{x}: %{z:.2f}<extra></extra>'
    };
    
    const layout = {
        title: 'Feature Correlation Matrix',
        xaxis: {
            tickangle: -45
        },
        yaxis: {
            autorange: 'reversed'
        },
        margin: {
            l: 100,
            r: 50,
            t: 100,
            b: 100
        },
        height: 500
    };
    
    Plotly.newPlot('correlationHeatmap', [trace], layout, {responsive: true});
}

// Create stress distribution chart
function createDistributionChart() {
    if (!stressDistribution) {
        console.error('Stress distribution data not available');
        return;
    }

    const ctx = document.getElementById('distributionChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Low Stress', 'Normal', 'Medium Stress', 'High Stress', 'Very High Stress'],
            datasets: [{
                label: 'Number of Cases',
                data: stressDistribution,
                backgroundColor: [
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 206, 86, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(255, 99, 132, 0.8)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(255, 159, 64, 1)',
                    'rgba(255, 99, 132, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Distribution of Stress Levels in Dataset',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Cases'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Stress Level'
                    }
                }
            }
        }
    });
}