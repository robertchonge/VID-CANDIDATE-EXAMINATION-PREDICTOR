// VID DRIVER PULSE - Frontend JavaScript
// Handles form submission, API call to /predict, and result display

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect form data
    const data = {
        age: parseInt(document.getElementById('age').value),
        gender: document.getElementById('gender').value,
        test_station: document.getElementById('test_station').value,
        vehicle_type: document.getElementById('vehicle_type').value,
        licence_type: document.getElementById('licence_type').value,
        test_manoeuvre: document.getElementById('test_manoeuvre').value,
        training_hours: parseFloat(document.getElementById('training_hours').value),
        attempt_number: parseInt(document.getElementById('attempt_number').value)
    };

    // Validate required fields
    if (isNaN(data.age) || isNaN(data.training_hours) || isNaN(data.attempt_number)) {
        alert('Please fill all fields correctly.');
        return;
    }

    const resultDiv = document.getElementById('result');
    const predText = document.getElementById('predictionText');
    const confText = document.getElementById('confidenceText');
    const probDetails = document.getElementById('probabilityDetails');

    // Show loading state
    predText.textContent = '⏳';
    predText.className = 'display-4';
    confText.textContent = 'Predicting...';
    resultDiv.style.display = 'block';
    if (probDetails) probDetails.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.error) {
            predText.textContent = '⚠️ Error';
            predText.className = 'display-4 text-danger';
            confText.textContent = result.error;
            if (probDetails) probDetails.style.display = 'none';
        } else {
            // Display prediction and confidence
            predText.textContent = result.prediction;
            predText.className = `display-4 ${result.prediction}`;
            const confidencePercent = (result.confidence * 100).toFixed(1);
            confText.textContent = `Confidence: ${confidencePercent}%`;

            // Show detailed probabilities if element exists (optional)
            if (probDetails) {
                probDetails.style.display = 'block';
                probDetails.innerHTML = `
                    <small>Pass probability: ${(result.probabilities.pass * 100).toFixed(1)}%<br>
                    Fail probability: ${(result.probabilities.fail * 100).toFixed(1)}%</small>
                `;
            }

            // Optional: Add animation or color coding
            if (result.prediction === 'PASS') {
                confText.style.color = '#28a745';
            } else {
                confText.style.color = '#dc3545';
            }
        }
    } catch (err) {
        console.error('Fetch error:', err);
        predText.textContent = '❌ Network Error';
        predText.className = 'display-4 text-danger';
        confText.textContent = 'Could not reach server. Is the backend running?';
        if (probDetails) probDetails.style.display = 'none';
    }
});

// Optional: Add real-time validation for training hours (0-100)
document.getElementById('training_hours').addEventListener('input', function() {
    let val = parseFloat(this.value);
    if (val < 0) this.value = 0;
    if (val > 200) this.value = 200;
});

// Optional: Add help text or tooltips dynamically (Bootstrap tooltips)
document.addEventListener('DOMContentLoaded', function() {
    // Add placeholder for probability details if not present in HTML
    const resultDiv = document.getElementById('result');
    if (!document.getElementById('probabilityDetails')) {
        const detailsDiv = document.createElement('div');
        detailsDiv.id = 'probabilityDetails';
        detailsDiv.className = 'mt-2 small text-muted';
        detailsDiv.style.display = 'none';
        resultDiv.appendChild(detailsDiv);
    }
});
