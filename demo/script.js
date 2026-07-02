// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeLiveDemo();
});

// Show spectrogram
document.querySelectorAll(".show-spec-btn").forEach(btn => {
    btn.addEventListener("click", function(){
        const spectrogram =
            this.closest(".audio-item")
            .querySelector(".spectrogram-container");

        if (spectrogram.style.display === "block") {
            spectrogram.style.display = "none";
            this.innerHTML = "›";
            this.style.transform = "rotate(0deg)";
        } 
        else {
            spectrogram.style.display = "block";
            this.innerHTML = "›";
            this.style.transform = "rotate(90deg)";
        }
    });
});

let controller = null;
let isCancelled = false;

function initializeLiveDemo() {
    console.log('Initializing live demo...');
    
    // Initialize process button
    initializeProcessButton();

    // Initialize clear button
    initializeClearButton();
    
    console.log('Live demo initialized');
}

// Process Button Functionality
function initializeProcessButton() {
    const processBtn = document.getElementById('process-audio');
    if (!processBtn) {
        console.error('Process button not found');
        return;
    }
    processBtn.addEventListener('click', processAudio);
}

async function processAudio() {
    try {
        console.log('Processing...');
        isCancelled = false;
        
        currentText = document.getElementById("textInput").value;
        if (!currentText) {
            showError('Please type some text to generate speech audio');
            return;
        }

        // Show processing status
        showProcessingStatus();
        
        console.log('Sending request to API...');
        controller = new AbortController();
        
        // Send request to API
        const response = await fetch('https://divorcee-evaporate-friction.ngrok-free.dev/tts-muong', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'accept': 'application/json'
            },
            body: JSON.stringify({
                message: currentText
            }),
            signal: controller.signal
        });
        
        console.log('Response status:', response.status);
        if (isCancelled) return;
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API request failed: ${response.status} ${response.statusText}\n${errorText}`);
        }
        if (isCancelled) return;

        // Get audio
        const audioBlob = await response.blob();
        
        console.log('Audio received:', {
            size: audioBlob.size,
            type: audioBlob.type
        });
        
        // Show results
        showResults(audioBlob);
        
        console.log('Audio processing complete');
        
    } catch (error) {
        if (error.name === "AbortError") {
            console.log("Request aborted");
            return;
        }

        console.error('Error processing:', error);
        hideProcessingStatus();
        
        let errorMessage = 'Failed to process audio. ';
        
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            errorMessage += 'Network error - please check your internet connection and try again.';
        } else if (error.message.includes('API request failed')) {
            errorMessage += 'Server error - the processing service may be temporarily unavailable. Please try again later.';
        } else {
            errorMessage += error.message;
        }
        showError(errorMessage);
    }
}

// Clear Button Functionality
function initializeClearButton() {
    const clearBtn = document.getElementById('clear-all');
    if (!clearBtn) {
        console.error('Clear button not found');
        return;
    }
    clearBtn.addEventListener('click', clearAll);
}

async function clearAll() {
    console.log('Clearing...');
    if (controller) { controller.abort(); }
    isCancelled = true;
    
    document.getElementById("textInput").value = "";
    
    const processingStatus = document.getElementById('processing-status');
    const demoResults = document.getElementById('demo-results');
    const demoError = document.getElementById('demo-error');
    
    if (processingStatus) processingStatus.style.display = 'none';
    if (demoResults) demoResults.style.display = 'none';
    if (demoError) demoError.style.display = 'none';
}

// UI Helper Functions
function showProcessingStatus() {
    const processingStatus = document.getElementById('processing-status');
    const demoResults = document.getElementById('demo-results');
    const demoError = document.getElementById('demo-error');
    
    if (processingStatus) processingStatus.style.display = 'block';
    if (demoResults) demoResults.style.display = 'none';
    if (demoError) demoError.style.display = 'none';
}

function hideProcessingStatus() {
    const processingStatus = document.getElementById('processing-status');
    if (processingStatus) processingStatus.style.display = 'none';
}

function showResults(audioBlob) {
    hideProcessingStatus();
    
    const demoResults = document.getElementById('demo-results');
    const audio = document.getElementById('result');
    const downloadBtn = document.getElementById('download');
    
    if (!demoResults || !audio) return;
    
    // Set audio sources
    audio.src = URL.createObjectURL(audioBlob);
    
    // Setup download button
    if (downloadBtn) {
        downloadBtn.onclick = () => {
            const url = URL.createObjectURL(audioBlob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `audio_${Date.now()}.wav`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        };
    }
    
    // Show results
    demoResults.style.display = 'block';
    
    // Scroll to results
    demoResults.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function showError(message) {
    hideProcessingStatus();
    
    const demoError = document.getElementById('demo-error');
    const errorMessage = document.getElementById('error-message');
    const demoResults = document.getElementById('demo-results');
    
    if (!demoError || !errorMessage) {
        console.error('Error display elements not found');
        alert(`Error: ${message}`);
        return;
    }
    
    errorMessage.textContent = message;
    demoError.style.display = 'block';
    if (demoResults) demoResults.style.display = 'none';
    
    // Scroll to error
    demoError.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function resetDemo() {
    console.log('Resetting demo...');
    clearAll()
}
