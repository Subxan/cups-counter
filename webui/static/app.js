// Main application JavaScript

let statsData = {
    in: 0,
    out: 0,
    net: 0,
    fps: 0.0
};

let sparklineData = [];
const MAX_SPARKLINE_POINTS = 100;

// Update preview image
function updatePreview() {
    const img = document.getElementById('preview-img');
    const timestamp = new Date().getTime();
    img.src = `/live.jpg?t=${timestamp}`;
}

// Update stats from API
async function updateStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        statsData = {
            in: data.in || 0,
            out: data.out || 0,
            net: data.net || 0,
            fps: data.fps || 0.0
        };

        // Update UI
        document.getElementById('stat-in').textContent = statsData.in;
        document.getElementById('stat-out').textContent = statsData.out;
        document.getElementById('stat-net').textContent = statsData.net;
        document.getElementById('stat-fps').textContent = statsData.fps.toFixed(1);

        // Update sparkline
        sparklineData.push(statsData.net);
        if (sparklineData.length > MAX_SPARKLINE_POINTS) {
            sparklineData.shift();
        }
        drawSparkline();

        // Update status
        document.getElementById('status').className = 'status-indicator active';
    } catch (error) {
        console.error('Error updating stats:', error);
        document.getElementById('status').className = 'status-indicator error';
    }
}

// Draw sparkline chart
function drawSparkline() {
    const canvas = document.getElementById('sparkline');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (sparklineData.length < 2) return;

    const min = Math.min(...sparklineData);
    const max = Math.max(...sparklineData);
    const range = max - min || 1;

    ctx.strokeStyle = '#4CAF50';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const stepX = width / (sparklineData.length - 1);
    sparklineData.forEach((value, index) => {
        const x = index * stepX;
        const y = height - ((value - min) / range) * height;
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();
}

// Export CSV
async function exportCSV() {
    try {
        const today = new Date().toISOString().split('T')[0];
        const response = await fetch(`/events?day=${today}`);
        const data = await response.json();

        // Generate CSV
        const csv = generateCSV(data.events || []);
        
        // Download
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${today}_counts.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error exporting CSV:', error);
        alert('Failed to export CSV');
    }
}

function generateCSV(events) {
    const headers = ['timestamp_utc', 'direction', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'];
    const rows = events.map(event => [
        event.ts_utc,
        event.direction,
        event.track_id,
        event.bbox[0],
        event.bbox[1],
        event.bbox[2],
        event.bbox[3],
        event.conf
    ]);

    return [headers, ...rows].map(row => row.join(',')).join('\n');
}

// WebSocket connection (optional - for real-time updates)
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    try {
        const ws = new WebSocket(wsUrl);
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            statsData = {
                in: data.in || 0,
                out: data.out || 0,
                net: data.net || 0,
                fps: data.fps || 0.0
            };
            
            // Update UI
            document.getElementById('stat-in').textContent = statsData.in;
            document.getElementById('stat-out').textContent = statsData.out;
            document.getElementById('stat-net').textContent = statsData.net;
            document.getElementById('stat-fps').textContent = statsData.fps.toFixed(1);
        };

        ws.onerror = (error) => {
            console.warn('WebSocket error:', error);
        };

        ws.onclose = () => {
            // Reconnect after delay
            setTimeout(connectWebSocket, 5000);
        };
    } catch (error) {
        console.warn('WebSocket not available, using polling');
    }
}

// Admin panel functions
async function updateAutocalProposals() {
    try {
        const response = await fetch('/autocal/proposals');
        const data = await response.json();
        const container = document.getElementById('autocal-proposals');
        
        if (!data.candidates || data.candidates.length === 0) {
            container.innerHTML = '<p>No proposals available</p>';
            return;
        }

        container.innerHTML = data.candidates.map((cand, idx) => `
            <div class="proposal-item">
                <p>Line: [${cand.start}] -> [${cand.end}]</p>
                <p>Direction: ${cand.direction}</p>
                <p>Confidence: ${(cand.confidence * 100).toFixed(1)}%</p>
                <button onclick="applyProposal(${idx})" class="btn-small">Apply</button>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error fetching proposals:', error);
    }
}

async function applyProposal(index) {
    try {
        const response = await fetch(`/autocal/apply?index=${index}`, { method: 'POST' });
        const data = await response.json();
        alert('Proposal applied!');
        updateAutocalProposals();
    } catch (error) {
        console.error('Error applying proposal:', error);
        alert('Failed to apply proposal');
    }
}

async function updateDriftStatus() {
    try {
        const response = await fetch('/drift/status');
        const data = await response.json();
        
        document.getElementById('drift-ssim').textContent = data.ssim.toFixed(3);
        document.getElementById('drift-edge-iou').textContent = data.edge_iou.toFixed(3);
        document.getElementById('drift-brightness').textContent = data.brightness_var.toFixed(1);
        document.getElementById('drift-camera').textContent = data.camera_shifted ? 'Yes' : 'No';
        document.getElementById('drift-lighting').textContent = data.lighting_bad ? 'Yes' : 'No';
    } catch (error) {
        console.error('Error fetching drift status:', error);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Update preview every 200ms
    setInterval(updatePreview, 200);

    // Update stats every second
    setInterval(updateStats, 1000);
    updateStats(); // Initial update

    // Export button
    document.getElementById('export-btn').addEventListener('click', exportCSV);

    // Admin panel toggle
    const adminToggle = document.getElementById('admin-toggle');
    const adminPanel = document.getElementById('admin-panel');
    adminToggle.addEventListener('click', () => {
        const isVisible = adminPanel.style.display !== 'none';
        adminPanel.style.display = isVisible ? 'none' : 'block';
        if (!isVisible) {
            updateAutocalProposals();
            updateDriftStatus();
            setInterval(updateDriftStatus, 5000); // Update drift every 5s
        }
    });

    // Try WebSocket
    connectWebSocket();
});

