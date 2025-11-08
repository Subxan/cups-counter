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

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Update preview every 200ms
    setInterval(updatePreview, 200);

    // Update stats every second
    setInterval(updateStats, 1000);
    updateStats(); // Initial update

    // Export button
    document.getElementById('export-btn').addEventListener('click', exportCSV);

    // Try WebSocket
    connectWebSocket();
});

