# Cups Counter - Complete Operation Guide

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [First-Time Configuration](#first-time-configuration)
3. [Running the Application](#running-the-application)
4. [Using Auto-Calibration](#using-auto-calibration)
5. [Manual Calibration](#manual-calibration)
6. [Web Dashboard](#web-dashboard)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Initial Setup

### Step 1: Install System Dependencies

On Raspberry Pi 5:
```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install required system packages
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libcap-dev \
    libcamera-dev \
    python3-libcamera

# For Hailo (if using Hailo HAT)
# Follow HailoRT installation instructions from Hailo
```

### Step 2: Setup Python Environment

```bash
cd ~/cups-counter

# Create virtual environment
make setup
# OR manually:
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If `picamera2` installation fails, install it system-wide:
```bash
sudo apt-get install -y python3-picamera2
```

### Step 3: Prepare Model File

Place your Hailo-compiled `.hef` model in the `models/` directory:
```bash
cp /path/to/your/model.hef models/yolov8n_coco.hef
```

Or update the path in `configs/site-default.yaml`:
```yaml
model:
  hef_path: "./models/your_model.hef"
```

---

## First-Time Configuration

### Step 1: Edit Configuration File

Open `configs/site-default.yaml` and adjust:

**Camera Settings:**
```yaml
camera:
  source: "picamera2"    # Use "opencv" for USB cameras
  width: 1280
  height: 720
  fps: 30
  rotate: 0              # 0, 90, 180, or 270 if camera is rotated
```

**For USB Camera:**
```yaml
camera:
  source: "opencv"
  index: 0               # Usually 0, try 1, 2 if multiple cameras
```

**Model Settings (if using mock mode for testing):**
```yaml
model:
  mock_mode: true        # Set to true to test without Hailo hardware
```

**Auto-Calibration Settings:**
```yaml
autocal:
  enabled: true
  warmup_seconds: 120    # How long to observe before proposing line
  auto_apply_if_confident: false  # Set true to auto-apply best line
```

---

## Running the Application

### Option 1: Development Mode (Interactive)

```bash
# Activate virtual environment
source venv/bin/activate

# Run with default config
python apps/edge_service.py

# Or use Makefile
make run

# Run with mock mode (no hardware needed)
python apps/edge_service.py --mock

# Run with custom config
python apps/edge_service.py --config configs/camera-topdown.yaml

# Run headless (no overlay drawing, better performance)
python apps/edge_service.py --headless
```

### Option 2: Production Mode (Systemd Service)

```bash
# Copy service file
sudo cp ops/cups-counter.service /etc/systemd/system/

# Edit service file to match your installation path
sudo nano /etc/systemd/system/cups-counter.service
# Update WorkingDirectory and ExecStart paths if needed

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable cups-counter
sudo systemctl start cups-counter

# Check status
sudo systemctl status cups-counter

# View logs
sudo journalctl -u cups-counter -f
```

---

## Using Auto-Calibration

### How Auto-Calibration Works

1. On startup, if `autocal.enabled: true`, the system:
   - Collects frames for `warmup_seconds` (default: 120 seconds = 2 minutes)
   - Builds a motion heatmap from detected cup movements
   - Analyzes flow direction (bar → counter or counter → bar)
   - Uses Hough line detection to find candidate counting lines
   - Scores candidates and returns top 3 proposals

2. If `auto_apply_if_confident: true` and confidence > 70%:
   - Automatically applies the best line
   - Updates config file
   - Starts counting immediately

3. If `auto_apply_if_confident: false`:
   - Proposals are stored for manual review
   - Access via Web UI Admin Panel

### Using Auto-Calibration

**Method 1: Automatic (Recommended for Lab/Testing)**
```yaml
# In configs/site-default.yaml
autocal:
  enabled: true
  auto_apply_if_confident: true  # Auto-apply best line
```

**Method 2: Manual Review (Recommended for Production)**
```yaml
autocal:
  enabled: true
  auto_apply_if_confident: false  # Review proposals first
```

Then:
1. Start the service: `python apps/edge_service.py`
2. Wait 2 minutes for warmup
3. Open Web UI: `http://<pi-ip>:8080`
4. Click "Admin" button
5. Review proposals in "Auto-Calibration Proposals" section
6. Click "Apply" on the best line

### Debug Outputs

If `autocal.save_debug: true`, check `debug/` directory:
- `motion_heatmap.png` - Shows where cups moved
- `edge_map.png` - Edge detection used for line finding

---

## Manual Calibration

If you prefer to set the line manually:

```bash
# Activate venv
source venv/bin/activate

# Run calibration tool
python apps/calibrate.py

# Or with custom config
python apps/calibrate.py --config configs/camera-topdown.yaml
```

**Steps:**
1. A window opens showing live camera feed
2. Click two points to define the counting line
   - First click: Start point
   - Second click: End point
3. A green line appears connecting the points
4. System runs a 2-minute test with mock detections
5. Line coordinates are saved to config file
6. Screenshot saved to `debug/calib.png`

**Tips:**
- Place line where cups cross (e.g., bar to counter)
- Line should be horizontal or near-horizontal
- Avoid placing too close to image edges
- Press 'q' to quit early

---

## Web Dashboard

### Accessing the Dashboard

1. Start the edge service
2. Open browser: `http://<raspberry-pi-ip>:8080`
   - Replace `<raspberry-pi-ip>` with your Pi's IP address
   - Find IP: `hostname -I` or `ip addr`

### Dashboard Features

**Main View:**
- **Live Preview**: Updates every 200ms showing annotated frame
- **Counters**: IN, OUT, NET counts in large display
- **FPS**: Current frames per second
- **Sparkline**: Net count trend over time
- **Export Button**: Download today's CSV

**Admin Panel** (Click "Admin" button):
- **Auto-Calibration Proposals**: 
  - View proposed lines with confidence scores
  - Click "Apply" to accept a proposal
- **Drift Status**:
  - SSIM: Structural similarity (1.0 = perfect, <0.9 = drift)
  - Edge IoU: Edge map overlap
  - Brightness: Current brightness variance
  - Camera Shifted: Yes/No indicator
  - Lighting Bad: Yes/No indicator

### API Endpoints

You can also access data programmatically:

```bash
# Get current stats
curl http://localhost:8080/stats

# Get health status
curl http://localhost:8080/healthz

# Get drift status
curl http://localhost:8080/drift/status

# Get auto-cal proposals
curl http://localhost:8080/autocal/proposals

# Apply proposal #0
curl -X POST "http://localhost:8080/autocal/apply?index=0"
```

### Prometheus Metrics

Metrics available at: `http://<pi-ip>:9109/metrics`

Key metrics:
- `cups_in_total` - Total cups counted in
- `cups_out_total` - Total cups counted out
- `fps` - Current FPS
- `drift_ssim` - Drift SSIM value
- `autocal_runs_total` - Number of auto-calibration runs
- `recalibrations_total` - Number of drift-triggered recalibrations

---

## Monitoring & Maintenance

### Daily Operations

**Check Service Status:**
```bash
# If using systemd
sudo systemctl status cups-counter

# Check logs
sudo journalctl -u cups-counter -n 100
```

**View Logs:**
```bash
# Application log file
tail -f cups-counter.log

# Systemd logs
sudo journalctl -u cups-counter -f
```

**Check Database:**
```bash
sqlite3 data/cups.db "SELECT COUNT(*) FROM cup_events WHERE DATE(ts_utc) = DATE('now');"
```

### CSV Exports

CSV files are automatically exported daily at the time specified in config:
```yaml
storage:
  export_daily_time: "23:59"  # Exports at 11:59 PM
```

Files saved to: `data/exports/YYYY-MM-DD_counts.csv`

**Manual Export:**
```bash
# Run nightly jobs manually
python apps/nightly_jobs.py
```

### Parameter Tuning

The system automatically tunes parameters nightly (default: 3:10 AM):

1. Records a 90-second clip
2. Tests all parameter combinations from grid
3. Scores each combination
4. Saves best profile to override file

**View Tuning Results:**
- Check logs for tuning results
- Best parameters saved to config override (if `keep_best_profile: true`)

**Manual Tuning:**
```bash
# Process a video file
python apps/backfill_replay.py video.mp4 --output results.csv
```

### Drift Monitoring

The system continuously monitors:
- **Camera Shift**: Detects if camera moved (SSIM < 0.90)
- **Lighting Changes**: Detects glare/darkness (brightness variance)

**When Drift is Detected:**
- If `drift.re_calibrate_on_drift: true`:
  - Auto-calibration runs automatically
  - New line is applied
  - Reference frame is updated
  - Cooldown period prevents rapid re-calibration

**Check Drift Status:**
- Web UI Admin Panel → Drift Status section
- API: `curl http://localhost:8080/drift/status`
- Metrics: `drift_ssim`, `edge_iou`, `brightness_var` gauges

---

## Troubleshooting

### Camera Not Found

**Symptoms:** Error "Failed to open camera" or "picamera2 not installed"

**Solutions:**
```bash
# Test camera
libcamera-hello --list-cameras

# For USB camera, try different index
# Edit config: camera.index: 1 or 2

# Check permissions
sudo usermod -a -G video $USER
# Log out and back in
```

### Low FPS

**Symptoms:** FPS < 15, choppy video

**Solutions:**
1. Reduce resolution in config:
   ```yaml
   camera:
     width: 640
     height: 480
   ```

2. Disable overlays:
   ```yaml
   ui:
     draw_overlays: false
   ```

3. Or run headless:
   ```bash
   python apps/edge_service.py --headless
   ```

4. Check CPU temperature:
   ```bash
   vcgencmd measure_temp
   ```

### Auto-Calibration Not Working

**Symptoms:** No proposals generated, or proposals seem wrong

**Solutions:**
1. Ensure cups are moving during warmup period
2. Check debug outputs in `debug/` directory
3. Increase warmup time:
   ```yaml
   autocal:
     warmup_seconds: 180  # 3 minutes
   ```
4. Verify camera view shows the counting area clearly

### Drift False Positives

**Symptoms:** System recalibrating too often

**Solutions:**
1. Adjust thresholds:
   ```yaml
   drift:
     ssim_threshold: 0.85  # Lower = less sensitive
     min_minutes_between_recal: 120  # Longer cooldown
   ```

2. Disable auto-recalibration:
   ```yaml
   drift:
     re_calibrate_on_drift: false
   ```

### Web UI Not Accessible

**Symptoms:** Can't connect to `http://<ip>:8080`

**Solutions:**
1. Check service is running:
   ```bash
   ps aux | grep edge_service
   ```

2. Check firewall:
   ```bash
   sudo ufw allow 8080
   ```

3. Verify IP address:
   ```bash
   hostname -I
   ```

4. Check if port is in use:
   ```bash
   sudo netstat -tulpn | grep 8080
   ```

### Database Locked Errors

**Symptoms:** "database is locked" errors

**Solutions:**
1. Ensure WAL mode is enabled:
   ```yaml
   storage:
     wal_mode: true
   ```

2. Check disk space:
   ```bash
   df -h
   ```

3. Check file permissions:
   ```bash
   ls -la data/cups.db
   chmod 664 data/cups.db
   ```

### Mock Mode Testing

To test without hardware:

```bash
# Set mock mode in config
# OR use flag
python apps/edge_service.py --mock

# Mock mode generates synthetic cup detections
# Perfect for testing the full pipeline
```

---

## Quick Reference Commands

```bash
# Setup
make setup
source venv/bin/activate

# Run
make run                    # Normal mode
python apps/edge_service.py --mock  # Mock mode
python apps/edge_service.py --headless  # No overlays

# Calibration
make calibrate              # Manual calibration
python apps/auto_calibrate.py  # Standalone auto-cal

# Testing
make test                   # Run all tests
pytest tests/ -v            # Verbose test output

# Maintenance
python apps/nightly_jobs.py  # Run nightly jobs manually
python apps/backfill_replay.py video.mp4  # Process video

# Service Management
sudo systemctl start cups-counter
sudo systemctl stop cups-counter
sudo systemctl restart cups-counter
sudo journalctl -u cups-counter -f
```

---

## Best Practices

1. **First Run:**
   - Use manual calibration to set initial line
   - Verify counting works correctly
   - Then enable auto-calibration for maintenance

2. **Production Deployment:**
   - Set `autocal.auto_apply_if_confident: false` for safety
   - Review proposals before applying
   - Monitor drift status regularly

3. **Performance:**
   - Use ROI masking (enabled by default) to improve FPS
   - Run headless if you don't need visual overlays
   - Adjust resolution based on your needs

4. **Monitoring:**
   - Check `/healthz` endpoint regularly
   - Monitor Prometheus metrics
   - Review logs for errors

5. **Backup:**
   - Regularly backup `data/cups.db`
   - Keep config files in version control
   - Export CSVs for external analysis

---

## Example Workflow

**Day 1 - Initial Setup:**
```bash
# 1. Install dependencies
make setup

# 2. Configure camera/model paths
nano configs/site-default.yaml

# 3. Manual calibration
python apps/calibrate.py

# 4. Test run
python apps/edge_service.py --mock

# 5. Verify web UI works
# Open http://localhost:8080
```

**Day 2 - Production:**
```bash
# 1. Enable auto-calibration (manual review)
# Edit config: autocal.auto_apply_if_confident: false

# 2. Start as service
sudo systemctl start cups-counter

# 3. Monitor via web UI
# Check Admin panel for proposals

# 4. Apply best proposal when ready
```

**Ongoing:**
- Check web dashboard daily
- Review drift status weekly
- Export CSVs for reconciliation
- Monitor logs for errors

---

This guide covers the main operations. The system is designed to be self-managing with auto-calibration and drift monitoring, but manual oversight is recommended for production use.

