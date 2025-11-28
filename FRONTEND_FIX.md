# Instructions to Fix "Not Found" Error

## Problem
You're running the API-only server (`api/app.py`) which doesn't serve the frontend.

## Solution

### Step 1: Stop the Current Server
1. Go to the terminal/command prompt where the server is running
2. Press `Ctrl+C` to stop it

### Step 2: Start the Correct Server
Run this command instead:

```bash
python server.py
```

Or use the batch file:
```bash
start_server.bat
```

### Step 3: Access the Frontend
Open your browser and go to:
- Local: `http://localhost:5000`
- Network: `http://10.12.98.248:5000`

## Quick Test
After starting `server.py`, test it:
```bash
python test_frontend.py
```

You should see:
- GET / - Status: 200 (OK - Frontend HTML is being served correctly)
- GET /static/css/style.css - Status: 200 (OK - CSS file is being served correctly)
- GET /health - Status: 200 (OK - API is working correctly)

## Difference Between Servers

- `python api/app.py` - API only (no frontend)
- `python server.py` - Full stack (frontend + API)

Make sure you're using `server.py` for the web interface!

