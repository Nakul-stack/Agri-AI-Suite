# Gemini API Key Setup Guide

## Your API Key Has Expired

Your Google Gemini API key has expired and needs to be renewed. Follow these steps:

## Steps to Fix:

1. **Get a new API key:**
   - Visit: https://aistudio.google.com/app/apikey
   - Sign in with your Google account
   - Create a new API key or regenerate your existing one

2. **Create `.env.local` file in the `frontend` directory:**
   - Create a new file named `.env.local` in the `frontend` folder
   - Add the following line (replace `YOUR_API_KEY_HERE` with your actual API key):
   ```
   VITE_GEMINI_API_KEY=YOUR_API_KEY_HERE
   ```

3. **Restart your development server:**
   - Stop the current server (Ctrl+C)
   - Run `npm run dev` again from the `frontend` directory

## Example `.env.local` file:

```
VITE_GEMINI_API_KEY=AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567
```

## Important Notes:

- Never commit `.env.local` to version control (it should be in `.gitignore`)
- The API key is used for pest detection image analysis
- If you're running a production build, make sure to set the environment variable in your hosting platform

## Alternative: Set Environment Variable Directly

If you prefer not to use a `.env.local` file, you can set the environment variable directly:

**Windows PowerShell:**
```powershell
$env:VITE_GEMINI_API_KEY="YOUR_API_KEY_HERE"
npm run dev
```

**Windows Command Prompt:**
```cmd
set VITE_GEMINI_API_KEY=YOUR_API_KEY_HERE
npm run dev
```

**Linux/Mac:**
```bash
export VITE_GEMINI_API_KEY=YOUR_API_KEY_HERE
npm run dev
```
