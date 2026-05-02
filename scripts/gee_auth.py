"""
Run this ONCE interactively before any GEE download scripts.
It opens a browser, you log in with your Google account, paste the token back.
Credentials are saved to ~/.config/earthengine/credentials and reused automatically.

Usage:
    python scripts/gee_auth.py
"""

import ee

ee.Authenticate()

# Replace with your Google Cloud project ID that has Earth Engine API enabled.
# Find it at https://console.cloud.google.com → select your project → copy the ID from the top bar.
# If you don't have one: create a project, then enable "Earth Engine API" in APIs & Services.
PROJECT_ID = "gen-lang-client-0293562798"

ee.Initialize(project=PROJECT_ID)
print("GEE auth successful. You can now run the download scripts.")
