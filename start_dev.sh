#!/bin/bash

# Power Theft Detection System - Development Server
# Auto-restarts on file changes

echo "🚀 Starting Power Theft Detection Development Server"
echo "=================================================="

# Kill any existing servers on port 8105
echo "🔍 Checking for existing servers..."
lsof -ti:8105 | xargs kill -9 2>/dev/null || true

# Start the development server with auto-reload
echo "🎯 Starting server with auto-reload..."
python3 app.py

echo "✅ Development server stopped"
