#!/bin/bash

# BabyAI Environment Server Startup Script
# Starts BabyAI server with the conda environment from setup_agentgym_rl.sh

set -e

echo "🚀 Starting BabyAI Environment Server"
echo "===================================="

# Configuration
ENV_NAME="agentgym-rl"
SERVER_HOST="0.0.0.0"
SERVER_PORT="36005"
SERVER_DIR="/root/AgentGym-RL/AgentGym/agentenv-babyai"

# Check if conda is available and setup
if [ ! -d "/root/miniconda" ]; then
    echo "❌ Conda not found. Please run setup_agentgym_rl.sh first"
    exit 1
fi

# Export conda to PATH and activate environment
export PATH="/root/miniconda/bin:$PATH"
source /root/miniconda/etc/profile.d/conda.sh

# Check if environment exists
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "❌ Conda environment '$ENV_NAME' not found"
    echo "Please run setup_agentgym_rl.sh first to create the environment"
    exit 1
fi

echo "✅ Activating conda environment: $ENV_NAME"
conda activate $ENV_NAME

# Check if BabyAI environment directory exists
if [ ! -d "$SERVER_DIR" ]; then
    echo "❌ BabyAI environment directory not found: $SERVER_DIR"
    echo "Please run setup_agentgym_rl.sh first to install BabyAI environment"
    exit 1
fi

# Check if BabyAI command is available
if ! command -v babyai &> /dev/null; then
    echo "❌ BabyAI command not found"
    echo "Please run setup_agentgym_rl.sh first to install BabyAI environment"
    exit 1
fi

# Check if port is already in use
if lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "❌ Port $SERVER_PORT is already in use"
    echo "Please stop the existing server or use a different port"
    exit 1
fi

echo ""
echo "🌍 Starting BabyAI environment server..."
echo "- Host: $SERVER_HOST"
echo "- Port: $SERVER_PORT"
echo "- Directory: $SERVER_DIR"
echo ""

# Navigate to server directory and start the server
cd "$SERVER_DIR"

# Start the server
babyai --host $SERVER_HOST --port $SERVER_PORT

SERVER_PID=$!

echo "✅ BabyAI server started successfully!"
echo "Server is running at: http://$SERVER_HOST:$SERVER_PORT"
echo "Process ID: $SERVER_PID"
echo ""
echo "To stop the server, press Ctrl+C"
echo ""
echo "You can now run training in a separate terminal with:"
echo "conda activate $ENV_NAME"
echo "cd /root/AgentGym-RL"
echo "bash train_babyai_a40.sh"