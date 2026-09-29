#!/bin/bash
echo "Pulling models..."

ollama pull qwen3:latest
ollama pull qwen3.5:9b
ollama pull deepseek-r1:8b

echo "All models pulled successfully"
