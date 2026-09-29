#!/bin/bash
# Start services
docker-compose up -d

# Wait for Ollama to be ready
echo "Waiting for Ollama..."
sleep 10

# Pull models if not already cached
docker-compose exec ollama ollama pull qwen3:latest
docker-compose exec ollama ollama pull qwen3.5:9b
docker-compose exec ollama ollama pull deepseek-r1:8b

echo "deepVerified API is ready"
