#!/usr/bin/env bash

# Exit immediately on error
set -e

# Create a torchserve model archive file
torch-model-archiver --model-name pathway-ranker --version 1.0 --model-file tree_lstm/model.py --serialized-file trained_model/treeLSTM512-fp2048.pt --handler tree_lstm/handler.py

# Build a torchserve docker image containing the model archive file
docker build -t registry.gitlab.com/mlpds_mit/askcos/askcos-data/pathway-ranker:1.0 .
