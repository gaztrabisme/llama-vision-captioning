#!/bin/bash

# Make scripts executable
chmod +x setup.sh
chmod +x run.py

echo "✅ Scripts made executable!"
echo ""
echo "Next steps:"
echo "1. Run setup: ./setup.sh"
echo "2. Run captioning: conda activate llama-vision && python run.py --hf-token YOUR_TOKEN"
