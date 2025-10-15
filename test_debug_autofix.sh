#!/bin/bash

# Test script to demonstrate debug features

echo "=== Testing Autofix Debug Features ==="
echo ""
echo "1. Clearing state and cache..."
rm -f BuildFixAgents/state/.batch_state BuildFixAgents/state/.error_count_cache

echo ""
echo "2. Running with DEBUG=true VERBOSE=true TIMING=true"
echo "   This will show:"
echo "   - Detailed logging with timestamps"
echo "   - Performance metrics for each operation"
echo "   - System resource usage"
echo ""

# Run with all debug features
DEBUG=true VERBOSE=true TIMING=true BATCH_SIZE=5 bash BuildFixAgents/autofix_batch.sh run

echo ""
echo "3. Checking final status..."
bash BuildFixAgents/autofix_batch.sh status