#!/bin/bash
# AutoResearch Credit - Quick Start Script

set -e

echo "========================================================================"
echo "AutoResearch for Credit Risk Modeling - Setup"
echo "========================================================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Prepare data
echo ""
echo "Preparing data..."
if [ ! -f data/cache/processed/X_train.pkl ]; then
    python prepare.py ./reference/train.csv
else
    echo "Checking cache compatibility..."
    # Try to load the cached data to verify it's compatible
    if python -c "import pickle; pickle.load(open('data/cache/processed/X_train.pkl', 'rb'))" 2>/dev/null; then
        echo "✓ Data already prepared"
    else
        echo "⚠ Cache incompatible (pandas version changed), regenerating..."
        rm -rf data/cache/processed/*.pkl
        python prepare.py ./reference/train.csv
    fi
fi

# Run baseline
echo ""
echo "========================================================================"
echo "Running baseline experiment..."
echo "========================================================================"
python train.py

echo ""
echo "========================================================================"
echo "✅ Setup complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Review the baseline results above"
echo "  2. Initialize git: git init"
echo "  3. Start AI research in Claude Code:"
echo "     'Please read program.md and start autonomous research!'"
echo ""
