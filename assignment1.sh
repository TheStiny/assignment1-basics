#!/bin/bash

# 1. Manually set the directory to be safe
export MY_DIR=/home/fast/dokpekpe/Experiments/cs336/assignment1-basics
cd "$MY_DIR"

# 2. Define the exact path to the uv tool
UV_BIN=/home/dokpekpe/.local/bin/uv

# 3. Clear Conda to prevent library conflicts
if command -v conda >/dev/null 2>&1; then
    # Some clusters require 'conda deactivate' multiple times if nested
    conda deactivate
fi

# 4. Sync the environment FIRST
# We use the full path variable $UV_BIN here
#echo "Checking dependencies with uv..."
#$UV_BIN sync

# 5. Activate the environment
# This must happen AFTER sync to ensure .venv exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv/bin/activate not found. Sync might have failed."
    exit 1
fi

#echo "Active Python: $(which python)"

# 6. Run the diagnostic
#python -c "import torch; print(f'Torch Success! Version: {torch.__version__}'); print(f'GPU Available: {torch.cuda.is_available()}')"


# 7. GPU info
nvidia-smi


#$UV_BIN run $MY_DIR/cs336_basics/main.py
$UV_BIN run pytest $MY_DIR/tests/test_train_bpe.py