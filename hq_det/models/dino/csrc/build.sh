#!/bin/bash
# Build script for asymmetric focal loss CUDA extension

set -e

echo "Building asymmetric focal loss CUDA extension..."

# Get PyTorch paths
PYTHON=${PYTHON:-python}
TORCH_INCLUDE=$($PYTHON -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])")
TORCH_LIB=$($PYTHON -c "import torch; print(torch.utils.cpp_extension.library_paths()[0])")

echo "Torch include: $TORCH_INCLUDE"
echo "Torch lib: $TORCH_LIB"

# Build with setuptools
$PYTHON setup.py build_ext --inplace

echo "Build complete!"
echo ""
echo "To install system-wide, run:"
echo "  python setup.py install"
