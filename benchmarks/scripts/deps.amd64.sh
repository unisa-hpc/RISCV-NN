#!/bin/bash

# Configuration options
SKIP_SPACK_INSTALL=false  # Set to true to skip Spack installation
SKIP_CONDA_INSTALL=false  # Set to true to skip Conda/Miniforge installation

# Set root directory
root_dir=~/riscvnn_rootdir
spack_dir="$root_dir/spack"
conda_dir="$root_dir/miniforge3"
env_file="$root_dir/env.sh"
conda_env_name="py39"

# Error handling function
exit_on_error() {
    echo "Error: $1"
    exit 1
}

# Create root directory if it doesn't exist
mkdir -p "$root_dir" || exit_on_error "Failed to create root directory"

# --- Install Spack ---
if [ "$SKIP_SPACK_INSTALL" = false ]; then
    if [ ! -d "$spack_dir" ]; then
        echo "Spack not found. Installing it in $spack_dir..."
        git clone https://github.com/spack/spack.git "$spack_dir" || exit_on_error "Failed to clone Spack repository"

        SPACK_CONFIG="$HOME/.spack/config.yaml"
        mkdir -p "$HOME/.spack"
        THREADS=$(nproc)
        echo -e "config:\n  jobs: $THREADS" > "$SPACK_CONFIG"
        echo "Updated $SPACK_CONFIG with $THREADS jobs"

        # Source Spack environment setup for subsequent commands
        source "$spack_dir/share/spack/setup-env.sh"

        # Install specified LLVM and GCC versions
        echo "Installing specified LLVM and GCC versions..."

        # Install GCC versions
        spack install gcc@13.3.0 || exit_on_error "Failed to install GCC 13.3.0"
        spack install gcc@14.2.0 || exit_on_error "Failed to install GCC 14.2.0"

        # Install LLVM versions
        spack install llvm@17.0.6 || exit_on_error "Failed to install LLVM 17.0.6"
        spack install llvm@18.1.8 || exit_on_error "Failed to install LLVM 18.1.8"

        echo "Finished installing the required compilers with Spack."
    else
        echo "Spack directory already exists, skipping installation..."
    fi
else
    echo "Skipping Spack installation as per configuration..."
fi

# --- Install Miniforge ---
if [ "$SKIP_CONDA_INSTALL" = false ]; then
    if [ ! -d "$conda_dir" ]; then
        echo "Conda not found. Installing Miniforge in $conda_dir..."

        miniforge_url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"

        # Download and install Miniforge
        wget -O "$root_dir/miniforge.sh" "$miniforge_url" || exit_on_error "Failed to download Miniforge"
        bash "$root_dir/miniforge.sh" -b -p "$conda_dir" || exit_on_error "Miniforge installation failed"
        rm "$root_dir/miniforge.sh"

        # Configure conda to use only conda-forge
        "$conda_dir/bin/conda" config --set channel_priority strict
        "$conda_dir/bin/conda" config --remove channels defaults 2>/dev/null || true
        "$conda_dir/bin/conda" config --add channels conda-forge

        # --- Create Conda Environment with Python 3.9 ---
        echo "Creating conda environment with Python 3.9..."
        "$conda_dir/bin/conda" create -y -n "$conda_env_name" -c conda-forge python=3.9 || exit_on_error "Failed to create Conda environment"

        # --- Install Python Packages ---
        echo "Installing required Python packages in '$conda_env_name'..."
        # Install packages one by one to better handle failures
        for package in jq pathlib colorama; do
            echo "Installing $package..."
            "$conda_dir/bin/conda" run -n "$conda_env_name" conda install -y -c conda-forge "$package" || exit_on_error "Failed to install $package"
        done
        # install these with pip in conda
        "$conda_dir/bin/conda" run -n "$conda_env_name" pip install pandas numpy seaborn matplotlib openpyxl kernel_tuner orjson || exit_on_error "Failed to install pyyaml"
    else
        echo "Conda directory already exists, skipping installation..."
    fi
else
    echo "Skipping Conda installation as per configuration..."
fi

# --- Create env.sh ---
echo "Creating environment setup script at $env_file..."
cat <<'EOF' > "$env_file"
# Environment setup script for Spack and Conda

# Spack setup
if [ -d "$ROOT_DIR/spack" ]; then
    export SPACK_ROOT="$ROOT_DIR/spack"
    export PATH="$SPACK_ROOT/bin:$PATH"
    source "$SPACK_ROOT/share/spack/setup-env.sh"
fi

# Conda setup
if [ -d "$ROOT_DIR/miniforge3" ]; then
    export PATH="$ROOT_DIR/miniforge3/bin:$PATH"
    source "$ROOT_DIR/miniforge3/etc/profile.d/conda.sh"

    # Activate specific environment
    if [ -n "$ROOT_DIR" ]; then
        conda activate py39
    fi
fi
EOF

# Create a wrapper script to set ROOT_DIR dynamically
wrapper_script="$root_dir/activate_env.sh"
cat <<'EOF' > "$wrapper_script"
#!/bin/bash
export ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$ROOT_DIR/env.sh"
EOF
chmod +x "$wrapper_script"

echo "Installation complete. To enable the environment, run:"
echo "    source $wrapper_script"