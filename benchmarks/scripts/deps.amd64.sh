#!/bin/bash

# Set root directory
root_dir=~/riscvnn_rootdir
spack_dir="$root_dir/spack"
conda_dir="$root_dir/miniconda3"
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
    spack update || exit_on_error "Failed to update Spack"

    # Install LLVM versions
    spack install llvm@17.0.6 || exit_on_error "Failed to install LLVM 17.0.6"
    spack install llvm@18.1.8 || exit_on_error "Failed to install LLVM 18.1.8"

    # Install GCC versions
    spack install gcc@13.2.0 || exit_on_error "Failed to install GCC 13.2.0"
    spack install gcc@14.2.0 || exit_on_error "Failed to install GCC 14.2.0"
fi

# --- Install Miniconda (x86_64 only) ---
if [ ! -d "$conda_dir" ]; then
    echo "Conda not found. Installing Miniconda in $conda_dir..."

    miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

    # Download and install Miniconda
    wget -O "$root_dir/miniconda.sh" "$miniconda_url" || exit_on_error "Failed to download Miniconda"
    bash "$root_dir/miniconda.sh" -b -p "$conda_dir" || exit_on_error "Miniconda installation failed"
    rm "$root_dir/miniconda.sh"
fi

# --- Create Conda Environment with Python 3.9 ---
"$conda_dir/bin/conda" create -y -n "$conda_env_name" python=3.9 || exit_on_error "Failed to create Conda environment"

# --- Install Python Packages ---
echo "Installing required Python packages in '$conda_env_name'..."
"$conda_dir/bin/conda" run -n "$conda_env_name" conda install -y argparse pandas jq numpy seaborn matplotlib pathlib colorama || exit_on_error "Failed to install Python packages"

# --- Create env.sh ---
echo "Creating environment setup script at $env_file..."
cat <<'EOF' > "$env_file"
# Environment setup script for Spack and Conda

# Spack setup
export SPACK_ROOT="$ROOT_DIR/spack"
export PATH="$SPACK_ROOT/bin:$PATH"
source "$SPACK_ROOT/share/spack/setup-env.sh"

# Conda setup
export PATH="$ROOT_DIR/miniconda3/bin:$PATH"
source "$ROOT_DIR/miniconda3/etc/profile.d/conda.sh"

# Activate specific environment
if [ -n "$ROOT_DIR" ] && [ -d "$ROOT_DIR/miniconda3" ]; then
    conda activate py39
else
    echo "Error: ROOT_DIR not set or Miniconda directory not found"
    return 1
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

echo "Installation complete. To enable Spack and Conda, run:"
echo "    source $wrapper_script"