# Check if there is at least one argument supplied
if [ $# -lt 1 ]; then
    echo "Usage: $0 machine_name compiler_exec"
    exit 1
fi
machine=$1
compiler=$2

# Prompt user about deleting the dumps directory
read -p "Do you want to delete the related sub-dumps directories related to each benchId? (y/n): " delete_dumps_input
if [[ "$delete_dumps_input" =~ ^[Yy]$ ]]; then
    delete_dumps=true
else
    delete_dumps=false
fi

pip install --user argparse pandas colorama pathlib matplotlib numpy seaborn

cd 02
if [ "$delete_dumps" = true ]; then
    bash runme.sh --machine=$machine -d
fi
bash runme.sh --machine=$machine $compiler --auto-tune
bash runme.sh --machine=$machine $compiler
cd ..

cd 07
if [ "$delete_dumps" = true ]; then
    bash runme.sh --machine=$machine -d
fi
bash runme.sh --machine=$machine $compiler --auto-tune
bash runme.sh --machine=$machine $compiler
cd ..

cd 08
if [ "$delete_dumps" = true ]; then
    bash runme.sh --machine=$machine -d
fi
bash runme.sh --machine=$machine $compiler --auto-tune
bash runme.sh --machine=$machine $compiler
cd ..