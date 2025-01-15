# Check if there is at least one argument supplied
if [ $# -lt 1 ]; then
    echo "Usage: $0 machine_name"
    exit 1
fi
machine=$1

cd 02
bash runme.sh --machine=$machine --auto-tune
bash runme.sh --machine=$machine
cd ..

cd 07
bash runme.sh --machine=$machine --auto-tune
bash runme.sh --machine=$machine
cd ..

cd 08
bash runme.sh --machine=$machine --auto-tune
bash runme.sh --machine=$machine
cd ..