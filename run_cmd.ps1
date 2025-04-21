# Check the version of cuda installed
#nvidia-smi

#Reinstall PyTorch with CUDA
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126

# Install the environment
#pip install -e .[gradio]

# check if CUDA is available
#python .\demo\torch_cuda.py

python .\demo\app_januspro.py
#python .\demo\app_januspro.py --clean_config
