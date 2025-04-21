FROM pytorch/pytorch
WORKDIR /app
COPY . /app
# modfiy cu126 to your own cuda version
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
RUN pip install -e .[gradio]
CMD ["python", "demo/app_januspro.py"]
