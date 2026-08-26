FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY dependencies.txt .
RUN pip install --no-cache-dir -r dependencies.txt

COPY . .

CMD ["python", "-u", "main.py"]