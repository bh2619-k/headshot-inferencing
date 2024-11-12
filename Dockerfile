# Include Python
FROM python:3.10.12-buster
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Define your working directory
WORKDIR /

# Add your file
ADD . .

# Install packages
RUN pip install -r requirements.txt

# Call your file when your container starts
CMD [ "python", "./index.py" ]