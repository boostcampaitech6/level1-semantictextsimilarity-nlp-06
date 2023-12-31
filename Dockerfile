FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN transformer==4.28.0

RUN pip install tqdm boto3 requests regex sentencepiece sacremoses

RUN pip install pandas numpy matplotlib 

WORKDIR /workspace 

CMD ["bash"]
