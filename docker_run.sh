docker run --gpus all --rm -it -d --name diflow-ae --shm-size 16G \
  -v ./models:/root/models \
  smarterlsy/diflow-ae:latest \
  sleep infinity