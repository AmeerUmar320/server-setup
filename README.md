# server-setup
Repo containing the script to setup locate-anything web socket server on server.
# command to set it up on server
curl -sL https://raw.githubusercontent.com/AmeerUmar320/server-setup/main/setup.sh | bash

# command to start the server
cd /workspace && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uvicorn server:app --host 0.0.0.0 --port 8000
