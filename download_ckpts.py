from huggingface_hub import login, hf_hub_download

login()
hf_hub_download("MahmoodLab/CONCH", filename="pytorch_model.bin", local_dir="./model_ckpt/", force_download=True)
