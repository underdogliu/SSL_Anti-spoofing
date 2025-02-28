git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git

conda create -n SSL_Spoofing python=3.7
conda activate SSL_Spoofing

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./
pip install -r requirements.txt