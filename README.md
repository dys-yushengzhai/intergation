# intergation
intergation of graph partitioning


path:
sudo snap install --classic code
cd /home/dys
bash Anaconda3-2022.10-Linux-x86_64.sh
sudo dpkg -i com.alibabainc.dingtalk_1.4.0.20829_amd
echo export PATH=\$PATH:/home/dys/anaconda3/bin >> ~/.bashrc
source ~/.bashrc
source activate
conda create -n graph_gpu python==3.10.6
conda activate graph_gpu
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric
pip install networkx
pip install pandas
conda install -c conda-forge pymetis
