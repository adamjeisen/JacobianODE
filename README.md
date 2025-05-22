# JacobianODE

To install this environment, run the following commands, ensuring that you are in the JacobianODE directory:

```
conda create -n jacobianode -y python=3.11
conda activate jacobianode
conda install -y lightning pytorch=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge
conda install -y jupyter jupyterlab matplotlib numpy pandas scikit-learn scipy seaborn tqdm cython -c conda-forge
pip install hydra-core wandb OmegaConf hydra-submitit-launcher torchdiffeq sdeint loguru autoray
python -m pip install --editable . 
```