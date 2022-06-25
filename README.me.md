```bash
python3 -m venv venv
. ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --pre "torch>1.13.0.dev20220610" "torchvision>0.14.0.dev20220609" --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```