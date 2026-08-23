# Anytime GTMP

## Setup:
### Download Anytime Gtmp
```
git clone https://github.com/commalab/anytime_gtmp
cd anytime_gtmp
git submodule update --init --recursive
cd vamp
git checkout benchmark_aorrtc_backend
cd ..
```


### Install Dependencies Anytime Gtmp
```
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ./pyroffi
uv pip install -e ./vamp
uv pip install -r requirements.txt
```