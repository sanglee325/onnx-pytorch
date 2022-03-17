# onnx-pytorch

## Environment Setting

### Create Conda
* Create conda environment.

    ```bash
    conda create -n onnx-pytorch python=3.8
    conda activate onnx-pytorch
    ```

* Install requirements.txt.

    ```bash
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    ```

* Create directories.
    
    ```bash
    mkdir ckpt
    mkdir data
    ```

## Start ONNX

### Train model and Save models

* Run `train.py`. The trained models will be saved in `./ckpt`

    ```bash
    python train.py
    ```

### Compare ONNX

* Run `compare.py` to compare the ONNX model and general model weight.

    ```bash
    python compare.py
    ```

### PyTorch to Keras

* Run `onnx_to_keras.py` to run torch model as keras.

    ```bash
    python onnx_to_keras.py
    ```