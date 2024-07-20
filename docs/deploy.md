
# Convert To ONNX
we provide a notebook to convert pytorch torch models to onnx format
[onnx_converter](../onnx_trt/onnx_converter.ipynb)

# Convert To TensorRT
we convert onnx files, swin.onnx for example, to TensorRt engings using the following command:
```
cd onnx_trt
./trt_run.sh swin --int8
```

# Evaluate TensorRT engines
```
python eval_trt.py --model=swin --modality=rgb --num_img=1000
```
