#!/usr/bin/env python3
import numpy as np
import onnx
import librosa
from onnxruntime.quantization import (
    quantize_dynamic, quantize_static,
    CalibrationDataReader, QuantFormat,
)


class Wav2Vev2DataReader(CalibrationDataReader):
    def __init__(self):
        self.data = [
            # -1.7147152 * np.ones((1, 16000), dtype=np.float32),
            # np.linspace(-1.7147152, 1.7147152, 16000, dtype=np.float32).reshape(1, -1),
            # 1.7147152 * np.ones((1, 16000), dtype=np.float32),
            # (1.7147152 * np.fromfile('/tmp/x.txt', sep='\n', dtype=np.float32)).reshape(1, -1),
            (1.7147152 * librosa.resample(*librosa.load('/home/sergey/Downloads/554c1b4c-8058-4680-8caa-ed3f035ef437.webm'), 16000)).reshape(1, -1)
        ]
        self._iter = iter(self.data)

    def get_next(self):
        item = next(self._iter, None)
        if item is not None:
            return dict(input_values=item)


if __name__ == '__main__':
    model_fp32 = './wav2vec2.onnx'
    model_quant = './wav2vec2-quant.onnx'
    dr = Wav2Vev2DataReader()
    quantize_dynamic(model_fp32, model_quant)
    # quantize_static(model_fp32, model_quant, dr, quant_format=QuantFormat.QDQ)
        
