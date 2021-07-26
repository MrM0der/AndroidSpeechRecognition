#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC

MODEL_FILE = './wav2vec2.onnx'

processor = Wav2Vec2Processor.from_pretrained('anton-l/wav2vec2-large-xlsr-53-russian')
pt_inputs = processor(np.linspace(-1, 1, 16000, dtype=np.float32),
                      sampling_rate=16000, return_tensors='pt')
inputs = {
    'input_values': pt_inputs.input_values.numpy().astype(np.float32),
    # 'attention_mask': pt_inputs.attention_mask.numpy().astype(np.int64),
}
input_names = ['input_values', 'attention_mask']
output_names = ['output_0']

class MyWav2Vec2Model(Wav2Vec2ForCTC):
    def forward(self,
                input_values,
                attention_mask=None,
                mask_time_indices=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
    ):
        extract_features = self.wav2vec2.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        hidden_states = self.wav2vec2.feature_projection(extract_features)
        encoder_outputs = self.wav2vec2.encoder(
            hidden_states)#, attention_mask=attention_mask)
        return torch.argmax(self.lm_head(encoder_outputs[0]), dim=-1)

model = MyWav2Vec2Model.from_pretrained('anton-l/wav2vec2-large-xlsr-53-russian')

torch.onnx.export(model, tuple(pt_inputs.values()),
                  MODEL_FILE, verbose=False, opset_version=13,
                  input_names=input_names, output_names=output_names,
                  dynamic_axes={
                      'input_values': {0: 'batch', 1: 'sequence'},
                      'attention_mask': {0: 'batch', 1: 'sequence'}
                  })

# sess = ort.InferenceSession(MODEL_FILE)
# print(sess.run(None, inputs)[0].shape)

