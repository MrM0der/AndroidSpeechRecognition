package com.example.androidspeechrecognition

class SpeechRecognizer(modelPath: String) {
    init {
        System.loadLibrary("onnxruntime")
        System.loadLibrary("custom_op_library")
        System.loadLibrary("speech_recognizer")
        init(modelPath)
    }

    external fun init(modelPath: String)

    external fun infer(samples: FloatArray): String
}