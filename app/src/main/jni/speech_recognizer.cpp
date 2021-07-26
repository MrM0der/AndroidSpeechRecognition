//
// Created by sergey on 7/11/21.
//

#include <cstring>
#include <algorithm>
#include <string>
#include <regex>
#include <memory>

#include <jni.h>
#include <android/log.h>
#include <onnxruntime_cxx_api.h>
#include <nnapi_provider_factory.h>

#define STRINGIFY(x) STRINGIFY_(x)
#define STRINGIFY_(x) # x

static std::unique_ptr<Ort::Env> ort_env;
static std::unique_ptr<Ort::SessionOptions> session_options;
static std::unique_ptr<Ort::Session> session;


extern "C"
JNIEXPORT void JNICALL
Java_com_example_androidspeechrecognition_SpeechRecognizer_init(JNIEnv *env, jobject thiz,
                                                               jstring modelPath)
{
    const char *model_path = env->GetStringUTFChars(modelPath, nullptr);
    __android_log_print(ANDROID_LOG_INFO, STRINGIFY(APP_NAME), "%s", model_path);

    ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_INFO, "speech_recognizer_ort");
    session_options = std::make_unique<Ort::SessionOptions>();
    uint32_t nnapi_flags = NNAPI_FLAG_CPU_DISABLED;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(*session_options, nnapi_flags));
    session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
//    void* library_handle = nullptr;
//    Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(so, "libcustom_op_library.so", &library_handle));
    session = std::make_unique<Ort::Session>(*ort_env, model_path, *session_options);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_androidspeechrecognition_SpeechRecognizer_infer(JNIEnv *env, jobject thiz,
                                                                 jfloatArray samplesArray) {
    jboolean is_copy;
    float *samples = env->GetFloatArrayElements(samplesArray, &is_copy);
    unsigned samples_len = env->GetArrayLength(samplesArray);
    __android_log_print(ANDROID_LOG_INFO, STRINGIFY(APP_NAME), "%p %d", samples, samples_len);

    for (unsigned i = 0; i < samples_len; ++i) samples[i] *= 1.7147152f;

    std::array<int64_t, 2> input_shape = {1, samples_len};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, samples, samples_len,
            input_shape.data(), input_shape.size());

    auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            (const char*[]){"input_values"}, &input_tensor, 1,
            (const char*[]){"output_0"}, 1);

    auto result = output_tensors.front().GetTensorMutableData<uint64_t>();
    auto result_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    __android_log_print(ANDROID_LOG_INFO, STRINGIFY(APP_NAME), "%ld %ld", result_shape[0], result_shape[1]);
    for (unsigned i = 0; i < result_shape[1]; ++i) {
        __android_log_print(ANDROID_LOG_VERBOSE, STRINGIFY(APP_NAME), "%ld", result[i]);
    }

    std::u16string alphabet(u"|abcefghiklmnoprstxzабвгдежзийклмнопрстуфхцчшщъыьэюя");
    auto res = std::accumulate(result, result + result_shape[1], std::u16string(u""),
                               [&alphabet](auto acc, auto id) {
        if (id < 53)
            return acc += alphabet[id];
        else
            return acc;
    });

    for (std::string::size_type pos = 0, end; pos < res.size(); ++pos) {
        end = res.find_first_not_of(res[pos], pos + 1);
        if (end == std::string::npos) end = res.size();
        if (end == pos + 1) continue;
        res.replace(pos, end - pos, 1, (res[pos] == alphabet[0]) ? u' ' : res[pos]);
    }
    return env->NewString((jchar *)res.c_str(), res.size());
}
