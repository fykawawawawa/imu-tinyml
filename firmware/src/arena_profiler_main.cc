#include <cstddef>
#include <cstdint>
#include <iostream>

#include "inference/model_contract.h"
#include "inference/model_loader.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"

namespace {

using ModelOpResolver = tflite::MicroMutableOpResolver<1>;

// 分析器使用与生产解释器相同的候选 arena 大小。RecordingMicroInterpreter 会在
// AllocateTensors() 后报告 head、tail 和各类持久分配，便于据此设置合理余量。
alignas(16) std::uint8_t
    g_recording_arena[imu::inference::contract::kTensorArenaSize] = {};

}  // namespace

int main() {
    const imu::inference::ModelLoadResult load_result =
        imu::inference::LoadEmbeddedModel();
    if (load_result.status != imu::inference::ModelLoadStatus::kOk ||
        load_result.model == nullptr) {
        std::cerr << "Arena profiling failed to load the embedded model: "
                  << imu::inference::ModelLoadStatusMessage(load_result.status)
                  << '\n';
        return 1;
    }

    ModelOpResolver resolver;
    if (resolver.AddFullyConnected() != kTfLiteOk) {
        std::cerr << "Arena profiling failed to register FULLY_CONNECTED.\n";
        return 1;
    }

    tflite::RecordingMicroInterpreter interpreter(
        load_result.model,
        resolver,
        g_recording_arena,
        sizeof(g_recording_arena));
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        std::cerr << "Arena profiling AllocateTensors() failed with "
                  << sizeof(g_recording_arena) << " bytes.\n";
        return 1;
    }

    std::cout << "Recording arena capacity: " << sizeof(g_recording_arena)
              << " bytes\n";
    interpreter.GetMicroAllocator().PrintAllocations();
    return 0;
}
