#pragma once

#include "tensorflow/lite/micro/micro_interpreter.h"

namespace imu::inference {

enum class InterpreterInitStatus {
    kOk,
    kModelLoadFailed,
    kOperatorRegistrationFailed,
    kTensorAllocationFailed,
    kTensorCountMismatch,
    kInputContractMismatch,
    kOutputContractMismatch,
};

// Initializes the process-wide TFLM interpreter and validates its input and
// output tensors against model_contract.h. Call this before accessing tensors.
InterpreterInitStatus InitializeInterpreter();

const char* InterpreterInitStatusMessage(InterpreterInitStatus status);

// These accessors return nullptr until InitializeInterpreter succeeds. The
// returned objects are owned by this module and must not be deleted.
tflite::MicroInterpreter* GetInterpreter();
TfLiteTensor* GetInputTensor();
TfLiteTensor* GetOutputTensor();

}  // namespace imu::inference
