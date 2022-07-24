#include "NvInfer.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

// A logger is REQUIRED for TensorRT
// Ref: https://github.com/cyrusbehr/tensorrt-cpp-api
using Severity = nvinfer1::ILogger::Severity;
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;
};

void Logger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

Logger logger;

// A helper function to calcualte memory useage
size_t get_memeory_size(const nvinfer1::Dims& dims, const int32_t elem_size)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
}

// A struct to store the detection results
struct Result {
    float score;
    cv::Rect box;
    int class_id;
};

int run(std::string engile_file_path, std::string image_file_path)
{
    /* PART I: Initialize the YOLOV5 model */

    // 读取engine文件载入模型
    std::ifstream engine_file(engile_file_path, std::ios::binary);
    if (engine_file.fail()) {
        std::cout << "Failed to read model file." << std::endl;
        return -1;
    }
    engine_file.seekg(0, std::ifstream::end);
    auto fsize = engine_file.tellg();
    engine_file.seekg(0, std::ifstream::beg);
    std::vector<char> engineData(fsize);
    engine_file.read(engineData.data(), fsize);

    std::unique_ptr<nvinfer1::IRuntime> runtime { nvinfer1::createInferRuntime(logger) };
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine { runtime->deserializeCudaEngine(engineData.data(), fsize) };
    if (mEngine.get() == nullptr) {
        std::cout << "Failed to deserialize CUDA engine." << std::endl;
        return -1;
    }

    // Create execution context
    std::unique_ptr<nvinfer1::IExecutionContext> context { mEngine->createExecutionContext() };
    if (context.get() == nullptr) {
        std::cout << "Failed to create CUDA context." << std::endl;
        return -1;
    }

    // Keep track of 1 input and 4 output device memory buffers
    void* bindings[5];

    // Allocate CUDA memory for input bindings
    const char* input_name = "images";
    int32_t input_idx = mEngine->getBindingIndex(input_name);
    if (input_idx == -1) {
        std::cout << "ERROR: failed to get input by name: " << input_name << std::endl;
        return -1;
    }

    int32_t channels = 3, height = 640, width = 640;
    nvinfer1::Dims4 input_dims { 1, channels, height, width };
    context->setBindingDimensions(input_idx, input_dims);

    size_t input_mem_size = get_memeory_size(input_dims, sizeof(float));
    void* cuda_mem_input { nullptr };
    if (cudaMalloc(&cuda_mem_input, input_mem_size) != cudaSuccess) {
        std::cout << "ERROR: input cuda memory allocation failed, size = " << input_mem_size << " bytes" << std::endl;
        return -1;
    }
    bindings[0] = cuda_mem_input;

    // Allocate CUDA memory for output bindings
    std::vector<std::string> output_node_names { "339", "392", "445", "output" };
    std::vector<size_t> output_mem_sizes;
    bool output_mem_inited = true;
    for (size_t i = 0; i < output_node_names.size(); i++) {
        int32_t output_idx = mEngine->getBindingIndex(output_node_names[i].c_str());
        if (output_idx == -1) {
            std::cout << "ERROR: failed to get output by name: " << output_node_names[i] << std::endl;
            output_mem_inited = false;
            break;
        }
        auto output_dims = context->getBindingDimensions(output_idx);
        auto output_size = get_memeory_size(output_dims, sizeof(float));
        output_mem_sizes.push_back(output_size);
        void* cuda_mem_output { nullptr };
        if (cudaMalloc(&cuda_mem_output, output_size) != cudaSuccess) {
            std::cout << "ERROR: output cuda memory allocation failed, size = " << output_size << " bytes" << std::endl;
            output_mem_inited = false;
            break;
        } else {
            bindings[1 + i] = cuda_mem_output;
        }
    }

    // 一旦输出内存申请失败怎么办
    if (!output_mem_inited) {
        for (auto p : bindings) {
            cudaFree(p);
        }
        return -1;
    }

    // Create a CUDA stream
    cudaStream_t stream { nullptr };
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cout << "ERROR: cuda stream creation failed." << std::endl;
        for (auto p : bindings) {
            cudaFree(p);
        }
        return -1;
    }

    /* PART II: Prepare for model input */

    // Read in an image with OpenCV
    cv::Mat img_bgr = cv::imread(image_file_path);

    // Input preprocessing
    cv::resize(img_bgr, img_bgr, cv::Size(width, height));
    float input_buffer[height * width * channels] { 0 };
    for (int c = 0; c < channels; c++) {
        for (int j = 0, HW = height * width; j < HW; ++j) {
            input_buffer[c * HW + j] = static_cast<float>(img_bgr.data[j * channels + 2 - c]) / 255.0f;
        }
    }

    // Memeory copy: CPU-MEM to GPU-MEM
    if (cudaMemcpyAsync(cuda_mem_input, input_buffer, input_mem_size, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::cout << "ERROR: CUDA memory copy of input failed, size = " << input_mem_size << " bytes" << std::endl;
        return -1;
    }

    /* PART III: Run inference */

    // 异步执行推演
    bool status = context->enqueueV2(bindings, stream, nullptr);
    if (!status) {
        std::cout << "ERROR: TensorRT inference failed." << std::endl;
        return -1;
    }

    // 同步结果
    cudaStreamSynchronize(stream);

    /* PART IV: Model outputs postpreocess. */

    // 分配输出内存空间: "339", "392", "445", "output"
    std::vector<float*> output_buffers;
    for (size_t i = 0; i < output_mem_sizes.size(); i++) {
        float* buf = new float[output_mem_sizes[i] / sizeof(float)];
        output_buffers.push_back(buf);
    }

    // Memory copy: GPU-MEM to CPU-MEM
    for (size_t i = 0; i < output_mem_sizes.size(); i++) {
        auto mem_to_host_result = cudaMemcpyAsync(output_buffers[i], bindings[1 + i], output_mem_sizes[i], cudaMemcpyDeviceToHost, stream);
        if (mem_to_host_result != cudaSuccess) {
            std::cout << "ERROR: CUDA memory copy of output " << i << " failed, size = " << output_mem_sizes[i] << " bytes" << std::endl;
            return -1;
        }
    }

    // YOLOV5后处理
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    float* p = output_buffers[3];
    int step = 85, proposal_count = 25200;
    float score_threshold = 0.5, nms_theshold = 0.45;
    float scale = 1.0;
    for (size_t i = 0; i < proposal_count; i++) {

        // What happens if the confidence is lower than score threshold?
        float obj_score = p[4];
        if (obj_score < score_threshold) {
            p += step;
            continue;
        }

        // The type of the object?
        int c_id = -1;
        float c_score = 0;
        for (size_t j = 5; j < step; j++) {
            if (p[j] > c_score) {
                c_score = p[j];
                c_id = j - 5;
            }
        }

        scores.push_back(c_score * obj_score);
        class_ids.push_back(c_id);
        boxes.push_back(cv::Rect((p[0] - p[2] / 2) / scale, (p[1] - p[3] / 2) / scale,
            p[2] / scale, p[3] / scale));

        p += step;
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_theshold, indices);

    // Collect the detection result
    std::vector<Result> results;
    for (auto i : indices) {
        results.push_back(Result { scores[i], boxes[i], class_ids[i] });
    }

    // Draw all the results in the image
    for (auto& r : results) {
        cv::rectangle(img_bgr, r.box, cv::Scalar(0, 255, 255), 2);
    }

    // Save it
    cv::imwrite("result.jpg", img_bgr);

    return 0;
}

int main(int argc, char const* argv[])
{
    if (argc != 3) {
        std::cout << "Run like this:\n    " << argv[0] << "yolov5s.engine input.jpg" << std::endl;
        return -1;
    }

    run(argv[1], argv[2]);

    return 0;
}
