#include <tensorflow/c/c_api.h>
#include <iostream>

int main() {
    // Initialize TensorFlow
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* sessionOptions = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(sessionOptions, status);

    // Load the saved model
    const char* saved_model_dir = "path_to_your_saved_model_directory";
    const char* tags = "serve";
    const int tags_len = strlen(tags);
    const int ntags = 1;
    TF_Graph* graph = TF_NewGraph();
    TF_Buffer* run_options = NULL;
    TF_Buffer* meta_graph_def = TF_NewBufferFromString(tags, tags_len);
    TF_Session* session = TF_LoadSessionFromSavedModel(sessionOptions, run_options,
        saved_model_dir, &tags, ntags, graph, NULL, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error loading model: " << TF_Message(status) << std::endl;
        return 1;
    }

    float input_data[28][28]; // Input data


    // Create input tensor
    const int64_t in_dims[] = {1, 28, 28, 1}; // Assuming batch size 1 and input shape 28x28x1
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, in_dims, 4, input_data, sizeof(input_data), &Deallocator, nullptr);

    // Prepare input tensor
    TF_Output input_op = {TF_GraphOperationByName(graph, "input_layer"), 0};
    TF_Output input_ops[1] = {input_op};
    TF_Tensor* input_tensors[1] = {input_tensor};

    // Prepare output tensor
    TF_Output output_op = {TF_GraphOperationByName(graph, "keras_tensor_15"), 0};
    TF_Output output_ops[1] = {output_op};
    TF_Tensor* output_tensors[1] = {nullptr}; // Output tensor

    // Run inference
    TF_SessionRun(session, nullptr, input_ops, input_tensors, 1, output_ops, output_tensors, 1, nullptr, 0, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Error running inference: " << TF_Message(status) << std::endl;
        return 1;
    }

    // Process output
    float* output_data = static_cast<float*>(TF_TensorData(output_tensors[0]));
    // Example: Print the predicted probabilities
    std::cout << "Predicted probabilities:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Class " << i << ": " << output_data[i] << std::endl;
    }

    // Cleanup
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(sessionOptions);
    TF_DeleteGraph(graph);
    TF_DeleteBuffer(run_options);
    TF_DeleteBuffer(meta_graph_def);
    TF_DeleteStatus(status);

    return 0;
}
