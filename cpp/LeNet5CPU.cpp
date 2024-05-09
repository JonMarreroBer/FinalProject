#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

// Define a struct to hold MNIST images and labels
struct MNISTData {
    std::vector<std::vector<uint8_t>> images;
    std::vector<int> labels;
};

// Function to load MNIST data
MNISTData loadMNISTData(const std::string& imagesPath, const std::string& labelsPath, int numImages) {
    MNISTData mnistData;
    // Read images
    std::ifstream imagesFile(imagesPath, std::ios::binary);
    if (!imagesFile.is_open()) {
        throw std::runtime_error("Unable to open images file: " + imagesPath);
    }
    imagesFile.seekg(16); // Skip the header
    for (int i = 0; i < numImages; ++i) {
        std::vector<uint8_t> image(28 * 28);
        imagesFile.read(reinterpret_cast<char*>(image.data()), 28 * 28);
        for (auto& pixel : image) {
            pixel /= 255.0;
        }
        mnistData.images.push_back(image);
    }
    imagesFile.close();

    std::cout << "Number of images loaded: " << mnistData.images.size() << std::endl;
    std::cout << "Image dimensions: " << mnistData.images[0].size() << std::endl; // Assuming all images have the same dimensions
   
    // Read labels
    std::ifstream labelsFile(labelsPath, std::ios::binary);
    if (!labelsFile.is_open()) {
        throw std::runtime_error("Unable to open labels file: " + labelsPath);
    }
    labelsFile.seekg(8); // Skip the header
    for (int i = 0; i < numImages; ++i) {
        unsigned char label;
        labelsFile.read(reinterpret_cast<char*>(&label), 1);
        mnistData.labels.push_back(label);
    }
    labelsFile.close();

    std::cout << "Number of labels loaded: " << mnistData.labels.size() << std::endl;

    return mnistData;
}

class Linear {
private:
    int in_features, out_features;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

public:
    Linear(int in_features, int out_features) : in_features(in_features), out_features(out_features) {
        // Initialize weights and biases (random initialization)
        weights.resize(out_features, std::vector<double>(in_features));
        biases.resize(out_features);

        // Random initialization of weights and biases
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 0.1);
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                weights[i][j] = distribution(generator);
            }
            biases[i] = distribution(generator);
        }
    }

    std::vector<double> forward(std::vector<double> input) {
        std::vector<double> output(out_features);

        for (int o = 0; o < out_features; ++o) {
            double sum = 0.0;
            for (int i = 0; i < in_features; ++i) {
                sum += input[i] * weights[o][i];
            }
            output[o] = sum + biases[o];
        }

        return output;
    }
};


class ReLU {
public:
    std::vector<double> forward(std::vector<double> input) {
        for (int i = 0; i < input.size(); ++i) {
            input[i] = std::max(0.0, input[i]);
        }
        return input;
    }
};

class Conv2d {
private:
    int in_channels, out_channels, kernel_size, stride, padding;
    std::vector<double> weights;
    std::vector<double> biases;

public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {
        // Initialize weights and biases (random initialization)
        if (padding < kernel_size / 2 || padding < 1) {
            padding = std::max(kernel_size / 2, 1);  // Set padding to at least half of the kernel size or 1
        }
        weights.resize(in_channels * out_channels * kernel_size * kernel_size);
        biases.resize(out_channels);

        // Random initialization of weights and biases
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 0.1);
        for (int i = 0; i < in_channels * out_channels * kernel_size * kernel_size; ++i) {
            weights[i] = distribution(generator);
        }
        for (int i = 0; i < out_channels; ++i) {
            biases[i] = distribution(generator);
        }
    }

    std::vector<double> forward(const std::vector<double>& input, int input_height, int input_width) {
        if (input_height < kernel_size || input_width < kernel_size) {
            throw std::invalid_argument("Input dimensions are smaller than the kernel size.");
        }
        int out_height = (input_height - kernel_size + 2 * padding) / stride + 1;
        int out_width = (input_width - kernel_size + 2 * padding) / stride + 1;
        
        if (out_height <= 0 || out_width <= 0) {
            throw std::invalid_argument("Invalid output dimensions. Adjust padding or stride.");
        }

        std::vector<double> output(out_channels * out_height * out_width);

        for (int o = 0; o < out_channels; ++o) {
            for (int y = 0; y < out_height; ++y) {
                for (int x = 0; x < out_width; ++x) {
                    double sum = biases[o];
                    for (int c = 0; c < in_channels; ++c) {
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int in_y = y * stride - padding + ky;
                                int in_x = x * stride - padding + kx;
                                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                                    int input_index = c * input_height * input_width + in_y * input_width + in_x;
                                    int weight_index = (o * in_channels * kernel_size * kernel_size) + (c * kernel_size * kernel_size) + (ky * kernel_size) + kx;
                                    sum += input[input_index] * weights[weight_index];
                                }
                            }
                        }
                    }
                    output[o * out_height * out_width + y * out_width + x] = sum;
                }
            }
        }

        return output;
    }

    std::vector<double>& parameters() {
        return weights; // Return reference to weights
    }
};

class MaxPool2d {
private:
    int kernel_size, stride;

public:
    MaxPool2d(int kernel_size, int stride) : kernel_size(kernel_size), stride(stride) {}

    std::vector<double> forward(const std::vector<double>& input, int input_channels, int input_height, int input_width) {
        int out_height = (input_height - kernel_size) / stride + 1;
        int out_width = (input_width - kernel_size) / stride + 1;
        std::vector<double> output(input_channels * out_height * out_width);

        for (int c = 0; c < input_channels; ++c) {
            for (int y = 0; y < out_height; ++y) {
                for (int x = 0; x < out_width; ++x) {
                    double max_val = -INFINITY;
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int in_y = y * stride + ky;
                            int in_x = x * stride + kx;
                            max_val = std::max(max_val, input[c * input_height * input_width + in_y * input_width + in_x]);
                        }
                    }
                    output[c * out_height * out_width + y * out_width + x] = max_val;
                }
            }
        }

        return output;
    }
};

class LeNet5 {
private:
    Conv2d conv1, conv2, conv3;
    MaxPool2d maxpool;
    ReLU relu;
    Linear fc1, fc2, fc3, fc4;

public:
    LeNet5()
        : conv1(1, 6, 5, 1, 2),
          conv2(6, 16, 5, 1, 0),
          conv3(16, 120, 5, 1, 0),
          maxpool(2, 2),
          fc1(120 * 5 * 5, 84),
          fc2(84, 50),
          fc3(50, 10),
          fc4(10, 10) {}

    std::vector<double> forward(const std::vector<double>& x) {
        // Apply convolutional layers and pooling
        std::vector<double> output = relu.forward(maxpool.forward(conv1.forward(x, 28, 28), 1, 28, 28));
        output = relu.forward(maxpool.forward(conv2.forward(output, 12, 12), 6, 12, 12));
        output = relu.forward(conv3.forward(output, 4, 4));

        // Flatten the output of the last convolutional layer
        std::vector<double> flattened(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            flattened[i] = output[i];
        }

        // Apply fully connected layers
        output = relu.forward(fc1.forward(flattened));
        output = relu.forward(fc2.forward(output));
        output = relu.forward(fc3.forward(output));
        output = fc4.forward(output);

        return output;
    }
};

class Model {
private:
    LeNet5 model;
    double lr;
    std::vector<double> train_loss;
    std::vector<double> val_loss;
    std::vector<double> train_acc;
    std::vector<double> val_acc;

public:
    Model(LeNet5 model, double learning_rate)
        : model(model), lr(learning_rate) {}

    void train_step(const std::vector<MNISTData>& dataset) {
        std::vector<double> batch_loss;
        std::vector<double> batch_acc;
        for (const auto& data : dataset) {
            auto inputs = data.images;
            auto targets = data.labels;

            for (const auto& input : inputs) {
                std::vector<double> converted_input(input.begin(), input.end()); // Convert input to std::vector<double>
                auto outputs = model.forward(converted_input);

                auto loss_val = cross_entropy_loss(outputs, targets);
                batch_loss.push_back(loss_val);
                batch_acc.push_back(batch_accuracy(outputs, targets));
            }
        }

        train_loss.push_back(std::accumulate(batch_loss.begin(), batch_loss.end(), 0.0) / batch_loss.size());
        train_acc.push_back(std::accumulate(batch_acc.begin(), batch_acc.end(), 0.0) / batch_acc.size());
    }

    void val_step(const std::vector<MNISTData>& dataset) {
        std::vector<double> batch_loss;
        std::vector<double> batch_acc;
        for (const auto& data : dataset) {
            auto inputs = data.images;
            auto targets = data.labels;

            for (const auto& input : inputs) {
                std::vector<double> converted_input(input.begin(), input.end()); // Convert input to std::vector<double>
                auto outputs = model.forward(converted_input);

                auto loss_val = cross_entropy_loss(outputs, targets);
                batch_loss.push_back(loss_val);
                batch_acc.push_back(batch_accuracy(outputs, targets));
            }
        }

        val_loss.push_back(std::accumulate(batch_loss.begin(), batch_loss.end(), 0.0) / batch_loss.size());
        val_acc.push_back(std::accumulate(batch_acc.begin(), batch_acc.end(), 0.0) / batch_acc.size());
    }

    double cross_entropy_loss(const std::vector<double>& outputs, const std::vector<int>& targets) {
        double loss = 0.0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            loss += -std::log(outputs[i] + 1e-9) * (targets[i] == 1 ? 1 : 0);
        }
        return loss / outputs.size();
    }

    double batch_accuracy(const std::vector<double>& outputs, const std::vector<int>& targets) {
        int correct = 0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            int predicted_label = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
            if (predicted_label == targets[i]) {
                correct++;
            }
        }
        return (correct / static_cast<double>(outputs.size())) * 100.0;
    }

    void train(const std::vector<MNISTData>& trainset, const std::vector<MNISTData>& valset, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
            train_step(trainset);
            val_step(valset);
            std::cout << "Train Loss: " << train_loss.back() << " | Train Accuracy: " << train_acc.back() << "%" << std::endl;
            std::cout << "Validation Loss: " << val_loss.back() << " | Validation Accuracy: " << val_acc.back() << "%" << std::endl;
        }
    }
};

int main() {
    // Load MNIST data
    std::string trainImagesPath = "./MNIST/train-images.idx3-ubyte";
    std::string trainLabelsPath = "./MNIST/train-labels.idx1-ubyte";
    std::string testImagesPath = "./MNIST/t10k-images.idx3-ubyte";
    std::string testLabelsPath = "./MNIST/t10k-labels.idx1-ubyte";
   
    int numTrainImages = 60000;
    int numTestImages = 10000;

    MNISTData trainData = loadMNISTData(trainImagesPath, trainLabelsPath, numTrainImages);
    MNISTData testData = loadMNISTData(testImagesPath, testLabelsPath, numTestImages);

    std::cout << "Dimensions of the first training image: " << trainData.images[0].size() << std::endl;

    std::cout << "Pixel values of the first training image:" << std::endl;
    for (double pixel : trainData.images[0]) {
        std::cout << pixel << " ";
    }
    std::cout << std::endl;

    // Normalize pixel values to range [0, 1]
    for (auto& image : trainData.images) {
        for (uint8_t& pixel : image) {
            pixel /= 255.0;
        }
    }

    std::cout << "Pixel values of the first training image after normalization:" << std::endl;
    for (double pixel : trainData.images[0]) {
        std::cout << pixel << " ";
    }
    std::cout << std::endl;

    for (auto& image : testData.images) {
        for (uint8_t& pixel : image) { // Change the type of 'pixel' from 'double' to 'uint8_t'
            pixel /= 255.0;
        }
    }
    // Define LeNet5 model
    LeNet5 model;

    // Define training parameters
    double learning_rate = 1e-4;
    int epochs = 5;

    // Initialize the model and train
    Model lenet_model(model, learning_rate);
    lenet_model.train({trainData}, {testData}, epochs);

    return 0;
}
