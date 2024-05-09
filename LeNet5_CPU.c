#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define IMAGE_SIZE 32
#define IMAGE_SIZE_1 28
#define IMAGE_SIZE_2 14
#define NUM_CLASSES 10
#define FILTER_SIZE 5
#define NUM_FILTERS_1 6
#define NUM_FILTERS_2 16
#define POOL_SIZE 2

// Structure to hold image data
typedef struct {
    float pixels[IMAGE_SIZE][IMAGE_SIZE];
    int label;
} Image;

// Function to read MNIST image data from file
void read_mnist_images(const char *filename, Image images[], int num_images){
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: could not open file %s\n", filename);
        exit(1);
    }

    // Skip header information
    fseek(file, 16, SEEK_SET);

    // Read image data
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            for (int k = 0; k < IMAGE_SIZE; k++) {
                unsigned char pixel;
                if (fread(&pixel, sizeof(pixel), 1, file) != 1) {
                    printf("Error: could not read image data\n");
                    exit(1);
                }
                images[i].pixels[j][k] = (float)pixel / 255.0; //scale pixel values to [0,1]
            }
        }
    }

    fclose(file);        
}

// Function to read MNIST label data from file
void read_mnist_labels(const char *filename, Image labels[], int num_labels){
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: could not open file %s\n", filename);
        exit(1);
    }

    // Skip header information
    fseek(file, 8, SEEK_SET);

    // Read label data
    for (int i = 0; i < num_labels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(label), 1, file) != 1) {
            printf("Error: could not read label data\n");
            exit(1);
        }
        labels[i].label = (int)label;
    }

    fclose(file);        
}

float relu(float x){
    return x > 0 ? x : 0;
}

//Function to perform convolution operations
void convolution(float input[IMAGE_SIZE][IMAGE_SIZE], float output[NUM_FILTERS_1][IMAGE_SIZE_1][IMAGE_SIZE_1], float filter[NUM_FILTERS_1][FILTER_SIZE][FILTER_SIZE], int stride){
    for (int i = 0; i < NUM_FILTERS_1; i++) {
        for (int j = 0; j < IMAGE_SIZE_1; j+=stride) {
            for (int k = 0; k < IMAGE_SIZE_1; k+=stride) {
                output[i][j][k] = 0;
                for (int l = 0; l < FILTER_SIZE; l++) {
                    for (int m = 0; m < FILTER_SIZE; m++) {
                        output[i][j][k] += input[j + l][k + m] * filter[i][l][m];
                    }
                }
            }
        }
    }
}

//Function to perform max pooling operations
void max_pooling(float input[NUM_FILTERS_1][IMAGE_SIZE_1][IMAGE_SIZE_1], float output[NUM_FILTERS_1][IMAGE_SIZE_1/POOL_SIZE][IMAGE_SIZE_1/POOL_SIZE]){
    for (int i = 0; i < NUM_FILTERS_1; i++) {
        for (int j = 0; j < IMAGE_SIZE_1/POOL_SIZE; j++) {
            for (int k = 0; k < IMAGE_SIZE_1/POOL_SIZE; k++) {
                float max = input[i][j*POOL_SIZE][k*POOL_SIZE];
                for (int l = 0; l < POOL_SIZE; l++) {
                    for (int m = 0; m < POOL_SIZE; m++) {
                        if (input[i][j*POOL_SIZE + l][k*POOL_SIZE + m] > max) {
                            max = input[i][j*POOL_SIZE + l][k*POOL_SIZE + m];
                        }
                    }
                }
                output[i][j][k] = max;
            }
        }
    }
}

// Funnction to perform fully connected layer operations
void fully_connected(float input[NUM_FILTERS_2][IMAGE_SIZE_2][IMAGE_SIZE_2], float weights[NUM_CLASSES][NUM_FILTERS_2*IMAGE_SIZE_2*IMAGE_SIZE_2], float biases[NUM_CLASSES],float output[NUM_CLASSES]){
    for (int i = 0; i < NUM_CLASSES; i++) {
        output[i] = 0;
        for (int j = 0; j < NUM_FILTERS_2; j++) {
            for (int k = 0; k < IMAGE_SIZE_2; k++) {
                for (int l = 0; l < IMAGE_SIZE_2; l++) {
                    output[i] += relu(input[j][k][l] * weights[i][j*IMAGE_SIZE_2*IMAGE_SIZE_2 + k*IMAGE_SIZE_2 + l]);
                }
            }
        }
    }
}

int main(){
    // Read MNIST training images and labels
    Image *train_images = (Image*)malloc(NUM_TRAIN_IMAGES * sizeof(Image));
    int *train_labels = (int*)malloc(NUM_TRAIN_IMAGES * sizeof(int));
    Image *test_images = (Image*)malloc(NUM_TEST_IMAGES * sizeof(Image));
    int *test_labels = (int*)malloc(NUM_TEST_IMAGES * sizeof(int));

    //Read training and test dataset
    read_mnist_images("train-images-idx3-ubyte", train_images, NUM_TRAIN_IMAGES);
    read_mnist_labels("train-labels-idx1-ubyte", train_labels, NUM_TRAIN_IMAGES);
    read_mnist_images("t10k-images-idx3-ubyte", test_images, NUM_TEST_IMAGES);
    read_mnist_labels("t10k-labels-idx1-ubyte", test_labels, NUM_TEST_IMAGES);

    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
        float input_image[IMAGE_SIZE][IMAGE_SIZE];
        for (int row = 0; row < IMAGE_SIZE; row++) {
            for (int col = 0; col < IMAGE_SIZE; col++) {
                input_image[row][col] = train_images[i].pixels[row][col];
            }
        }
    
        float conv1_output[NUM_FILTERS_1][IMAGE_SIZE_1][IMAGE_SIZE_1];
        float pool1_output[NUM_FILTERS_1][IMAGE_SIZE_1/POOL_SIZE][IMAGE_SIZE_1/POOL_SIZE];
        float conv2_output[NUM_FILTERS_2][IMAGE_SIZE_2][IMAGE_SIZE_2];
        float pool2_output[NUM_FILTERS_2][IMAGE_SIZE_2/POOL_SIZE][IMAGE_SIZE_2/POOL_SIZE];
        float flattened_output[NUM_FILTERS_2* 4 * 4];
        float fc_output[NUM_CLASSES];

        // Initialize filters and weights
        float filter_1[NUM_FILTERS_1][FILTER_SIZE][FILTER_SIZE];
        float filter_2[NUM_FILTERS_2][NUM_FILTERS_1][FILTER_SIZE][FILTER_SIZE];
        float weights[NUM_CLASSES][NUM_FILTERS_2* 4 * 4];
        float biases[NUM_CLASSES];

        convolution(input_image, conv1_output, filter_1, 1);
        max_pooling(conv1_output, pool1_output);
        convolution(pool1_output, conv2_output, filter_2, 1);
        max_pooling(conv2_output, pool2_output);
        fully_connected(pool2_output, weights, biases, fc_output);

        int index = 0;
        for(int i = 0; i < NUM_FILTERS_2; i++){
            for(int j = 0; j < 4; j++){
                for(int k = 0; k < 4; k++){
                    flattened_output[index] = pool2_output[i][j][k];
                }
            }
        }

        fully_connected(flattened_output, weights, biases, fc_output);

        //Apply softmax function
        float sum = 0;
        float probabilities[NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; c++) {
            probabilities[c] = exp(fc_output[c]);
            sum += probabilities[i];
        }
        for (int c = 0; c < NUM_CLASSES; c++) {
            probabilities[c] /= sum;
        }
    }

    return 0;
}