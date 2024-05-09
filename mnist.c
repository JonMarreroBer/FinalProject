#include <stdio.h>
#include <stdlib.h>

#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define IMAGE_SIZE 28

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

