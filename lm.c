#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

// Constants
#define MAX_VOCAB 32000  // Max vocabulary size (LLaMA-like)
#define MAX_SEQ_LEN 512  // Max sequence length
#define MAX_EMBD 768     // Embedding dimension (eg., TinyLlama)
#define MAX_LAYERS 12    // Number of transformer layers

// Quantization block sizes
#define Q4_BITS 4        // 4-bit quantization
#define QK4_0 32
#define QK5_0 32
#define QK8_0 32
#define QK_K 256

// GGFU parsing structures
typedef struct  {
    char magic[4];       // "GGUF"
    uint32_t version;
    uint64_t n_tensors;  // Number of tensors (weights) - should be uint64_t
    uint64_t n_kv;       // Number of key-value metadata pairs - should be uint64_t
} GGUFHeader;

// GGUF value types
typedef enum {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type;

// GGML tensor types
typedef enum {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
} ggml_type;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[4];   // Changed to uint64_t
    ggml_type type;
    uint64_t offset;    // offset in file where tensor data starts
    void *data;
} GGUFTensor;

// Q4_0 quantization block (32 values per block)
typedef struct {
    float d;        // ml.c (scale factor)
    uint8_t qs[QK4_0 / 2];     // Nibbles (4-bit values, 2 per byte)
} block_q4_0;

typedef struct {
    float d;
    float m;
    uint8_t qs[QK4_0 / 2];
} block_q4_1;

typedef struct {
    float d;
    uint8_t qs[QK5_0 / 2];
    uint8_t qh[QK5_0 / 8];
} block_q5_0;

typedef struct {
    float d;
    float m;
    uint8_t qs[QK5_0 / 2];
    uint8_t qh[QK5_0 / 8];
} block_q5_1;

typedef struct {
    float d;
    int8_t qs[QK8_0];
} block_q8_0;

typedef struct {
    float d;
    float m;
    uint8_t qs[QK8_0];
} block_q8_1;

typedef struct {
    float d;
    uint8_t qs[QK_K / 4];
    uint8_t scales[QK_K / 16];
} block_q2_k;

typedef struct {
    float d;
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
} block_q3_k;

typedef struct {
    float d;
    float dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
} block_q4_k;

typedef struct {
    float d;
    uint8_t qs[QK_K / 2];
    uint8_t qh[QK_K / 16];
    int8_t scales[QK_K / 32];
} block_q5_k;

typedef struct {
    float d;
    uint8_t q1[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
} block_q6_k;

typedef struct {
    int8_t qs[QK_K];
    float d;
    uint8_t sc[QK_K / 16];
} block_q8_k;

// Model structure
typedef struct{
    int n_vocab;        // Vocabulary size
    int n_embd;         // Embedding dimension
    int n_layer;        // Number of transformer layer
    int n_head;         // Number of attention heads
    int n_ff;           // Feed-forward dimension
    float *wte;         // Token embeddings (n_vocab x n_embd)
    float *wpe;         // Positional embeddings (max_seq_len x n_embd)
    float *ln_f_w;      // Final layer norm weight
    float *ln_f_b;      // Final layer norm bias
    float **attn_q;     // Attention Q weights (n_layer x n_embd x n_embd)
    float **attn_k;     // Attention K weights (n_layer x n_embd x n_embd)
    float **attn_v;     // Attention V weights (n_layer x n_embd x n_embd)
    float **attn_o;     // Attention output weights (n_layer x n_embd x n_embd)
    float **ffn_gate;   // Feed-forward gate weights (n_layer x n_embd x n_ff)
    float **ffn_up;     // Feed-forward up weights (n_layer x n_embd x n_ff)
    float **ffn_down;   // Feed-forward down weights (n_layer x n_ff x n_embd)
    float **ln_1_w;     // Layer norm 1 weights (n_layer x n_embd)
    float **ln_1_b;     // Layer norm 1 biases (n_layer x n_embd)
    float **ln_2_w;     // Layer norm 2 weights (n_layer x n_embd)
    float **ln_2_b;     // Layer norm 2 biases (n_layer x n_embd)
    char **vocab;       // Vocabulary (n_vocab strings)
} Model;

// Context structure
typedef struct {
    int seq_len;        // Current sequence length
    float *state;       // Hidden state (seq_len x n_embd)
    float *logits;      // Output logits (n_vocab)
    float *key_cache;   // KV cache for keys
    float *val_cache;   // KV cache for values
} Context;

// Utility functions
void* safe_malloc(size_t size){
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed for size: %zu\n", size);
        exit(1);
    }
    memset(ptr, 0, size); // Initialize to zero
    return ptr;
}

void read_exact(FILE *f, void *buf, size_t size){
    if (fread(buf, 1, size, f) != size) {
        fprintf(stderr, "Error reading file: expected %zu bytes\n", size);
        exit(1);
    }
}

//GGUF parsing with proper error handling
GGUFHeader read_gguf_header(FILE *f) {
    GGUFHeader header;
    read_exact(f, header.magic, 4);
    if (strncmp(header.magic, "GGUF", 4) != 0) {
        fprintf(stderr, "Invalid GGUF file: magic = %.4s\n", header.magic);
        exit(1);
    }
    read_exact(f, &header.version, 4);
    read_exact(f, &header.n_tensors, 8);     // 64-bit
    read_exact(f, &header.n_kv, 8);

    printf("GGUF version: %u, tensors: %llu, kv pairs: %llu\n", header.version, (unsigned long long)header.n_tensors, (unsigned long long)header.n_kv);
    return header;
}

char* read_string(FILE *f) {
    uint64_t len;
    read_exact(f, &len, 8); // String length is 64-bit in GGUF
    if (len > 1000000) {
        fprintf(stderr, "String too long: %llu\n", (unsigned long long)len);
        exit(1);
    }
    char *str = safe_malloc(len + 1);
    read_exact(f, str, len);
    str[len] = '\0';
    return str;
}

void skip_gguf_value(FILE *f, gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            fseek(f, 1, SEEK_CUR);
            break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            fseek(f, 2, SEEK_CUR);
            break;

        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            fseek(f, 4, SEEK_CUR);
            break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            fseek(f, 8, SEEK_CUR);
            break;
        case GGUF_TYPE_STRING: {
            char *str = read_string(f);
            free(str);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            gguf_type array_type;
            uint64_t array_len;
            read_exact(f, &array_type, 4);
            read_exact(f, &array_len, 8);
            for (uint64_t i = 0; i < array_len; i++) {
                skip_gguf_value(f, array_type);
            }
            break;
        }
    }
}

GGUFTensor read_tensor_info(FILE *f){
    GGUFTensor tensor;

    // Read tensor name
    tensor.name = read_string(f);

    // Read number of dimensions
    read_exact(f, &tensor.n_dims, 4);
    if (tensor.n_dims > 4) {
        fprintf(stderr, "Error: Tensor %s has too many dimensions: (%u)\n", tensor.name, tensor.n_dims);
        exit(1);
    }
    for (uint32_t i = 0; i < tensor.n_dims; i++) {
        read_exact(f, &tensor.dims[i], 8);  // 64-bit dimensions
    }
    read_exact(f, &tensor.type, 4);
    read_exact(f, &tensor.offset, 8);  // 64-bit offset
    tensor.data = NULL;  // will be loaded later
    return tensor;
}

// Calculate tensor size in bytes
size_t get_tensor_size(const GGUFTensor *tensor) {
    size_t elements = 1;
    for (uint32_t i = 0; i < tensor->n_dims; i++) {
        elements *= tensor->dims[i];
    }

    switch (tensor->type) {
        case GGML_TYPE_F32:
            return elements * sizeof(float);
        case GGML_TYPE_F16:
            return elements * sizeof(uint16_t);
        case GGML_TYPE_Q4_0:
            return ((elements + QK4_0 - 1) / QK4_0) * sizeof(block_q4_0);
        case GGML_TYPE_Q4_1:
            return ((elements + QK4_0 - 1) / QK4_0) * sizeof(block_q4_1);
        case GGML_TYPE_Q5_0:
            return ((elements + QK5_0 - 1) / QK5_0) * sizeof(block_q5_0);
        case GGML_TYPE_Q5_1:
            return ((elements + QK5_0 - 1) / QK5_0) * sizeof(block_q5_1);
        case GGML_TYPE_Q8_0:
            return ((elements + QK8_0 - 1) / QK8_0) * sizeof(block_q8_0);
        case GGML_TYPE_Q8_1:
            return ((elements + QK8_0 - 1) / QK8_0) * sizeof(block_q8_1);
        case GGML_TYPE_Q2_K:
            return ((elements + QK_K - 1) / QK_K) * sizeof(block_q2_k);
        case GGML_TYPE_Q3_K:
            return ((elements + QK_K - 1) / QK_K) * sizeof(block_q3_k);
        case GGML_TYPE_Q4_K:
            return ((elements + QK_K - 1) / QK_K) * sizeof(block_q4_k);
        case GGML_TYPE_Q5_K:
            return ((elements + QK_K - 1) / QK_K) * sizeof(block_q5_k);
        case GGML_TYPE_Q6_K:
            return ((elements + QK_K - 1) / QK_K) * sizeof(block_q6_k);
        case GGML_TYPE_Q8_K:
            return ((elements + QK_K - 1) / QK_K) * sizeof(block_q8_k);
        default:
            fprintf(stderr, "Error: Unsupported tensor type: %d for tensor %s\n", tensor->type, tensor->name);
            exit(1);
    }
}

int main(int argc, char *argv[]) {
    // Check if we get a file name
    if (argc != 2) {
        fprintf(stderr, "Please give a GGUF file! Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    // Get file name
    const char *model_path = argv[1];

    // Open the file
    FILE *f = fopen(model_path, "rb");  // "rb" means read binary
    if (!f) {
        fprintf(stderr, "Error: Could not open file %s!\n", model_path);
        return 1;
    }

    // Read the header
    GGUFHeader header = read_gguf_header(f);

    // Read key-value pair keys
    printf("\nReading key-value pair keys:\n");
    for (uint64_t i = 0; i < header.n_kv; i++) {
        // Read the key string
        char *key = read_string(f);

        // print the key
        printf("Key %llu: %s\n", (unsigned long long)i, key);

        // Read and skip the value
        uint32_t value_type;
        read_exact(f, &value_type, 4);
        skip_gguf_value(f, value_type);

        // Clean up the key
        free(key);
    }

    // Read tensor info
    printf("\nReading tensor information:\n");
    for (uint64_t i = 0; i < header.n_tensors; i++) {
        GGUFTensor tensor = read_tensor_info(f);

        // Print tensor info
        printf("Tensor %llu: %s, types: ", (unsigned long long)i, tensor.name);
        switch (tensor.type) {
            case GGML_TYPE_F32: printf("F32"); break;
            case GGML_TYPE_F16: printf("F16"); break;
            case GGML_TYPE_Q4_0: printf("Q4_0"); break;
            case GGML_TYPE_Q4_1: printf("Q4_1"); break;
            case GGML_TYPE_Q5_0: printf("Q5_0"); break;
            case GGML_TYPE_Q5_1: printf("Q5_1"); break;
            case GGML_TYPE_Q8_0: printf("Q8_0"); break;
            case GGML_TYPE_Q8_1: printf("Q8_1"); break;
            case GGML_TYPE_Q2_K: printf("Q2_K"); break;
            case GGML_TYPE_Q3_K: printf("Q3_K"); break;
            case GGML_TYPE_Q4_K: printf("Q4_K"); break;
            case GGML_TYPE_Q5_K: printf("Q5_K"); break;
            case GGML_TYPE_Q6_K: printf("Q6_K"); break;
            case GGML_TYPE_Q8_K: printf("Q8_K"); break;
            default: printf("Unknown (%u)", tensor.type);
        }
        printf(", shape: [");
        for (uint32_t j = 0; j < tensor.n_dims; j++) {
            printf("%llu", (unsigned long long)tensor.dims[j]);
            if (j < tensor.n_dims - 1) printf(", ");
        }
        printf("], size: %llu bytes\n", (unsigned long long)get_tensor_size(&tensor));
        // Close up
    }

    // Close the file
    fclose(f);

    return 0;
}
