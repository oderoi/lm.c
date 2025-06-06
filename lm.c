#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>

// GGUF Constants
#define GGUF_MAGIC 0x46554747  // "GGUF" in little-endian
#define GGUF_VERSION 3
#define ALIGNMENT 32

// GGML Types enum
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_COUNT,
} ggml_type;

// GGUF Metadata Value Types
typedef enum {
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
} gguf_metadata_value_type;

// GGUF String structure
typedef struct {
    uint64_t len;
    char *string;
} gguf_string_t;

// Forward declaration for recursive structure
typedef struct gguf_metadata_value gguf_metadata_value_t;

// GGUF Metadata Value union
struct gguf_metadata_value {
    union {
        uint8_t uint8;
        int8_t int8;
        uint16_t uint16;
        int16_t int16;
        uint32_t uint32;
        int32_t int32;
        float float32;
        uint64_t uint64;
        int64_t int64;
        double float64;
        bool bool_;
        gguf_string_t string;
        struct {
            gguf_metadata_value_type type;
            uint64_t len;
            gguf_metadata_value_t *array;
        } array;
    };
};

// GGUF Metadata Key-Value pair
typedef struct {
    gguf_string_t key;
    gguf_metadata_value_type value_type;
    gguf_metadata_value_t value;
} gguf_metadata_kv_t;

// GGUF Header
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    gguf_metadata_kv_t *metadata_kv;
} gguf_header_t;

// GGUF Tensor Info
typedef struct {
    gguf_string_t name;
    uint32_t n_dimensions;
    uint64_t *dimensions;
    ggml_type type;
    uint64_t offset;
} gguf_tensor_info_t;

// GGUF File structure
typedef struct {
    gguf_header_t header;
    gguf_tensor_info_t *tensor_infos;
    uint8_t *tensor_data;
    size_t tensor_data_size;
    size_t file_size;
} gguf_file_t;

// 🧠 Model Structure - Transformer Architecture
// Attention layer structure with separate Q/K/V weights
typedef struct {
    // Query, Key, Value weight matrices
    gguf_tensor_info_t *wq;      // Query weights
    gguf_tensor_info_t *wk;      // Key weights  
    gguf_tensor_info_t *wv;      // Value weights
    gguf_tensor_info_t *wo;      // Output projection weights
    
    // Optional bias tensors
    gguf_tensor_info_t *bq;      // Query bias (optional)
    gguf_tensor_info_t *bk;      // Key bias (optional)
    gguf_tensor_info_t *bv;      // Value bias (optional)
    gguf_tensor_info_t *bo;      // Output bias (optional)
    
    // Attention parameters
    uint32_t n_heads;            // Number of attention heads
    uint32_t head_dim;           // Dimension per head
    uint32_t n_kv_heads;         // Number of key-value heads (for GQA/MQA)
} attention_layer_t;

// Feed-forward network structure
typedef struct {
    gguf_tensor_info_t *w1;      // First linear layer (up projection)
    gguf_tensor_info_t *w2;      // Second linear layer (down projection)  
    gguf_tensor_info_t *w3;      // Third linear layer (gate projection, for SwiGLU)
    
    // Optional bias tensors
    gguf_tensor_info_t *b1;      // First layer bias (optional)
    gguf_tensor_info_t *b2;      // Second layer bias (optional)
    gguf_tensor_info_t *b3;      // Third layer bias (optional)
    
    uint32_t intermediate_size;   // Hidden dimension of FFN
    const char* activation;       // Activation function name
} ffn_layer_t;

// Layer normalization structure
typedef struct {
    gguf_tensor_info_t *weight;   // Layer norm weights
    gguf_tensor_info_t *bias;     // Layer norm bias (optional)
    float eps;                    // Epsilon for numerical stability
} layer_norm_t;

// Single transformer layer
typedef struct {
    // Pre-attention layer norm
    layer_norm_t attn_norm;
    
    // Self-attention
    attention_layer_t attention;
    
    // Pre-FFN layer norm  
    layer_norm_t ffn_norm;
    
    // Feed-forward network
    ffn_layer_t ffn;
    
    uint32_t layer_id;           // Layer index
} transformer_layer_t;

// Complete transformer model structure
typedef struct {
    // Model hyperparameters
    uint32_t vocab_size;         // Vocabulary size
    uint32_t hidden_size;        // Hidden dimension
    uint32_t intermediate_size;  // FFN intermediate size
    uint32_t n_layers;           // Number of transformer layers
    uint32_t n_heads;            // Number of attention heads
    uint32_t n_kv_heads;         // Number of key-value heads
    uint32_t head_dim;           // Dimension per head
    uint32_t max_seq_len;        // Maximum sequence length
    float rope_theta;            // RoPE theta parameter
    float rms_norm_eps;          // RMS norm epsilon
    
    // Model architecture type
    const char* arch;            // Architecture name (llama, mistral, etc.)
    const char* name;            // Model name
    
    // Embedding layers
    gguf_tensor_info_t *token_embd;    // Token embeddings
    gguf_tensor_info_t *pos_embd;      // Positional embeddings (optional)
    
    // Transformer layers
    transformer_layer_t *layers;        // Array of transformer layers
    
    // Output layers
    layer_norm_t output_norm;           // Final layer normalization
    gguf_tensor_info_t *output;         // Output projection (lm_head)
    
    // Model metadata
    uint64_t parameter_count;           // Total number of parameters
    size_t model_size_bytes;           // Model size in bytes
    
    // Reference to the loaded GGUF file
    gguf_file_t *gguf_file;
} transformer_model_t;

// Find tensor by name in GGUF file
gguf_tensor_info_t* find_tensor(gguf_file_t *gguf, const char *name) {
    for (uint64_t i = 0; i < gguf->header.tensor_count; i++) {
        if (strcmp(gguf->tensor_infos[i].name.string, name) == 0) {
            return &gguf->tensor_infos[i];
        }
    }
    return NULL;
}

// Get metadata value as string
const char* get_metadata_string(gguf_file_t *gguf, const char *key) {
    for (uint64_t i = 0; i < gguf->header.metadata_kv_count; i++) {
        if (strcmp(gguf->header.metadata_kv[i].key.string, key) == 0) {
            if (gguf->header.metadata_kv[i].value_type == GGUF_METADATA_VALUE_TYPE_STRING) {
                return gguf->header.metadata_kv[i].value.string.string;
            }
        }
    }
    return NULL;
}

// Get metadata value as uint32
uint32_t get_metadata_uint32(gguf_file_t *gguf, const char *key, uint32_t default_val) {
    for (uint64_t i = 0; i < gguf->header.metadata_kv_count; i++) {
        if (strcmp(gguf->header.metadata_kv[i].key.string, key) == 0) {
            switch (gguf->header.metadata_kv[i].value_type) {
                case GGUF_METADATA_VALUE_TYPE_UINT32:
                    return gguf->header.metadata_kv[i].value.uint32;
                case GGUF_METADATA_VALUE_TYPE_UINT64:
                    return (uint32_t)gguf->header.metadata_kv[i].value.uint64;
                case GGUF_METADATA_VALUE_TYPE_INT32:
                    return (uint32_t)gguf->header.metadata_kv[i].value.int32;
                case GGUF_METADATA_VALUE_TYPE_INT64:
                    return (uint32_t)gguf->header.metadata_kv[i].value.int64;
                default:
                    break;
            }
        }
    }
    return default_val;
}

// Get metadata value as float
float get_metadata_float(gguf_file_t *gguf, const char *key, float default_val) {
    for (uint64_t i = 0; i < gguf->header.metadata_kv_count; i++) {
        if (strcmp(gguf->header.metadata_kv[i].key.string, key) == 0) {
            switch (gguf->header.metadata_kv[i].value_type) {
                case GGUF_METADATA_VALUE_TYPE_FLOAT32:
                    return gguf->header.metadata_kv[i].value.float32;
                case GGUF_METADATA_VALUE_TYPE_FLOAT64:
                    return (float)gguf->header.metadata_kv[i].value.float64;
                default:
                    break;
            }
        }
    }
    return default_val;
}

// Build transformer model structure from GGUF file
transformer_model_t* build_transformer_model(gguf_file_t *gguf) {
    printf("\n==== Building Transformer Model Structure ====\n");
    
    transformer_model_t *model = calloc(1, sizeof(transformer_model_t));
    if (!model) {
        fprintf(stderr, "Error allocating memory for transformer model\n");
        return NULL;
    }
    
    model->gguf_file = gguf;
    
    // Extract model hyperparameters from metadata
    model->arch = get_metadata_string(gguf, "general.architecture");
    model->name = get_metadata_string(gguf, "general.name");
    
    // Common hyperparameters with architecture-specific prefixes
    const char *arch_prefix = model->arch ? model->arch : "llama";
    
    char key_buffer[256];
    
    // Get model dimensions
    snprintf(key_buffer, sizeof(key_buffer), "%s.embedding_length", arch_prefix);
    model->hidden_size = get_metadata_uint32(gguf, key_buffer, 4096);
    
    snprintf(key_buffer, sizeof(key_buffer), "%s.feed_forward_length", arch_prefix);
    model->intermediate_size = get_metadata_uint32(gguf, key_buffer, 11008);
    
    snprintf(key_buffer, sizeof(key_buffer), "%s.block_count", arch_prefix);
    model->n_layers = get_metadata_uint32(gguf, key_buffer, 32);
    
    snprintf(key_buffer, sizeof(key_buffer), "%s.attention.head_count", arch_prefix);
    model->n_heads = get_metadata_uint32(gguf, key_buffer, 32);
    
    snprintf(key_buffer, sizeof(key_buffer), "%s.attention.head_count_kv", arch_prefix);
    model->n_kv_heads = get_metadata_uint32(gguf, key_buffer, model->n_heads);
    
    model->head_dim = model->hidden_size / model->n_heads;
    
    snprintf(key_buffer, sizeof(key_buffer), "%s.context_length", arch_prefix);
    model->max_seq_len = get_metadata_uint32(gguf, key_buffer, 2048);
    
    snprintf(key_buffer, sizeof(key_buffer), "%s.attention.layer_norm_rms_epsilon", arch_prefix);
    model->rms_norm_eps = get_metadata_float(gguf, key_buffer, 1e-6f);
    
    snprintf(key_buffer, sizeof(key_buffer), "%s.rope.theta", arch_prefix);
    model->rope_theta = get_metadata_float(gguf, key_buffer, 10000.0f);
    
    // Get vocabulary size from tokenizer
    model->vocab_size = get_metadata_uint32(gguf, "tokenizer.ggml.tokens", 32000);
    
    printf("Architecture: %s\n", model->arch ? model->arch : "unknown");
    printf("Model Name: %s\n", model->name ? model->name : "unknown");
    printf("Hidden Size: %u\n", model->hidden_size);
    printf("Intermediate Size: %u\n", model->intermediate_size);
    printf("Layers: %u\n", model->n_layers);
    printf("Attention Heads: %u\n", model->n_heads);
    printf("KV Heads: %u\n", model->n_kv_heads);
    printf("Head Dimension: %u\n", model->head_dim);
    printf("Max Sequence Length: %u\n", model->max_seq_len);
    printf("Vocabulary Size: %u\n", model->vocab_size);
    printf("RoPE Theta: %.1f\n", model->rope_theta);
    printf("RMS Norm Epsilon: %e\n", model->rms_norm_eps);
    
    // Find embedding tensors
    model->token_embd = find_tensor(gguf, "token_embd.weight");
    if (!model->token_embd) {
        model->token_embd = find_tensor(gguf, "tok_embeddings.weight");
    }
    
    // Find output tensors
    model->output_norm.weight = find_tensor(gguf, "output_norm.weight");
    if (!model->output_norm.weight) {
        model->output_norm.weight = find_tensor(gguf, "norm.weight");
    }
    
    model->output = find_tensor(gguf, "output.weight");
    if (!model->output) {
        model->output = find_tensor(gguf, "lm_head.weight");
    }
    
    // Allocate transformer layers
    model->layers = calloc(model->n_layers, sizeof(transformer_layer_t));
    if (!model->layers) {
        fprintf(stderr, "Error allocating memory for transformer layers\n");
        free(model);
        return NULL;
    }
    
    // Build each transformer layer
    for (uint32_t i = 0; i < model->n_layers; i++) {
        transformer_layer_t *layer = &model->layers[i];
        layer->layer_id = i;
        
        // Build tensor names for this layer
        char tensor_name[256];
        
        // Attention layer norm
        snprintf(tensor_name, sizeof(tensor_name), "blk.%u.attn_norm.weight", i);
        layer->attn_norm.weight = find_tensor(gguf, tensor_name);
        
        // Attention weights - try different naming conventions
        snprintf(tensor_name, sizeof(tensor_name), "blk.%u.attn_q.weight", i);
        layer->attention.wq = find_tensor(gguf, tensor_name);
        if (!layer->attention.wq) {
            snprintf(tensor_name, sizeof(tensor_name), "layers.%u.attention.wq.weight", i);
            layer->attention.wq = find_tensor(gguf, tensor_name);
        }
        
        snprintf(tensor_name, sizeof(tensor_name), "blk.%u.attn_k.weight", i);
        layer->attention.wk = find_tensor(gguf, tensor_name);
        if (!layer->attention.wk) {
            snprintf(tensor_name, sizeof(tensor_name), "layers.%u.attention.wk.weight", i);
            layer->attention.wk = find_tensor(gguf, tensor_name);
        }
        
        snprintf(tensor_name, sizeof(tensor_name), "blk.%u.attn_v.weight", i);
        layer->attention.wv = find_tensor(gguf, tensor_name);
        if (!layer->attention.wv) {
            snprintf(tensor_name, sizeof(tensor_name), "layers.%u.attention.wv.weight", i);
            layer->attention.wv = find_tensor(gguf, tensor_name);
        }
        
        snprintf(tensor_name, sizeof(tensor_name), "blk.%u.attn_output.weight", i);
        layer->attention.wo = find_tensor(gguf, tensor_name);
        if (!layer->attention.wo) {
            snprintf(tensor_name, sizeof(tensor_name), "layers.%u.attention.wo.weight", i);
            layer->attention.wo = find_tensor(gguf, tensor_name);
        }
        
        // Set attention parameters
        layer->attention.n_heads = model->n_heads;
        layer->attention.n_kv_heads = model->n_kv_heads;
        layer->attention.head_dim = model->head_dim;
        
        // FFN layer norm
        snprintf(tensor_name, sizeof(tensor_name), "blk.%u.ffn_norm.weight", i);
        layer->ffn_norm.weight = find_tensor(gguf, tensor_name);
        
        // FFN weights
        snprintf(tensor_name, sizeof(tensor_name), "blk.%u.ffn_gate.weight", i);
        layer->ffn.w1 = find_tensor(gguf, tensor_name);
        if (!layer->ffn.w1) {
            snprintf(tensor_name, sizeof(tensor_name), "layers.%u.feed_forward.w1.weight", i);
            layer->ffn.w1 = find_tensor(gguf, tensor_name);
        }
        
        snprintf(tensor_name, sizeof(tensor_name), "blk.%u.ffn_down.weight", i);
        layer->ffn.w2 = find_tensor(gguf, tensor_name);
        if (!layer->ffn.w2) {
            snprintf(tensor_name, sizeof(tensor_name), "layers.%u.feed_forward.w2.weight", i);
            layer->ffn.w2 = find_tensor(gguf, tensor_name);
        }
        
        snprintf(tensor_name, sizeof(tensor_name), "blk.%u.ffn_up.weight", i);
        layer->ffn.w3 = find_tensor(gguf, tensor_name);
        if (!layer->ffn.w3) {
            snprintf(tensor_name, sizeof(tensor_name), "layers.%u.feed_forward.w3.weight", i);
            layer->ffn.w3 = find_tensor(gguf, tensor_name);
        }
        
        layer->ffn.intermediate_size = model->intermediate_size;
        layer->ffn.activation = "silu"; // Default to SiLU/Swish
        
        printf("Layer %u: ", i);
        printf("Attn[Q:%s K:%s V:%s O:%s] ", 
               layer->attention.wq ? "✓" : "✗",
               layer->attention.wk ? "✓" : "✗", 
               layer->attention.wv ? "✓" : "✗",
               layer->attention.wo ? "✓" : "✗");
        printf("FFN[W1:%s W2:%s W3:%s]\n",
               layer->ffn.w1 ? "✓" : "✗",
               layer->ffn.w2 ? "✓" : "✗", 
               layer->ffn.w3 ? "✓" : "✗");
    }
    
    // Calculate model statistics
    model->parameter_count = 0;
    model->model_size_bytes = 0;
    
    for (uint64_t i = 0; i < gguf->header.tensor_count; i++) {
        gguf_tensor_info_t *tensor = &gguf->tensor_infos[i];
        
        // Calculate tensor parameter count
        uint64_t tensor_params = 1;
        for (uint32_t j = 0; j < tensor->n_dimensions; j++) {
            tensor_params *= tensor->dimensions[j];
        }
        model->parameter_count += tensor_params;
        
        // Estimate tensor size (this is approximate)
        size_t element_size = 4; // Default to 4 bytes (float32)
        switch (tensor->type) {
            case GGML_TYPE_F16:
                element_size = 2;
                break;
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
                element_size = 1; // Approximation for quantized
                break;
            default:
                break;
        }
        model->model_size_bytes += tensor_params * element_size;
    }
    
    printf("\nModel Statistics:\n");
    printf("Total Parameters: %lu (%.2fB)\n", model->parameter_count, model->parameter_count / 1e9);
    printf("Estimated Model Size: %.2f MB\n", model->model_size_bytes / (1024.0 * 1024.0));
    printf("Actual File Size: %.2f MB\n", gguf->file_size / (1024.0 * 1024.0));
    
    return model;
}
uint64_t align_offset(uint64_t offset) {
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
}

const char* ggml_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case GGML_TYPE_IQ2_XS: return "IQ2_XS";
        case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case GGML_TYPE_IQ1_S: return "IQ1_S";
        case GGML_TYPE_IQ4_NL: return "IQ4_NL";
        case GGML_TYPE_IQ3_S: return "IQ3_S";
        case GGML_TYPE_IQ2_S: return "IQ2_S";
        case GGML_TYPE_IQ4_XS: return "IQ4_XS";
        case GGML_TYPE_I8: return "I8";
        case GGML_TYPE_I16: return "I16";
        case GGML_TYPE_I32: return "I32";
        case GGML_TYPE_I64: return "I64";
        case GGML_TYPE_F64: return "F64";
        case GGML_TYPE_IQ1_M: return "IQ1_M";
        default: return "UNKNOWN";
    }
}

// Read string from file
int read_gguf_string(FILE *fp, gguf_string_t *str) {
    if (fread(&str->len, sizeof(uint64_t), 1, fp) != 1) {
        return -1;
    }
    
    if (str->len > 0) {
        str->string = malloc(str->len + 1);
        if (!str->string) {
            return -1;
        }
        
        if (fread(str->string, 1, str->len, fp) != str->len) {
            free(str->string);
            return -1;
        }
        str->string[str->len] = '\0';  // Null terminate for safety
    } else {
        str->string = NULL;
    }
    
    return 0;
}

// Read metadata value from file
int read_gguf_metadata_value(FILE *fp, gguf_metadata_value_type type, gguf_metadata_value_t *value) {
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8:
            return fread(&value->uint8, sizeof(uint8_t), 1, fp) == 1 ? 0 : -1;
        case GGUF_METADATA_VALUE_TYPE_INT8:
            return fread(&value->int8, sizeof(int8_t), 1, fp) == 1 ? 0 : -1;
        case GGUF_METADATA_VALUE_TYPE_UINT16:
            return fread(&value->uint16, sizeof(uint16_t), 1, fp) == 1 ? 0 : -1;
        case GGUF_METADATA_VALUE_TYPE_INT16:
            return fread(&value->int16, sizeof(int16_t), 1, fp) == 1 ? 0 : -1;
        case GGUF_METADATA_VALUE_TYPE_UINT32:
            return fread(&value->uint32, sizeof(uint32_t), 1, fp) == 1 ? 0 : -1;
        case GGUF_METADATA_VALUE_TYPE_INT32:
            return fread(&value->int32, sizeof(int32_t), 1, fp) == 1 ? 0 : -1;
        case GGUF_METADATA_VALUE_TYPE_FLOAT32:
            return fread(&value->float32, sizeof(float), 1, fp) == 1 ? 0 : -1;
        case GGUF_METADATA_VALUE_TYPE_UINT64:
            return fread(&value->uint64, sizeof(uint64_t), 1, fp) == 1 ? 0 : -1;
        case GGUF_METADATA_VALUE_TYPE_INT64:
            return fread(&value->int64, sizeof(int64_t), 1, fp) == 1 ? 0 : -1;
        case GGUF_METADATA_VALUE_TYPE_FLOAT64:
            return fread(&value->float64, sizeof(double), 1, fp) == 1 ? 0 : -1;
        case GGUF_METADATA_VALUE_TYPE_BOOL:
            return fread(&value->bool_, sizeof(bool), 1, fp) == 1 ? 0 : -1;
        case GGUF_METADATA_VALUE_TYPE_STRING:
            return read_gguf_string(fp, &value->string);
        case GGUF_METADATA_VALUE_TYPE_ARRAY:
            if (fread(&value->array.type, sizeof(gguf_metadata_value_type), 1, fp) != 1) {
                return -1;
            }
            if (fread(&value->array.len, sizeof(uint64_t), 1, fp) != 1) {
                return -1;
            }
            
            if (value->array.len > 0) {
                value->array.array = malloc(sizeof(gguf_metadata_value_t) * value->array.len);
                if (!value->array.array) {
                    return -1;
                }
                
                for (uint64_t i = 0; i < value->array.len; i++) {
                    if (read_gguf_metadata_value(fp, value->array.type, &value->array.array[i]) != 0) {
                        free(value->array.array);
                        return -1;
                    }
                }
            } else {
                value->array.array = NULL;
            }
            return 0;
        default:
            return -1;
    }
}

// 1. Load GGUF Header
int load_gguf_header(FILE *fp, gguf_header_t *header) {
    printf("==== Loading GGUF Header ====\n");
    
    // Read magic number
    if (fread(&header->magic, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "Error reading magic number\n");
        return -1;
    }
    
    if (header->magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic number: 0x%08X (expected 0x%08X)\n", 
                header->magic, GGUF_MAGIC);
        return -1;
    }
    
    // Read version
    if (fread(&header->version, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "Error reading version\n");
        return -1;
    }
    
    if (header->version != GGUF_VERSION) {
        fprintf(stderr, "Unsupported GGUF version: %u (expected %u)\n", 
                header->version, GGUF_VERSION);
        return -1;
    }
    
    // Read tensor count
    if (fread(&header->tensor_count, sizeof(uint64_t), 1, fp) != 1) {
        fprintf(stderr, "Error reading tensor count\n");
        return -1;
    }
    
    // Read metadata count
    if (fread(&header->metadata_kv_count, sizeof(uint64_t), 1, fp) != 1) {
        fprintf(stderr, "Error reading metadata count\n");
        return -1;
    }
    
    printf("Magic: 0x%08X\n", header->magic);
    printf("Version: %u\n", header->version);
    printf("Tensor Count: %lu\n", header->tensor_count);
    printf("Metadata KV Count: %lu\n", header->metadata_kv_count);
    
    return 0;
}

// 2. Load Metadata Key-Value Store
int load_gguf_metadata(FILE *fp, gguf_header_t *header) {
    printf("\n==== Loading Metadata Key-Value Store ====\n");
    
    if (header->metadata_kv_count == 0) {
        header->metadata_kv = NULL;
        return 0;
    }
    
    header->metadata_kv = malloc(sizeof(gguf_metadata_kv_t) * header->metadata_kv_count);
    if (!header->metadata_kv) {
        fprintf(stderr, "Error allocating memory for metadata\n");
        return -1;
    }
    
    for (uint64_t i = 0; i < header->metadata_kv_count; i++) {
        gguf_metadata_kv_t *kv = &header->metadata_kv[i];
        
        // Read key
        if (read_gguf_string(fp, &kv->key) != 0) {
            fprintf(stderr, "Error reading metadata key %lu\n", i);
            return -1;
        }
        
        // Read value type
        if (fread(&kv->value_type, sizeof(gguf_metadata_value_type), 1, fp) != 1) {
            fprintf(stderr, "Error reading metadata value type %lu\n", i);
            return -1;
        }
        
        // Read value
        if (read_gguf_metadata_value(fp, kv->value_type, &kv->value) != 0) {
            fprintf(stderr, "Error reading metadata value %lu\n", i);
            return -1;
        }
        
        printf("Metadata[%lu]: Key='%s', Type=%u\n", i, kv->key.string, kv->value_type);
    }
    
    return 0;
}

// 3. Load Tensor Info
int load_gguf_tensor_info(FILE *fp, gguf_file_t *gguf) {
    printf("\n==== Loading Tensor Info ====\n");
    
    if (gguf->header.tensor_count == 0) {
        gguf->tensor_infos = NULL;
        return 0;
    }
    
    gguf->tensor_infos = malloc(sizeof(gguf_tensor_info_t) * gguf->header.tensor_count);
    if (!gguf->tensor_infos) {
        fprintf(stderr, "Error allocating memory for tensor infos\n");
        return -1;
    }
    
    for (uint64_t i = 0; i < gguf->header.tensor_count; i++) {
        gguf_tensor_info_t *info = &gguf->tensor_infos[i];
        
        // Read tensor name
        if (read_gguf_string(fp, &info->name) != 0) {
            fprintf(stderr, "Error reading tensor name %lu\n", i);
            return -1;
        }
        
        // Read number of dimensions
        if (fread(&info->n_dimensions, sizeof(uint32_t), 1, fp) != 1) {
            fprintf(stderr, "Error reading tensor dimensions count %lu\n", i);
            return -1;
        }
        
        // Read dimensions
        if (info->n_dimensions > 0) {
            info->dimensions = malloc(sizeof(uint64_t) * info->n_dimensions);
            if (!info->dimensions) {
                fprintf(stderr, "Error allocating memory for tensor dimensions %lu\n", i);
                return -1;
            }
            
            if (fread(info->dimensions, sizeof(uint64_t), info->n_dimensions, fp) != info->n_dimensions) {
                fprintf(stderr, "Error reading tensor dimensions %lu\n", i);
                return -1;
            }
        } else {
            info->dimensions = NULL;
        }
        
        // Read tensor type
        if (fread(&info->type, sizeof(ggml_type), 1, fp) != 1) {
            fprintf(stderr, "Error reading tensor type %lu\n", i);
            return -1;
        }
        
        // Read tensor offset
        if (fread(&info->offset, sizeof(uint64_t), 1, fp) != 1) {
            fprintf(stderr, "Error reading tensor offset %lu\n", i);
            return -1;
        }
        
        printf("Tensor[%lu]: Name='%s', Dims=%u, Type=%s, Offset=%lu\n", 
               i, info->name.string, info->n_dimensions, 
               ggml_type_name(info->type), info->offset);
        
        if (info->n_dimensions > 0) {
            printf("  Dimensions: ");
            for (uint32_t j = 0; j < info->n_dimensions; j++) {
                printf("%lu%s", info->dimensions[j], (j < info->n_dimensions - 1) ? "x" : "");
            }
            printf("\n");
        }
    }
    
    return 0;
}

// 4. Load Tensor Data
int load_gguf_tensor_data(FILE *fp, gguf_file_t *gguf) {
    printf("\n==== Loading Tensor Data ====\n");
    
    // Calculate current position and align to tensor data start
    long current_pos = ftell(fp);
    if (current_pos == -1) {
        fprintf(stderr, "Error getting file position\n");
        return -1;
    }
    
    uint64_t aligned_pos = align_offset(current_pos);
    if (fseek(fp, aligned_pos, SEEK_SET) != 0) {
        fprintf(stderr, "Error seeking to aligned tensor data position\n");
        return -1;
    }
    
    // Get file size to calculate tensor data size
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    if (file_size == -1) {
        fprintf(stderr, "Error getting file size\n");
        return -1;
    }
    
    gguf->file_size = file_size;
    gguf->tensor_data_size = file_size - aligned_pos;
    
    // Seek back to tensor data start
    fseek(fp, aligned_pos, SEEK_SET);
    
    // Allocate memory for tensor data
    gguf->tensor_data = malloc(gguf->tensor_data_size);
    if (!gguf->tensor_data) {
        fprintf(stderr, "Error allocating memory for tensor data (%zu bytes)\n", 
                gguf->tensor_data_size);
        return -1;
    }
    
    // Read tensor data
    size_t read_bytes = fread(gguf->tensor_data, 1, gguf->tensor_data_size, fp);
    if (read_bytes != gguf->tensor_data_size) {
        fprintf(stderr, "Error reading tensor data: read %zu bytes, expected %zu\n", 
                read_bytes, gguf->tensor_data_size);
        free(gguf->tensor_data);
        return -1;
    }
    
    printf("Tensor data loaded: %zu bytes\n", gguf->tensor_data_size);
    printf("Tensor data starts at file offset: %lu\n", aligned_pos);
    
    // Verify tensor offsets
    for (uint64_t i = 0; i < gguf->header.tensor_count; i++) {
        gguf_tensor_info_t *info = &gguf->tensor_infos[i];
        if (info->offset >= gguf->tensor_data_size) {
            fprintf(stderr, "Warning: Tensor '%s' offset %lu exceeds tensor data size %zu\n",
                    info->name.string, info->offset, gguf->tensor_data_size);
        }
    }
    
    return 0;
}

// Load complete GGUF file
gguf_file_t* load_gguf_file(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s': %s\n", filename, strerror(errno));
        return NULL;
    }
    
    gguf_file_t *gguf = calloc(1, sizeof(gguf_file_t));
    if (!gguf) {
        fprintf(stderr, "Error allocating memory for GGUF file structure\n");
        fclose(fp);
        return NULL;
    }
    
    // Load all four parts
    if (load_gguf_header(fp, &gguf->header) != 0) {
        free(gguf);
        fclose(fp);
        return NULL;
    }
    
    if (load_gguf_metadata(fp, &gguf->header) != 0) {
        free(gguf);
        fclose(fp);
        return NULL;
    }
    
    if (load_gguf_tensor_info(fp, gguf) != 0) {
        free(gguf);
        fclose(fp);
        return NULL;
    }
    
    if (load_gguf_tensor_data(fp, gguf) != 0) {
        free(gguf);
        fclose(fp);
        return NULL;
    }
    
    fclose(fp);
    
    printf("\n==== GGUF File Loaded Successfully ====\n");
    printf("File: %s\n", filename);
    printf("Total file size: %zu bytes\n", gguf->file_size);
    printf("Tensors: %lu\n", gguf->header.tensor_count);
    printf("Metadata entries: %lu\n", gguf->header.metadata_kv_count);
    
    return gguf;
}

// Free transformer model memory
void free_transformer_model(transformer_model_t *model) {
    if (!model) return;
    
    // Free layers array
    free(model->layers);
    
    // Free main structure
    free(model);
}

// Get tensor data pointer for a specific tensor
uint8_t* get_tensor_data(transformer_model_t *model, gguf_tensor_info_t *tensor) {
    if (!model || !tensor || !model->gguf_file || !model->gguf_file->tensor_data) {
        return NULL;
    }
    
    if (tensor->offset >= model->gguf_file->tensor_data_size) {
        return NULL;
    }
    
    return model->gguf_file->tensor_data + tensor->offset;
}

// Print detailed model architecture information
void print_model_architecture(transformer_model_t *model) {
    if (!model) return;
    
    printf("\n==== Model Architecture Details ====\n");
    printf("Architecture: %s\n", model->arch ? model->arch : "unknown");
    printf("Model Name: %s\n", model->name ? model->name : "unknown");
    
    printf("\nModel Dimensions:\n");
    printf("  Hidden Size: %u\n", model->hidden_size);
    printf("  Intermediate Size: %u\n", model->intermediate_size);
    printf("  Vocabulary Size: %u\n", model->vocab_size);
    printf("  Max Sequence Length: %u\n", model->max_seq_len);
    
    printf("\nAttention Configuration:\n");
    printf("  Number of Heads: %u\n", model->n_heads);
    printf("  Number of KV Heads: %u\n", model->n_kv_heads);
    printf("  Head Dimension: %u\n", model->head_dim);
    printf("  Total Attention Dim: %u\n", model->n_heads * model->head_dim);
    
    if (model->n_kv_heads != model->n_heads) {
        printf("  Using Grouped Query Attention (GQA)\n");
        printf("  Query Groups: %u\n", model->n_heads / model->n_kv_heads);
    }
    
    printf("\nLayer Configuration:\n");
    printf("  Number of Layers: %u\n", model->n_layers);
    printf("  RoPE Theta: %.1f\n", model->rope_theta);
    printf("  RMS Norm Epsilon: %e\n", model->rms_norm_eps);
    
    printf("\nEmbedding Layers:\n");
    printf("  Token Embeddings: %s\n", model->token_embd ? "✓" : "✗");
    printf("  Position Embeddings: %s\n", model->pos_embd ? "✓" : "✗");
    
    printf("\nOutput Layers:\n");
    printf("  Output Norm: %s\n", model->output_norm.weight ? "✓" : "✗");
    printf("  Output Projection: %s\n", model->output ? "✓" : "✗");
    
    printf("\nLayer-by-Layer Breakdown:\n");
    for (uint32_t i = 0; i < model->n_layers; i++) {
        transformer_layer_t *layer = &model->layers[i];
        
        printf("  Layer %2u: ", i);
        
        // Attention components
        printf("Attn[");
        printf("Norm:%s ", layer->attn_norm.weight ? "✓" : "✗");
        printf("Q:%s ", layer->attention.wq ? "✓" : "✗");
        printf("K:%s ", layer->attention.wk ? "✓" : "✗");
        printf("V:%s ", layer->attention.wv ? "✓" : "✗");
        printf("O:%s", layer->attention.wo ? "✓" : "✗");
        printf("] ");
        
        // FFN components
        printf("FFN[");
        printf("Norm:%s ", layer->ffn_norm.weight ? "✓" : "✗");
        printf("W1:%s ", layer->ffn.w1 ? "✓" : "✗");
        printf("W2:%s ", layer->ffn.w2 ? "✓" : "✗");
        printf("W3:%s", layer->ffn.w3 ? "✓" : "✗");
        printf("]\n");
    }
    
    printf("\nModel Statistics:\n");
    printf("  Total Parameters: %lu (%.2fB)\n", 
           model->parameter_count, model->parameter_count / 1e9);
    printf("  Estimated Size: %.2f MB\n", 
           model->model_size_bytes / (1024.0 * 1024.0));
    printf("  File Size: %.2f MB\n", 
           model->gguf_file->file_size / (1024.0 * 1024.0));
    
    // Calculate compression ratio
    if (model->model_size_bytes > 0) {
        double compression_ratio = (double)model->gguf_file->file_size / model->model_size_bytes;
        printf("  Compression Ratio: %.2fx\n", 1.0 / compression_ratio);
    }
}
void free_gguf_file(gguf_file_t *gguf) {
    if (!gguf) return;
    
    // Free metadata
    if (gguf->header.metadata_kv) {
        for (uint64_t i = 0; i < gguf->header.metadata_kv_count; i++) {
            free(gguf->header.metadata_kv[i].key.string);
            // Note: More complex cleanup needed for nested arrays and strings in values
        }
        free(gguf->header.metadata_kv);
    }
    
    // Free tensor infos
    if (gguf->tensor_infos) {
        for (uint64_t i = 0; i < gguf->header.tensor_count; i++) {
            free(gguf->tensor_infos[i].name.string);
            free(gguf->tensor_infos[i].dimensions);
        }
        free(gguf->tensor_infos);
    }
    
    // Free tensor data
    free(gguf->tensor_data);
    
    // Free main structure
    free(gguf);
}

// Main function for testing
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <gguf_file>\n", argv[0]);
        return 1;
    }
    
    const char *filename = argv[1];
    
    printf("Loading GGUF file: %s\n", filename);
    printf("========================================\n");
    
    // Load GGUF file
    gguf_file_t *gguf = load_gguf_file(filename);
    if (!gguf) {
        fprintf(stderr, "Failed to load GGUF file\n");
        return 1;
    }
    
    // Build transformer model structure
    transformer_model_t *model = build_transformer_model(gguf);
    if (!model) {
        fprintf(stderr, "Failed to build transformer model structure\n");
        free_gguf_file(gguf);
        return 1;
    }
    
    // Print detailed architecture information
    print_model_architecture(model);
    
    // Example: Access tensor data for the first layer's query weights
    if (model->layers[0].attention.wq) {
        uint8_t *wq_data = get_tensor_data(model, model->layers[0].attention.wq);
        if (wq_data) {
            printf("\nFirst layer query weights data loaded successfully at address: %p\n", (void*)wq_data);
            
            // Print tensor information
            gguf_tensor_info_t *wq = model->layers[0].attention.wq;
            printf("Tensor: %s\n", wq->name.string);
            printf("Type: %s\n", ggml_type_name(wq->type));
            printf("Dimensions: ");
            for (uint32_t i = 0; i < wq->n_dimensions; i++) {
                printf("%lu%s", wq->dimensions[i], (i < wq->n_dimensions - 1) ? "x" : "");
            }
            printf("\n");
            printf("Offset: %lu bytes\n", wq->offset);
        }
    }
    
    // Example usage: You can now access any tensor data like this:
    // uint8_t *token_embd_data = get_tensor_data(model, model->token_embd);
    // uint8_t *layer0_ffn_w1_data = get_tensor_data(model, model->layers[0].ffn.w1);
    
    printf("\n==== Model successfully loaded and structured ====\n");
    printf("The transformer model is now ready for inference!\n");
    
    // Clean up
    free_transformer_model(model);
    free_gguf_file(gguf);
    
    return 0;
}
