
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <inttypes.h>

// Define alignment constant (32 bytes as per GGUF spec)
#define ALIGNMENT 32

// GGML tensor types
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
    GGML_TYPE_COUNT
} ggml_type;

// GGUF metadata value types
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
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12
} gguf_metadata_value_type;

// GGUF string type
typedef struct {
    uint64_t len;
    char* string;
} gguf_string_t;

// GGUF metadata value
typedef union {
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
        void* array;  // Will be cast to appropriate type
    } array;
} gguf_metadata_value_t;

// GGUF metadata key-value pair
typedef struct {
    gguf_string_t key;
    gguf_metadata_value_type value_type;
    gguf_metadata_value_t value;
} gguf_metadata_kv_t;

// GGUF header
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    gguf_metadata_kv_t* metadata_kv;
} gguf_header_t;

// GGUF tensor info
typedef struct {
    gguf_string_t name;
    uint32_t n_dimensions;
    uint64_t* dimensions;
    ggml_type type;
    uint64_t offset;
} gguf_tensor_info_t;

// GGUF file structure
typedef struct {
    gguf_header_t header;
    gguf_tensor_info_t* tensor_infos;
    size_t data_offset;
} gguf_file_t;

// Alignment helper
static uint64_t align_offset(uint64_t offset) {
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
}

// Print helpers
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

const char* metadata_type_name(gguf_metadata_value_type type) {
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8: return "uint8";
        case GGUF_METADATA_VALUE_TYPE_INT8: return "int8";
        case GGUF_METADATA_VALUE_TYPE_UINT16: return "uint16";
        case GGUF_METADATA_VALUE_TYPE_INT16: return "int16";
        case GGUF_METADATA_VALUE_TYPE_UINT32: return "uint32";
        case GGUF_METADATA_VALUE_TYPE_INT32: return "int32";
        case GGUF_METADATA_VALUE_TYPE_FLOAT32: return "float32";
        case GGUF_METADATA_VALUE_TYPE_BOOL: return "bool";
        case GGUF_METADATA_VALUE_TYPE_STRING: return "string";
        case GGUF_METADATA_VALUE_TYPE_ARRAY: return "array";
        case GGUF_METADATA_VALUE_TYPE_UINT64: return "uint64";
        case GGUF_METADATA_VALUE_TYPE_INT64: return "int64";
        case GGUF_METADATA_VALUE_TYPE_FLOAT64: return "float64";
        default: return "unknown";
    }
}

// File reading helpers
//Byte Reading Function
static uint16_t read_uint16(FILE* file) {
    uint16_t val;
    fread(&val, sizeof(val), 1, file);
    return val;
}

static uint32_t read_uint32(FILE* file) {
    uint32_t val;
    fread(&val, sizeof(val), 1, file);
    return val;
}

static uint64_t read_uint64(FILE* file) {
    uint64_t val;
    fread(&val, sizeof(val), 1, file);
    return val;
}

static float read_float32(FILE* file) {
    float val;
    fread(&val, sizeof(val), 1, file);
    return val;
}

static double read_float64(FILE* file) {
    double val;
    fread(&val, sizeof(val), 1, file);
    return val;
}

// String Reader
static gguf_string_t read_string(FILE* file) {
    gguf_string_t str;
    str.len = read_uint64(file);
    str.string = malloc(str.len + 1);
    if (str.string == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    fread(str.string, 1, str.len, file);
    str.string[str.len] = '\0';
    return str;
}

// Read metadata value (Handles Recursive Types)
static void read_metadata_value(FILE* file, gguf_metadata_value_type type, 
                               gguf_metadata_value_t* value) {
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8:
            value->uint8 = fgetc(file);
            break;
        case GGUF_METADATA_VALUE_TYPE_INT8:
            value->int8 = fgetc(file);
            break;
        case GGUF_METADATA_VALUE_TYPE_UINT16:
            value->uint16 = read_uint16(file);
            break;
        case GGUF_METADATA_VALUE_TYPE_INT16:
            value->int16 = read_uint16(file); // Same size, will be cast
            break;
        case GGUF_METADATA_VALUE_TYPE_UINT32:
            value->uint32 = read_uint32(file);
            break;
        case GGUF_METADATA_VALUE_TYPE_INT32:
            value->int32 = read_uint32(file); // Same size
            break;
        case GGUF_METADATA_VALUE_TYPE_FLOAT32:
            value->float32 = read_float32(file);
            break;
        case GGUF_METADATA_VALUE_TYPE_BOOL:
            value->bool_ = (fgetc(file) != 0);
            break;
        case GGUF_METADATA_VALUE_TYPE_STRING:
            value->string = read_string(file);
            break;
        case GGUF_METADATA_VALUE_TYPE_ARRAY: {
            gguf_metadata_value_type elem_type = read_uint32(file);
            uint64_t len = read_uint64(file);
            value->array.type = elem_type;
            value->array.len = len;
            
            // Allocate space for array elements
            gguf_metadata_value_t* array = malloc(len * sizeof(gguf_metadata_value_t));
            if (array == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
            
            // Read each element
            for (uint64_t i = 0; i < len; i++) {
                read_metadata_value(file, elem_type, &array[i]);
            }
            value->array.array = array;
            break;
        }
        case GGUF_METADATA_VALUE_TYPE_UINT64:
            value->uint64 = read_uint64(file);
            break;
        case GGUF_METADATA_VALUE_TYPE_INT64:
            value->int64 = read_uint64(file); // Same size
            break;
        case GGUF_METADATA_VALUE_TYPE_FLOAT64:
            value->float64 = read_float64(file);
            break;
        default:
            fprintf(stderr, "Unknown metadata type: %u\n", type);
            exit(EXIT_FAILURE);
    }
}

// Free metadata value memory
// Recursive value cleaner
static void free_metadata_value(gguf_metadata_value_type type, gguf_metadata_value_t* value) {
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_STRING:
            free(value->string.string);
            break;
        case GGUF_METADATA_VALUE_TYPE_ARRAY:
            if (value->array.array != NULL) {
                for (uint64_t i = 0; i < value->array.len; i++) {
                    // Recursively free array elements
                    gguf_metadata_value_t* elem = 
                        &((gguf_metadata_value_t*)value->array.array)[i];
                    free_metadata_value(value->array.type, elem);
                }
                free(value->array.array);
            }
            break;
        default:
            // Primitive types don't need freeing
            break;
    }
}

// Free GGUF file structure
void free_gguf_file(gguf_file_t* file) {
    if (file == NULL) return;
    
    // Free metadata
    if (file->header.metadata_kv != NULL) {
        for (uint64_t i = 0; i < file->header.metadata_kv_count; i++) {
            gguf_metadata_kv_t* kv = &file->header.metadata_kv[i];
            free(kv->key.string);
            free_metadata_value(kv->value_type, &kv->value);
        }
        free(file->header.metadata_kv);
    }
    
    // Free tensor infos
    if (file->tensor_infos != NULL) {
        for (uint64_t i = 0; i < file->header.tensor_count; i++) {
            gguf_tensor_info_t* tensor = &file->tensor_infos[i];
            free(tensor->name.string);
            free(tensor->dimensions);
        }
        free(file->tensor_infos);
    }
}



// Load GGUF file
// Main Loader Function
gguf_file_t* load_gguf_file(const char* filename) {
    // 1. OPen file and validate magic number
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }
    
    // 2. Read header
    gguf_file_t* gguf = malloc(sizeof(gguf_file_t));
    if (!gguf) {
        fclose(file);
        return NULL;
    }
    memset(gguf, 0, sizeof(gguf_file_t));
    
    // Read and validate magic number
    char magic[5] = {0};
    fread(magic, 1, 4, file);
    if (memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "Invalid GGUF file (bad magic)\n");
        fclose(file);
        free(gguf);
        return NULL;
    }
    gguf->header.magic = *(uint32_t*)magic;
    
    // Read header
    gguf->header.version = read_uint32(file);
    gguf->header.tensor_count = read_uint64(file);
    gguf->header.metadata_kv_count = read_uint64(file);
    
    // Read metadata
    // 3. Read metadata key-value pairs
    gguf->header.metadata_kv = malloc(gguf->header.metadata_kv_count * 
                                     sizeof(gguf_metadata_kv_t));
    if (!gguf->header.metadata_kv) {
        fclose(file);
        free(gguf);
        return NULL;
    }
    
    for (uint64_t i = 0; i < gguf->header.metadata_kv_count; i++) {
        gguf_metadata_kv_t* kv = &gguf->header.metadata_kv[i];
        kv->key = read_string(file);
        kv->value_type = read_uint32(file);
        read_metadata_value(file, kv->value_type, &kv->value);
    }
    
    // 4. Read tensor infos
    gguf->tensor_infos = malloc(gguf->header.tensor_count * 
                               sizeof(gguf_tensor_info_t));
    if (!gguf->tensor_infos) {
        fclose(file);
        free_gguf_file(gguf);
        return NULL;
    }
    
    for (uint64_t i = 0; i < gguf->header.tensor_count; i++) {
        gguf_tensor_info_t* tensor = &gguf->tensor_infos[i];
        tensor->name = read_string(file);
        tensor->n_dimensions = read_uint32(file);
        
        tensor->dimensions = malloc(tensor->n_dimensions * sizeof(uint64_t));
        if (!tensor->dimensions) {
            fclose(file);
            free_gguf_file(gguf);
            return NULL;
        }
        
        for (uint32_t j = 0; j < tensor->n_dimensions; j++) {
            tensor->dimensions[j] = read_uint64(file);
        }
        
        tensor->type = read_uint32(file);
        tensor->offset = read_uint64(file);
    }
    
    // 5. Calculate data offset (end of tensor infos + padding)
    gguf->data_offset = ftell(file);
    gguf->data_offset = align_offset(gguf->data_offset);
    
    fclose(file);
    return gguf;
}

void print_metadata_value(gguf_metadata_value_type type, const gguf_metadata_value_t* value) {
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8:
            printf("%u", value->uint8);
            break;
        case GGUF_METADATA_VALUE_TYPE_INT8:
            printf("%d", value->int8);
            break;
        case GGUF_METADATA_VALUE_TYPE_UINT16:
            printf("%u", value->uint16);
            break;
        case GGUF_METADATA_VALUE_TYPE_INT16:
            printf("%d", value->int16);
            break;
        case GGUF_METADATA_VALUE_TYPE_UINT32:
            printf("%u", value->uint32);
            break;
        case GGUF_METADATA_VALUE_TYPE_INT32:
            printf("%d", value->int32);
            break;
        case GGUF_METADATA_VALUE_TYPE_FLOAT32:
            printf("%.4g", value->float32);
            break;
        case GGUF_METADATA_VALUE_TYPE_BOOL:
            printf(value->bool_ ? "true" : "false");
            break;
        case GGUF_METADATA_VALUE_TYPE_STRING:
            printf("\"%s\"", value->string.string);
            break;
        case GGUF_METADATA_VALUE_TYPE_ARRAY: {
            printf("[");
            for (uint64_t i = 0; i < value->array.len; i++) {
                if (i > 0) printf(", ");
                print_metadata_value(value->array.type, (const gguf_metadata_value_t*)((const char*)value->array.array + i * sizeof(gguf_metadata_value_t)));
            }
            printf("]");
            break;
        }
        case GGUF_METADATA_VALUE_TYPE_UINT64:
            printf("%" PRIu64, value->uint64);
            break;
        case GGUF_METADATA_VALUE_TYPE_INT64:
            printf("%" PRId64, value->int64);
            break;
        case GGUF_METADATA_VALUE_TYPE_FLOAT64:
            printf("%.4g", value->float64);
            break;
        default:
            printf("<unknown type>");
    }
}

// Print GGUF file structure
void print_gguf_file(const gguf_file_t* file) {
    if (!file) return;
    
    // Print header info
    printf("╔═════════════════════════════════════╗\n");
    printf("║        GGUF File Information        ║\n");
    printf("╠═════════════════════════════════════╣\n");
    printf("║ Magic: GGUF                        ║\n");
    printf("║ Version: %-26u ║\n", file->header.version);
    printf("║ Tensors: %-25" PRIu64 " ║\n", file->header.tensor_count);
    printf("║ Metadata Entries: %-18" PRIu64 " ║\n", file->header.metadata_kv_count);
    printf("║ Data Offset: 0x%016" PRIx64 "   ║\n", file->data_offset);
    printf("╠═════════════════════════════════════╣\n");
    
    // Print metadata
    printf("║ Metadata:\n");
    for (uint64_t i = 0; i < file->header.metadata_kv_count; i++) {
        const gguf_metadata_kv_t* kv = &file->header.metadata_kv[i];
        printf("║   %-20s: [%s] ", kv->key.string, metadata_type_name(kv->value_type));
        print_metadata_value(kv->value_type, &kv->value);
        printf("\n");
    }
    
    // Print tensors
    printf("╠═════════════════════════════════════╣\n");
    printf("║ Tensors:\n");
    for (uint64_t i = 0; i < file->header.tensor_count; i++) {
        const gguf_tensor_info_t* tensor = &file->tensor_infos[i];
        printf("║   %s [", tensor->name.string);
        for (uint32_t j = 0; j < tensor->n_dimensions; j++) {
            printf("%" PRIu64, tensor->dimensions[j]);
            if (j < tensor->n_dimensions - 1) printf(" x ");
        }
        printf("] %s @ 0x%016" PRIx64 "\n", 
               ggml_type_name(tensor->type), 
               tensor->offset);
    }
    printf("╚═════════════════════════════════════╝\n");
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <gguf-file>\n", argv[0]);
        return 1;
    }
    
    gguf_file_t* file = load_gguf_file(argv[1]);
    if (!file) {
        return 1;
    }
    
    print_gguf_file(file);
    free_gguf_file(file);
    free(file);
    
    return 0;
}
