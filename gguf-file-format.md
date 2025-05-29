# GGUF File format

After Training phase, the models based on the llama.cpp architecture can be exchanged using the GGUF (GPT-Generated Unified Format) format.

GGUF is a new standard for storing models during inference. GGUF is a binary format designed for fast loading and saving of models, and for ease of reading.

GGUF inherits from GGML, its predecessor, but the GGML format had several shortcomings and has been completely depreciated and replaced by the GGUF format.

If you are interesting by reading the specification, you can go to Github [here](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).

## GGML format
The GGML format is a tensor library written in C language allowing the quantization of LLMs and enabling it to run on CPU-powered hardware. GGML format is designed for llama architecture.

> Note
> Tensor in Machine Learning refers to multidimensional array organization and representation of data. This is called M-way array or data tensor.
The key characteristic of data tensor is that Artificial Neural Networks, the foundation of LLMs, can analyze it.

By default, the parameters in LLMs are represented with 32-bit floating numbers. However, the GGML library can convert it into a 16-bit floating-point representation thus reducing the memory requirement by 50% to load & run a LLM.

This process is known as **quantization**. The quantization reduces the quality of LLM inference but is also a tradeoff between having GPU compute & high precision vs CPU compute and low precision.

GGML library also supports integer quantization (e.g. 4-bit, 5-bit, 8-bit, etc.) that can further reduce the memory and compute power required to run LLMs locally on the end user’s system or edge devices.

Finally, GGML is not just a tensor library but also a file format for LLMs quantized with GGML.

## GGUF format
GGUF is a new file format for the LLMs created with GGML library, which was announced in August 2023. GGUF is a highly efficient improvement over the GGML format that offers better tokenization, support for special tokens and better metadata storage.

> Note
> In LLMs, the special tokens act as delimiters to signify the end of the user prompt or system prompt or any special instructions.
Special tokens are generally defined in prompt templates of a given LLM and help create more effective prompts & are also useful during LLM fine-tuning.

GGUF format is a generic design, by opposition to GGML. GGUF extends the compatibility with non-llama architecture models like Falcon, Bloom, Phi, Mistral, etc.

GGUF saves all the metadata, data, and hyperparameters in a single file, like for GGML. However GGUF is designed to be more extensible & flexible allowing the addition of new features without breaking anything.

In addition, the key difference with the old format is the fact that GGUF uses key-value structure for the hyperparameters (called now metadata) rather than a list of untyped values.

> Note
> Hyperparameters are settings you can use to tune the algorithms during the training phase of the Models

## GGUF file structure
The GGUF has the following structure:

**GGUF Header** composed by:
1. Magic number (4 bytes) to announce that this is a GGUF file,
2. Version (4 bytes) to specify the version of the format implemented,
3. Tensor count (8 bytes): the number of tensors in the file,
4. Metadata key-value pairs count (8 bytes): the number of metadata key-value pairs,

**Metadata key-value pairs**

**Tensor information** composed by:

1. Name: name of the tensor, a standard GGUF string,
2. Dimension: number of dimensions in the tensor,
3. Type: very important information. This is the type of the tensor to determine the Quantization level of the Model.
4. Offset: position of the tensor data in the file

**Padding:** additional data adding during the training to the tensors in order to enhance the processing during inference and add consistency. The padding strategy consists to add for example zeros or specific values…

**Tensor data:** binary data close or identical to the data in the original model file, but may be different due to quantization or other optimizations for inference. If any deviation, it is recorded in the metadata.

> Note
> A higher number of tensors indicates generally a more complex Model architecture. Those Models might have a larger capacity to capture intricate relationships within the data and potentially generate more sophisticated predictions.

> Note
> The type of the tensor (tensor type) refers to the data type used to represent the numerical value within the tensor.

Common Tensor Types in GGUF:

- F32 (Single-precision floating-point): This is a widely used data type offering a balance between precision and efficiency. It stores numbers with 32 bits of information, allowing for a good range and decimal point representation.
- INT8 (8-bit signed integer): This data type uses only 8 bits per value, making it very memory efficient. However, it has a limited range and cannot represent decimal points. It’s often used in quantized models for faster inference on resource-constrained devices.
- BF16 (Brain Floating-Point 16-bit): This emerging data type offers a compromise between F32 and INT8. It uses 16 bits per value, providing higher precision than INT8 while being more memory-efficient than F32.

Here is the illustration of the GGUF structure:

<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/img/gguf-header.webp">
  <img alt="nan corp" src="/img/gguf-header.webp" width="100%" height="100%">
</picture>

</div>

[Source](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) 

## GGUF Models

In the Metadata section, the GGUF brings general.architecture key-value pair that lists the supported types of Models:

- [Llama](https://huggingface.co/docs/transformers/main/en/model_doc/llama )
- [mpt](https://huggingface.co/docs/transformers/main/en/model_doc/mpt )
- [gptneox](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox )
- [gptj](https://huggingface.co/docs/transformers/main/en/model_doc/gptj )
- [gpt2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2 )
- [bloom](https://huggingface.co/docs/transformers/main/en/model_doc/bloom )
- [falcon](https://huggingface.co/docs/transformers/main/en/model_doc/falcon )
- [mamba](https://huggingface.co/docs/transformers/main/en/model_doc/mamba )
- [rwkv](https://huggingface.co/docs/transformers/main/en/model_doc/rwkv )

They are all Text Models.

## GGUF model quantization

In the Tensor information section, the type is very important because it defines the level of quantization of the Model. The level of quantization depends on a list of values (`ggml_type`) that defines the quality and accuracy of the Model.

In the GGUF specification, the list of values are the followings:

```c
enum ggml_type: uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
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
};
```
