<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>lm.c Architecture - Lightweight CPU Inference Engine</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        
        .diagram {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin: 25px 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border: 2px solid #e8f4f8;
        }
        
        .flow-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        
        .flow-step {
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            color: white;
            padding: 15px 25px;
            border-radius: 25px;
            min-width: 200px;
            text-align: center;
            font-weight: bold;
            position: relative;
            box-shadow: 0 4px 15px rgba(116, 185, 255, 0.3);
            transition: transform 0.3s ease;
        }
        
        .flow-step:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(116, 185, 255, 0.4);
        }
        
        .flow-step::after {
            content: '↓';
            position: absolute;
            bottom: -25px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 20px;
            color: #74b9ff;
        }
        
        .flow-step:last-child::after {
            display: none;
        }
        
        .architecture-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .component {
            background: linear-gradient(135deg, #fd79a8, #e84393);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(253, 121, 168, 0.3);
            transition: transform 0.3s ease;
        }
        
        .component:hover {
            transform: scale(1.05);
        }
        
        .component h3 {
            margin-top: 0;
            font-size: 1.3em;
        }
        
        .tensor-flow {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            margin: 20px 0;
        }
        
        .tensor-box {
            background: linear-gradient(135deg, #55a3ff, #003d82);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 5px;
            text-align: center;
            min-width: 120px;
            box-shadow: 0 4px 10px rgba(85, 163, 255, 0.3);
        }
        
        .arrow {
            font-size: 24px;
            color: #74b9ff;
            margin: 0 10px;
        }
        
        .code-highlight {
            background: #2d3748;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            margin: 15px 0;
            border-left: 4px solid #4299e1;
        }
        
        .layer-diagram {
            display: flex;
            flex-direction: column;
            align-items: center;
            background: linear-gradient(135deg, #a8e6cf, #7fcdcd);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
        }
        
        .layer-box {
            background: white;
            border: 2px solid #4a90e2;
            border-radius: 10px;
            padding: 15px;
            margin: 10px;
            min-width: 200px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .memory-diagram {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        
        .memory-block {
            background: linear-gradient(135deg, #ff9ff3, #f368e0);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 0.9em;
            box-shadow: 0 4px 10px rgba(255, 159, 243, 0.3);
        }
        
        .roadmap-list {
            background: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 20px;
            border-radius: 0 10px 10px 0;
            margin: 20px 0;
        }
        
        .roadmap-list h3 {
            color: #2e7d32;
            margin-top: 0;
        }
        
        .roadmap-list ul {
            list-style-type: none;
            padding-left: 0;
        }
        
        .roadmap-list li {
            background: white;
            margin: 10px 0;
            padding: 12px;
            border-radius: 8px;
            border-left: 3px solid #4caf50;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .roadmap-list li.completed::before {
            content: '✓';
            color: #4caf50;
            font-weight: bold;
            margin-right: 10px;
        }

        .roadmap-list li.pending::before {
            content: '➤';
            color: #2196f3;
            font-weight: bold;
            margin-right: 10px;
        }

        @media (max-width: 768px) {
            .tensor-flow {
                flex-direction: column;
            }
            .memory-diagram {
                grid-template-columns: repeat(2, 1fr);
            }
            .architecture-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 lm.c Architecture</h1>
        <p style="text-align: center; font-size: 1.2em; color: #666;">
            Lightweight CPU Inference Engine for Large Language Models
        </p>

        <h2>🏗️ Overall System Architecture</h2>
        <div class="diagram">
            <div class="flow-container">
                <div class="flow-step">GGUF File Loading</div>
                <div class="flow-step">Header & Metadata Parsing</div>
                <div class="flow-step">Tensor Info Loading</div>
                <div class="flow-step">Quantization Handling</div>
                <div class="flow-step">Transformer Execution</div>
                <div class="flow-step">Token Generation</div>
                <div class="flow-step">Text Output</div>
            </div>
        </div>

        <h2>🧩 Core Components</h2>
        <div class="architecture-grid">
            <div class="component">
                <h3>🗂️ GGUF Parser</h3>
                <p>Handles all GGUF metadata types and quantization formats with zero dependencies</p>
            </div>
            <div class="component">
                <h3>🧠 Quantization Engine</h3>
                <p>Supports 30+ GGML quantization formats from F32 to IQ1_M</p>
            </div>
            <div class="component">
                <h3>⚡ CPU Inference</h3>
                <p>Optimized transformer execution with minimal memory footprint</p>
            </div>
            <div class="component">
                <h3>🌐 Portable Runtime</h3>
                <p>Single-file C99 implementation runs anywhere</p>
            </div>
        </div>

        <h2>📊 GGUF File Structure</h2>
        <div class="diagram">
            <div class="tensor-flow">
                <div class="tensor-box">Magic Header<br/>(GGUF)</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Version<br/>(uint32)</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Tensor Count<br/>(uint64)</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Metadata<br/>(Key-Value)</div>
            </div>
            <div class="tensor-flow">
                <div class="tensor-box">Tensor Names<br/>(Strings)</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Dimensions<br/>(uint64[])</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Quantization<br/>(GGML_TYPE)</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Tensor Data<br/>(Aligned)</div>
            </div>
            <div class="code-highlight">
struct gguf_header_t {
    uint32_t magic;          // "GGUF"
    uint32_t version;         // Format version
    uint64_t tensor_count;    // Number of tensors
    uint64_t metadata_kv_count;
    gguf_metadata_kv_t metadata_kv[];
};</div>
        </div>

        <h2>🏭 Transformer Layer Architecture</h2>
        <div class="layer-diagram">
            <div class="layer-box" style="background: #fff3cd; border-color: #ffc107;">Token Embeddings</div>
            
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin: 20px 0;">
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div class="layer-box">RMS Normalization</div>
                    <div class="layer-box">Multi-Head Attention</div>
                    <div class="layer-box" style="background: #d1ecf1; border-color: #17a2b8;">Q/K/V Projections</div>
                </div>
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div class="layer-box">RMS Normalization</div>
                    <div class="layer-box">Feed Forward Network</div>
                    <div class="layer-box" style="background: #d4edda; border-color: #28a745;">SwiGLU Activation</div>
                </div>
            </div>
            
            <div class="layer-box" style="background: #f8d7da; border-color: #dc3545;">Output Projection</div>
            <div class="layer-box">Sampling & Decoding</div>
        </div>

        <h2>💾 Memory Efficient Design</h2>
        <div class="memory-diagram">
            <div class="memory-block">GGUF Parser<br/>Minimal overhead</div>
            <div class="memory-block">Quantization<br/>On-the-fly dequant</div>
            <div class="memory-block">Tensor Mapping<br/>Zero-copy access</div>
            <div class="memory-block">Activation Buffers<br/>Reusable memory</div>
            <div class="memory-block">KV Cache<br/>Optimized storage</div>
            <div class="memory-block">Token Buffers<br/>Efficient allocation</div>
            <div class="memory-block">SIMD Registers<br/>Vectorized ops</div>
            <div class="memory-block">Thread Pools<br/>Parallel execution</div>
        </div>

        <h2>🚦 Development Roadmap</h2>
        <div class="roadmap-list">
            <h3>lm.c Implementation Progress</h3>
            <ul>
                <li class="completed"><strong>GGUF File Loader:</strong> Complete with metadata extraction</li>
                <li class="pending"><strong>Tensor Data Mapping:</strong> Memory-mapped tensor access</li>
                <li class="pending"><strong>Quantization Kernels:</strong> All 30+ GGML formats</li>
                <li class="pending"><strong>Transformer Layers:</strong> CPU-optimized implementation</li>
                <li class="pending"><strong>Tokenization:</strong> Byte-pair encoding support</li>
                <li class="pending"><strong>Sampling:</strong> Temperature-based token selection</li>
                <li class="pending"><strong>SIMD Optimization:</strong> AVX2/NEON acceleration</li>
                <li class="pending"><strong>Thread Parallelism:</strong> Multi-core support</li>
                <li class="pending"><strong>Interactive Mode:</strong> Chat interface</li>
            </ul>
        </div>

        <h2>⚙️ Inference Workflow</h2>
        <div class="diagram">
            <div class="tensor-flow">
                <div class="tensor-box">Input Text</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Tokenization</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Embedding Lookup</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Transformer Layers</div>
            </div>
            <div class="tensor-flow">
                <div class="tensor-box">Layer Norm</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Attention</div>
                <div class="arrow">→</div>
                <div class="tensor-box">FFN</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Residual Add</div>
            </div>
            <div class="tensor-flow">
                <div class="tensor-box">Final Norm</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Output Projection</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Sampling</div>
                <div class="arrow">→</div>
                <div class="tensor-box">Generated Text</div>
            </div>
        </div>

        <h2>🚀 Performance Optimizations</h2>
        <div class="diagram" style="background: linear-gradient(135deg, #ffeaa7, #fdcb6e); color: #2d3436;">
            <h3 style="text-align: center; margin-top: 0;">CPU-Specific Enhancements</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px;">
                    <strong>🔢 Quantization Aware Ops</strong><br/>
                    Process quantized weights directly
                </div>
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px;">
                    <strong>🧮 Block Processing</strong><br/>
                    Optimized cache utilization
                </div>
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px;">
                    <strong>📦 Memory Mapping</strong><br/>
                    Zero-copy weight access
                </div>
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px;">
                    <strong>🧵 Thread Parallelism</strong><br/>
                    Layer-wise execution
                </div>
            </div>
        </div>

        <div style="text-align: center; margin-top: 40px; padding: 20px; background: linear-gradient(135deg, #a8e6cf, #7fcdcd); border-radius: 15px;">
            <h3 style="margin-top: 0; color: #2d3436;">✨ Efficient CPU Inference for Everyone!</h3>
            <p style="margin-bottom: 0; color: #636e72;">
                lm.c brings large language models to any device with a CPU - from servers to embedded systems.
                Pure C implementation, zero dependencies, maximum portability.
            </p>
        </div>
    </div>
</body>
</html>
