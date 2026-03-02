using AiDotNet.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// ONNX-NER: Generic ONNX Runtime-based NER model for high-performance inference with any
/// exported NER model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ONNX-NER provides a model-agnostic wrapper for running any NER model exported to ONNX format
/// through ONNX Runtime. This enables high-performance inference regardless of the original
/// training framework (PyTorch, TensorFlow, JAX, etc.).
///
/// <b>ONNX Runtime Optimizations:</b>
/// - <b>Graph optimizations:</b> Operator fusion (MatMul+Add, Conv+BN), constant folding,
///   and redundant computation elimination
/// - <b>Hardware acceleration:</b> CUDA, TensorRT, DirectML, OpenVINO, CoreML execution providers
/// - <b>Quantization:</b> INT8/FP16 inference for reduced memory and faster execution
/// - <b>Memory optimization:</b> Memory-efficient attention patterns, memory arena pre-allocation
/// - <b>Threading:</b> Intra-op and inter-op parallelism configuration
///
/// <b>Supported Model Sources:</b>
/// - HuggingFace Transformers (exported via optimum or torch.onnx.export)
/// - PyTorch models (exported via torch.onnx.export)
/// - TensorFlow/Keras models (exported via tf2onnx)
/// - Custom models exported to ONNX format
///
/// <b>Performance (typical speedups over PyTorch):</b>
/// - CPU: 2-4x faster with graph optimizations
/// - GPU (CUDA): 1.5-3x faster with TensorRT
/// - INT8 quantized: 3-5x faster than FP32 with ~0.5% F1 loss
///
/// <b>Common ONNX NER Models:</b>
/// - dslim/bert-base-NER (ONNX): ~92.4% F1, ~5ms/sentence on GPU
/// - elastic/distilbert-base-cased-finetuned-conll03-english: ~91.1% F1, ~2ms/sentence on GPU
/// - Jean-Baptiste/camembert-ner (French NER): ~90.5% F1
/// </para>
/// <para>
/// <b>For Beginners:</b> ONNX-NER is not a specific NER architecture, but rather a way to run
/// any NER model that has been exported to the ONNX format. ONNX is like a "universal translator"
/// for neural network models - it lets you train in any framework and run inference efficiently.
/// Use ONNX-NER when you have a pre-trained ONNX model file and want the fastest possible
/// inference, or when you want to deploy NER models without depending on PyTorch/TensorFlow.
/// </para>
/// </remarks>
public class ONNXNER<T> : TransformerNERBase<T>
{
    /// <summary>
    /// Creates an ONNX-NER model in ONNX inference mode.
    /// This is the primary constructor since ONNX-NER is designed for inference.
    /// </summary>
    public ONNXNER(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions? options = null)
        : base(architecture, modelPath, options ?? new TransformerNEROptions(),
            "ONNX-NER", "ONNX Runtime")
    {
    }

    /// <summary>
    /// Creates an ONNX-NER model in native training mode.
    /// Note: ONNX-NER is primarily designed for inference. Use this constructor only when
    /// building a model to be exported to ONNX format after training.
    /// </summary>
    public ONNXNER(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new TransformerNEROptions(),
            "ONNX-NER", "ONNX Runtime", optimizer)
    {
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new TransformerNEROptions(NEROptions);
        if (!UseNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new ONNXNER<T>(Architecture, p, optionsCopy);
        return new ONNXNER<T>(Architecture, optionsCopy);
    }
}
