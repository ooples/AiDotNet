using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Denoising;

/// <summary>
/// FloRNN optical-flow-guided recurrent video denoising.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Flowing Recurrent Network for Video Denoising" (Li et al., AAAI 2022)</item>
/// </list></para>
/// <para><b>For Beginners:</b> FloRNN (Flow-guided Recurrent Neural Network) denoises video frames by using optical flow to align neighboring frames before applying recurrent processing. This flow-guided approach preserves temporal consistency.</para>
/// <para>
/// FloRNN uses optical flow to guide recurrent denoising, warping previous hidden states
/// for temporal alignment before feeding them to ConvLSTM/ConvGRU units, with occlusion-aware
/// gating to suppress unreliable aligned features.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a FloRNN model for optical-flow-guided video denoising
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Generative,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3);
/// var options = new FloRNNOptions();
/// var flornn = new FloRNN&lt;double&gt;(architecture, options);
///
/// // Or load a pre-trained ONNX model for inference
/// var flornnOnnx = new FloRNN&lt;double&gt;(architecture, "flornn_model.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Unidirectional Video Denoising by Mimicking Backward Recurrent Modules with Look-ahead Forward Ones",
    "https://arxiv.org/abs/2204.05532",
    Year = 2022,
    Authors = "Junyi Li, Xiaohe Wu, Zhenxing Niu, Wangmeng Zuo")]
public partial class FloRNN<T> : VideoDenoisingBase<T>
{
    private readonly FloRNNOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// The released FloRNN model consumes and emits normalized image tensors.
    /// </summary>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => VideoPixelInputDomain.NormalizedValue;

    /// <inheritdoc/>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => VideoPixelInputDomain.NormalizedValue;

    /// <summary>
    /// Creates a FloRNN model for ONNX inference.
    /// </summary>
    public FloRNN(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        FloRNNOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new FloRNNOptions();
        _useNativeMode = false;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FloRNN model for native training and inference.
    /// </summary>
    public FloRNN(
        NeuralNetworkArchitecture<T> architecture,
        FloRNNOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new FloRNNOptions();
        _useNativeMode = true;
        // The released FloRNN training script uses torch.optim.Adam at 1e-4
        // with Adam's standard fixed moments (train_models/sRGB_train.py).
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                UseAMSGrad = false
            });
        InitializeLayers();
    }

    /// <inheritdoc/>
    public override Tensor<T> Denoise(Tensor<T> noisyFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(noisyFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            int ch = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 128;
            int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 128;
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoDenoisingLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures));
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => rawFrames;

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => modelOutput;

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessTarget(Tensor<T> expectedOutput) => expectedOutput;

    /// <inheritdoc/>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? base.GetOrCreateBaseOptimizer();

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "FloRNN" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumRecurrentLayers", _options.NumRecurrentLayers },
                { "HiddenDim", _options.HiddenDim },
                { "NumFlowScales", _options.NumFlowScales }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FloRNN<T>));
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }
}
