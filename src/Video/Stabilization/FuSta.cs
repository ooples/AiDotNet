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

namespace AiDotNet.Video.Stabilization;

/// <summary>
/// FuSta hybrid full-frame video stabilization with warping and outpainting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "FuSta: Hybrid Approach for Full-frame Video Stabilization" (Liu et al., 2021)</item>
/// </list></para>
/// <para><b>For Beginners:</b> FuSta (Fusion Stabilization) stabilizes video by fusing multiple stabilization strategies including trajectory smoothing and homography warping for robust results.</para>
/// <para>
/// FuSta achieves full-frame stabilization through a two-stage approach: first warping frames
/// using optical-flow-based motion compensation, then using a neural outpainting network
/// to fill missing border regions, avoiding the field-of-view loss of traditional cropping.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a FuSta model for full-frame video stabilization
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3);
/// var options = new FuStaOptions();
/// var fuSta = new FuSta&lt;double&gt;(architecture, options);
///
/// // Or load a pre-trained ONNX model for inference
/// var fuStaOnnx = new FuSta&lt;double&gt;(architecture, "fusta_model.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Hybrid Neural Fusion for Full-frame Video Stabilization",
    "https://arxiv.org/abs/2102.06205",
    Year = 2021,
    Authors = "Yu-Lun Liu, Wei-Sheng Lai, Ming-Hsuan Yang, Yung-Yu Chuang, Jia-Bin Huang")]
public partial class FuSta<T> : VideoStabilizationBase<T>
{
    private readonly FuStaOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a FuSta model for ONNX inference.
    /// </summary>
    public FuSta(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        FuStaOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new FuStaOptions();
        _useNativeMode = false;
        SupportsFullFrame = true;
        CropRatio = 0.0;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FuSta model for native training and inference.
    /// </summary>
    public FuSta(
        NeuralNetworkArchitecture<T> architecture,
        FuStaOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new FuStaOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });
        SupportsFullFrame = true;
        CropRatio = 0.0;
        InitializeLayers();
    }

    /// <inheritdoc/>
    public override Tensor<T> Stabilize(Tensor<T> unstableFrames)
    {
        ThrowIfDisposed();
        var output = IsOnnxMode ? RunOnnxInference(unstableFrames) : Forward(unstableFrames);
        return output;
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
            // FuSta is a full-frame neural-rendering / fusion stabilizer (synthesis paradigm): it
            // produces a stabilized frame of the same dimensions as the input, so use the
            // length-preserving encoder-decoder rather than the 6-affine-param regressor.
            Layers.AddRange(LayerHelper<T>.CreateSynthesisVideoStabilizationLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w));
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "FuSta" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumLevels", _options.NumLevels },
                { "NumResBlocks", _options.NumResBlocks },
                { "NumHeads", _options.NumHeads }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FuSta<T>));
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
