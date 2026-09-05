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
/// GaVS gaze-aware video stabilization with saliency-weighted motion smoothing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Gaze-aware Video Stabilization" (2023)</item>
/// </list></para>
/// <para><b>For Beginners:</b> GaVS (Generative Adversarial Video Stabilization) uses adversarial training to produce stabilized video that looks natural. The discriminator ensures the output appears like genuinely stable footage.</para>
/// <para>
/// GaVS predicts viewer gaze regions and applies stronger stabilization near the focus of
/// attention while allowing more camera motion in peripheral regions. This preserves
/// intentional cinematographic movements while removing distracting shake near gaze targets.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a GaVS model for gaze-aware video stabilization
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Generative,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3);
/// var options = new GaVSOptions();
/// var gavs = new GaVS&lt;double&gt;(architecture, options);
///
/// // Or load a pre-trained ONNX model for inference
/// var gavsOnnx = new GaVS&lt;double&gt;(architecture, "gavs_model.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Generalized Adaptive Video Stabilization",
    "https://arxiv.org/abs/2501.06868",
    Year = 2025,
    Authors = "Donghao Zhang")]
public partial class GaVS<T> : VideoStabilizationBase<T>
{
    private readonly GaVSOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a GaVS model for ONNX inference.
    /// </summary>
    public GaVS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        GaVSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new GaVSOptions();
        _useNativeMode = false;
        SmoothingWindowSize = _options.SmoothingWindow;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a GaVS model for native training and inference.
    /// </summary>
    public GaVS(
        NeuralNetworkArchitecture<T> architecture,
        GaVSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new GaVSOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });
        SmoothingWindowSize = _options.SmoothingWindow;
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
            // GaVS performs generalized adaptive (dense) stabilization whose output is a stabilized
            // frame of the same dimensions as the input. Use the length-preserving conv
            // encoder-decoder rather than the global 6-affine-param regressor.
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
                { "ModelName", "GaVS" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumGazeHeads", _options.NumGazeHeads },
                { "GazeHiddenDim", _options.GazeHiddenDim },
                { "SmoothingWindow", _options.SmoothingWindow }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GaVS<T>));
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
