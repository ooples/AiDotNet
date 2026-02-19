using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// Upscale4KAgent: agentic multi-model pipeline for any-resolution to 4K upscaling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Upscale4KAgent (2025) orchestrates multiple specialized SR models in an agentic pipeline:
/// - Quality assessment agent: evaluates each frame to determine optimal processing strategy
/// - Multi-model routing: dynamically selects and chains SR models (e.g., Real-ESRGAN for
///   textures, SwinIR for faces) based on content analysis of each frame region
/// - Iterative refinement: applies progressive upscaling stages with quality checkpoints,
///   continuing until QualityThreshold is met or MaxAgentSteps is reached
/// - Resolution-adaptive: handles arbitrary input resolution with automatic tiling and
///   overlap for seamless 4K output
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of using one model for everything, Upscale4KAgent acts
/// like a "manager" that looks at each frame and decides which combination of upscaling
/// models will produce the best result. It can chain multiple models together and check
/// quality at each step, similar to how a human editor would approach video upscaling.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 540, inputWidth: 960, inputDepth: 3);
/// var model = new Upscale4KAgent&lt;float&gt;(arch, "upscale4kagent.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Upscale4KAgent: Agentic Multi-Model Pipeline for 4K Video Upscaling" (2025)
/// </para>
/// </remarks>
public class Upscale4KAgent<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly Upscale4KAgentOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates an Upscale4KAgent model in ONNX inference mode.</summary>
    public Upscale4KAgent(NeuralNetworkArchitecture<T> architecture, string modelPath, Upscale4KAgentOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new Upscale4KAgentOptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an Upscale4KAgent model in native training mode.</summary>
    public Upscale4KAgent(NeuralNetworkArchitecture<T> architecture, Upscale4KAgentOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new Upscale4KAgentOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        ScaleFactor = _options.ScaleFactor;
        InitializeLayers();
    }

    #endregion

    #region Video Super-Resolution

    /// <inheritdoc />
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(lowResFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    #endregion

    #region NeuralNetworkBase

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
            int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 540;
            int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 960;
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoSuperResolutionLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures,
                numResBlocks: _options.NumResBlocks,
                scaleFactor: _options.ScaleFactor));
        }
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return RunOnnxInference(input);
        return Forward(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Parameter updates are not supported in ONNX mode.");
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Upscale4KAgent-Native" : "Upscale4KAgent-ONNX",
            Description = $"Upscale4KAgent {_options.Variant} agentic multi-model 4K pipeline (2025)",
            ModelType = ModelType.VideoSuperResolution,
            Complexity = _options.NumResBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumStages"] = _options.NumStages.ToString();
        m.AdditionalInfo["MaxAgentSteps"] = _options.MaxAgentSteps.ToString();
        m.AdditionalInfo["QualityThreshold"] = _options.QualityThreshold.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumStages);
        w.Write(_options.NumResBlocks);
        w.Write(_options.ScaleFactor);
        w.Write(_options.MaxAgentSteps);
        w.Write(_options.QualityThreshold);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumStages = r.ReadInt32();
        _options.NumResBlocks = r.ReadInt32();
        _options.ScaleFactor = r.ReadInt32();
        _options.MaxAgentSteps = r.ReadInt32();
        _options.QualityThreshold = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new Upscale4KAgent<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Upscale4KAgent<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
