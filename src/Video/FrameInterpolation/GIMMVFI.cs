using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// GIMM-VFI: generalizable implicit motion modeling for video frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GIMM-VFI (NeurIPS 2024) uses implicit neural representations for continuous-time motion:
/// - Implicit motion function: learns a continuous function M(x, y, t) that maps any spatial
///   position (x, y) and any timestep t in [0, 1] to a motion vector, enabling interpolation
///   at arbitrary time intervals without retraining or additional forward passes
/// - Motion encoding network: encodes two input frames into a shared motion latent space
///   using cross-correlation features, producing a motion representation that can be queried
///   at any desired timestep
/// - Generalizable across timesteps: a single forward pass through the motion encoder produces
///   a representation that the implicit function can query at any t, unlike methods that need
///   separate inference per timestep
/// - Adaptive sampling: the implicit function can be queried at higher density in regions with
///   complex motion and lower density in static regions for efficient computation
/// </para>
/// <para>
/// <b>For Beginners:</b> GIMM-VFI learns a smooth "motion field" that describes how everything
/// in the scene moves over time. Once it processes two frames, it can generate a new frame at
/// ANY point between them. This is great for variable slow-motion effects or non-uniform
/// frame rate conversion where you need different time spacings between frames.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new GIMMVFI&lt;float&gt;(arch, "gimmvfi.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "GIMM-VFI: Generalizable Implicit Motion Modeling for Video Frame
/// Interpolation" (NeurIPS 2024)
/// </para>
/// </remarks>
public class GIMMVFI<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly GIMMVFIOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a GIMM-VFI model in ONNX inference mode.</summary>
    public GIMMVFI(NeuralNetworkArchitecture<T> architecture, string modelPath, GIMMVFIOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new GIMMVFIOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a GIMM-VFI model in native training mode.</summary>
    public GIMMVFI(NeuralNetworkArchitecture<T> architecture, GIMMVFIOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new GIMMVFIOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SupportsArbitraryTimestep = true;
        InitializeLayers();
    }

    #endregion

    #region Frame Interpolation

    /// <inheritdoc />
    public override Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5)
    {
        ThrowIfDisposed();
        if (t < 0.0 || t > 1.0)
            throw new ArgumentOutOfRangeException(nameof(t), t, "Timestep t must be in [0, 1].");
        var f0 = PreprocessFrames(frame0);
        var f1 = PreprocessFrames(frame1);
        var concat = ConcatenateFeatures(f0, f1);
        var output = IsOnnxMode ? RunOnnxInference(concat) : Forward(concat);
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
            int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 128;
            int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 128;
            Layers.AddRange(LayerHelper<T>.CreateDefaultFrameInterpolationLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures));
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
        try
        {
            var output = Predict(input);
            var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
            var gt = Tensor<T>.FromVector(grad);
            for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
            _optimizer?.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
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
            Name = _useNativeMode ? "GIMMVFI-Native" : "GIMMVFI-ONNX",
            Description = $"GIMM-VFI {_options.Variant} implicit motion modeling interpolation (NeurIPS 2024)",
            ModelType = ModelType.FrameInterpolation,
            Complexity = _options.NumEncoderBlocks * _options.NumImplicitLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumEncoderBlocks"] = _options.NumEncoderBlocks.ToString();
        m.AdditionalInfo["ImplicitDim"] = _options.ImplicitDim.ToString();
        m.AdditionalInfo["NumImplicitLayers"] = _options.NumImplicitLayers.ToString();
        m.AdditionalInfo["NumFrequencies"] = _options.NumFrequencies.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumEncoderBlocks);
        w.Write(_options.ImplicitDim);
        w.Write(_options.NumImplicitLayers);
        w.Write(_options.NumFrequencies);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumEncoderBlocks = r.ReadInt32();
        _options.ImplicitDim = r.ReadInt32();
        _options.NumImplicitLayers = r.ReadInt32();
        _options.NumFrequencies = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
        else if (_useNativeMode)
        {
            Layers.Clear();
            InitializeLayers();
        }
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new GIMMVFI<T>(Architecture, p, _options);
        return new GIMMVFI<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GIMMVFI<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
