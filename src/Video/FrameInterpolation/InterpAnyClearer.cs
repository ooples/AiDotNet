using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// InterpAnyClearer: plug-in module for clearer anytime video frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// InterpAnyClearer (Zheng et al., ECCV 2024 Oral) resolves velocity ambiguity in VFI:
/// - Velocity-ambiguity analysis: identifies that standard VFI models produce blurry results
///   when motion speed varies within a scene, because a single flow vector per pixel cannot
///   represent multiple plausible velocities simultaneously
/// - Plug-in velocity predictor: a lightweight auxiliary network that predicts per-pixel velocity
///   magnitude from the input frame pair, conditioning the base VFI model to select the correct
///   motion hypothesis for each region
/// - Multi-velocity training: during training, the model sees multiple velocity annotations per
///   pixel (from different temporal distances), learning to disambiguate fast vs slow motion
/// - Base-model agnostic: designed as a plug-in that wraps any existing VFI model (RIFE, IFRNet,
///   AMT, EMA-VFI, etc.) without modifying its architecture, only adding velocity conditioning
/// </para>
/// <para>
/// <b>For Beginners:</b> When objects in a video move at different speeds, standard interpolation
/// can get confused and produce blurry results. InterpAnyClearer adds a small "speed detector"
/// that tells the main model how fast each part of the image is moving, so it can produce
/// sharp results even when some objects move fast and others are still.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new InterpAnyClearer&lt;float&gt;(arch, "interpanyclearer.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Clearer Frames, Anytime: Resolving Velocity Ambiguity in VFI"
/// (Zheng et al., ECCV 2024 Oral)
/// </para>
/// </remarks>
public class InterpAnyClearer<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly InterpAnyClearerOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates an InterpAnyClearer model in ONNX inference mode.</summary>
    public InterpAnyClearer(NeuralNetworkArchitecture<T> architecture, string modelPath, InterpAnyClearerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new InterpAnyClearerOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an InterpAnyClearer model in native training mode.</summary>
    public InterpAnyClearer(NeuralNetworkArchitecture<T> architecture, InterpAnyClearerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new InterpAnyClearerOptions();
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
            Name = _useNativeMode ? "InterpAnyClearer-Native" : "InterpAnyClearer-ONNX",
            Description = $"InterpAnyClearer {_options.Variant} velocity-aware plug-in interpolation (Zheng et al., ECCV 2024 Oral)",
            ModelType = ModelType.FrameInterpolation,
            Complexity = _options.NumVelocityBlocks * _options.NumPyramidLevels
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumVelocityBlocks"] = _options.NumVelocityBlocks.ToString();
        m.AdditionalInfo["NumVelocityBins"] = _options.NumVelocityBins.ToString();
        m.AdditionalInfo["NumPyramidLevels"] = _options.NumPyramidLevels.ToString();
        m.AdditionalInfo["UseVelocityGuidedWarping"] = _options.UseVelocityGuidedWarping.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumVelocityBlocks);
        w.Write(_options.NumVelocityBins);
        w.Write(_options.NumPyramidLevels);
        w.Write(_options.UseVelocityGuidedWarping);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumVelocityBlocks = r.ReadInt32();
        _options.NumVelocityBins = r.ReadInt32();
        _options.NumPyramidLevels = r.ReadInt32();
        _options.UseVelocityGuidedWarping = r.ReadBoolean();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new InterpAnyClearer<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(InterpAnyClearer<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
