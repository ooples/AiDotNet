using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// ABME: asymmetric bilateral motion estimation for video frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ABME (Park et al., ICCV 2021) uses asymmetric bilateral motion estimation:
/// - Bilateral motion estimation: estimates motion from the target time to both input frames
///   simultaneously (t to 0 and t to 1), rather than from input frames toward the target,
///   which is more natural for the interpolation task
/// - Asymmetric motion model: the two bilateral motion fields are NOT constrained to be
///   symmetric; each has its own magnitude and direction, correctly handling non-linear motion
///   paths (acceleration, deceleration, curved trajectories)
/// - Iterative GRU refinement: iteratively refines both bilateral flows with separate update
///   heads that can correct each flow independently over N iterations
/// - Context-aware synthesis: the final frame is synthesized by combining bilaterally warped
///   features with a learned blending mask that accounts for occlusion and motion boundaries
/// </para>
/// <para>
/// <b>For Beginners:</b> Most methods assume motion is symmetric (if something moves right
/// from frame 0, it moves left by the same amount from frame 1). But real motion isn't
/// symmetric -- a ball speeding up moves more in the second half. ABME estimates motion
/// independently in both directions, so it handles acceleration, deceleration, and curved
/// paths much better than symmetric methods.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new ABME&lt;float&gt;(arch, "abme.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Asymmetric Bilateral Motion Estimation for Video Frame Interpolation"
/// (Park et al., ICCV 2021)
/// </para>
/// </remarks>
public class ABME<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly ABMEOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates an ABME model in ONNX inference mode.</summary>
    public ABME(NeuralNetworkArchitecture<T> architecture, string modelPath, ABMEOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ABMEOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an ABME model in native training mode.</summary>
    public ABME(NeuralNetworkArchitecture<T> architecture, ABMEOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ABMEOptions();
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
            Name = _useNativeMode ? "ABME-Native" : "ABME-ONNX",
            Description = $"ABME {_options.Variant} asymmetric bilateral motion estimation (Park et al., ICCV 2021)",
            ModelType = ModelType.FrameInterpolation,
            Complexity = _options.NumResBlocks * _options.NumRefinementIters
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumResBlocks"] = _options.NumResBlocks.ToString();
        m.AdditionalInfo["NumRefinementIters"] = _options.NumRefinementIters.ToString();
        m.AdditionalInfo["NumPyramidLevels"] = _options.NumPyramidLevels.ToString();
        m.AdditionalInfo["AsymmetricMotion"] = _options.AsymmetricMotion.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumResBlocks);
        w.Write(_options.NumRefinementIters);
        w.Write(_options.NumPyramidLevels);
        w.Write(_options.AsymmetricMotion);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumResBlocks = r.ReadInt32();
        _options.NumRefinementIters = r.ReadInt32();
        _options.NumPyramidLevels = r.ReadInt32();
        _options.AsymmetricMotion = r.ReadBoolean();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new ABME<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ABME<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
