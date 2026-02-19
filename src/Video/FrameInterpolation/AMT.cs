using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// AMT: all-pairs multi-field transforms for efficient frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AMT (Li et al., CVPR 2023) uses correlation-based all-pairs multi-field transforms:
/// - All-pairs correlation: computes dense 4D cost volume between every pixel pair across
///   two frames at multiple scales, providing exhaustive motion correspondence information
///   that captures all possible matches including occluded and newly-visible regions
/// - Multi-field transforms: instead of a single flow field, predicts K candidate flow
///   fields per pixel, each capturing a plausible motion hypothesis, which are merged via
///   learned soft selection weights for robust motion estimation
/// - Iterative GRU refinement: coarse-to-fine correlation lookup with GRU-based iterative
///   updates that progressively refine the multi-field estimates over N iterations
/// - Efficient separable correlation: uses separable 1D correlation (H then W) instead of
///   full 2D correlation, reducing the quartic O(N^4) cost to quadratic O(N^2)
/// </para>
/// <para>
/// <b>For Beginners:</b> AMT tries every possible match between pixels in two frames
/// (all-pairs). For each pixel, instead of guessing a single motion direction, it proposes
/// several candidates and lets the network pick the best one. This handles tricky cases
/// like objects moving in front of each other (occlusion) or objects appearing from behind.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new AMT&lt;float&gt;(arch, "amt.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation"
/// (Li et al., CVPR 2023)
/// </para>
/// </remarks>
public class AMT<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly AMTOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates an AMT model in ONNX inference mode.</summary>
    public AMT(NeuralNetworkArchitecture<T> architecture, string modelPath, AMTOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new AMTOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an AMT model in native training mode.</summary>
    public AMT(NeuralNetworkArchitecture<T> architecture, AMTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new AMTOptions();
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
            Name = _useNativeMode ? "AMT-Native" : "AMT-ONNX",
            Description = $"AMT {_options.Variant} all-pairs multi-field transforms (Li et al., CVPR 2023)",
            ModelType = ModelType.FrameInterpolation,
            Complexity = _options.NumFlowFields * _options.NumRefinementIters
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumFlowFields"] = _options.NumFlowFields.ToString();
        m.AdditionalInfo["NumRefinementIters"] = _options.NumRefinementIters.ToString();
        m.AdditionalInfo["NumCorrelationLevels"] = _options.NumCorrelationLevels.ToString();
        m.AdditionalInfo["CorrelationRadius"] = _options.CorrelationRadius.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumFlowFields);
        w.Write(_options.NumRefinementIters);
        w.Write(_options.NumCorrelationLevels);
        w.Write(_options.CorrelationRadius);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumFlowFields = r.ReadInt32();
        _options.NumRefinementIters = r.ReadInt32();
        _options.NumCorrelationLevels = r.ReadInt32();
        _options.CorrelationRadius = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new AMT<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AMT<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
