using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// DRVI: disentangled representations for video interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DRVI (2024) disentangles content and motion for video interpolation:
/// - Disentangled encoders: separate encoders for content (appearance, texture, color) and
///   motion (displacement, deformation) that process frame pairs independently
/// - Content encoder: extracts appearance features invariant to motion, shared across all
///   timesteps so the model doesn't re-extract appearance at each interpolation point
/// - Motion encoder: captures inter-frame displacement fields at multiple scales, enabling
///   the model to handle both global camera motion and local object motion
/// - Disentangled decoder: recombines content and motion representations with learned gating
///   at each scale, allowing fine control over which content features are warped by which
///   motion components
/// </para>
/// <para>
/// <b>For Beginners:</b> DRVI separates "what things look like" from "how things move".
/// By processing appearance and motion independently, it can better handle cases where
/// objects look similar but move differently, or where the same object appears in different
/// lighting conditions across frames.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new DRVI&lt;float&gt;(arch, "drvi.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "DRVI: Disentangled Representations for Video Interpolation" (2024)
/// </para>
/// </remarks>
public class DRVI<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly DRVIOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a DRVI model in ONNX inference mode.</summary>
    public DRVI(NeuralNetworkArchitecture<T> architecture, string modelPath, DRVIOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new DRVIOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a DRVI model in native training mode.</summary>
    public DRVI(NeuralNetworkArchitecture<T> architecture, DRVIOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new DRVIOptions();
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
            throw new ArgumentOutOfRangeException(nameof(t), t, "Timestep must be in [0, 1].");
        var f0 = PreprocessFrames(frame0);
        var f1 = PreprocessFrames(frame1);
        var concat = ConcatenateFeatures(f0, f1);
        var output = IsOnnxMode ? RunOnnxInference(concat) : Forward(concat);
        var result = PostprocessOutput(output);

        // Apply timestep for arbitrary-time interpolation (model predicts midpoint frame)
        if (Math.Abs(t - 0.5) > 1e-6)
        {
            var boundary = t < 0.5 ? frame0 : frame1;
            double alpha = t < 0.5 ? 2.0 * t : 2.0 * (1.0 - t);
            var a = NumOps.FromDouble(alpha);
            var oneMinusA = NumOps.FromDouble(1.0 - alpha);
            int len = Math.Min(result.Length, boundary.Length);
            for (int i = 0; i < len; i++)
                result.Data.Span[i] = NumOps.Add(
                    NumOps.Multiply(a, result.Data.Span[i]),
                    NumOps.Multiply(oneMinusA, boundary.Data.Span[i]));
        }

        return result;
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
            Name = _useNativeMode ? "DRVI-Native" : "DRVI-ONNX",
            Description = $"DRVI {_options.Variant} disentangled content-motion interpolation (2024)",
            ModelType = ModelType.FrameInterpolation,
            Complexity = _options.NumContentBlocks + _options.NumMotionBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumContentBlocks"] = _options.NumContentBlocks.ToString();
        m.AdditionalInfo["NumMotionBlocks"] = _options.NumMotionBlocks.ToString();
        m.AdditionalInfo["NumDecoderBlocks"] = _options.NumDecoderBlocks.ToString();
        m.AdditionalInfo["NumScales"] = _options.NumScales.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumContentBlocks);
        w.Write(_options.NumMotionBlocks);
        w.Write(_options.NumDecoderBlocks);
        w.Write(_options.NumScales);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumContentBlocks = r.ReadInt32();
        _options.NumMotionBlocks = r.ReadInt32();
        _options.NumDecoderBlocks = r.ReadInt32();
        _options.NumScales = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
        {
            OnnxModel?.Dispose();
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
        }
        else if (_useNativeMode)
        {
            Layers.Clear();
            InitializeLayers();
        }
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new DRVI<T>(Architecture, p, _options);
        return new DRVI<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DRVI<T>));
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
