using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// SoftSplat: softmax splatting for video frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SoftSplat (Niklaus &amp; Liu, CVPR 2020) uses softmax splatting for forward warping:
/// - Forward warping with softmax: source pixels are "splatted" to target positions, with
///   conflicts resolved via softmax weighting instead of backward warping
/// - Importance metric Z: each source pixel carries a learned importance metric Z that controls
///   its softmax weight, automatically learning foreground/background occlusion ordering
/// - Feature-space splatting: splatting is performed on deep feature maps rather than raw
///   pixels, providing richer representations for the synthesis network
/// - GridNet synthesis: a GridNet-style synthesis network takes splatted features and produces
///   the final interpolated frame with residual refinement
/// </para>
/// <para>
/// <b>For Beginners:</b> SoftSplat uses a smart voting system (softmax) where each pixel gets
/// a learned "importance score" to decide which pixel wins when there's a conflict at the same
/// target position, naturally handling which objects appear in front of others.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new SoftSplat&lt;float&gt;(arch, "softsplat.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Softmax Splatting for Video Frame Interpolation"
/// (Niklaus &amp; Liu, CVPR 2020)
/// </para>
/// </remarks>
public class SoftSplat<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly SoftSplatOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a SoftSplat model in ONNX inference mode.</summary>
    public SoftSplat(NeuralNetworkArchitecture<T> architecture, string modelPath, SoftSplatOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SoftSplatOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a SoftSplat model in native training mode.</summary>
    public SoftSplat(NeuralNetworkArchitecture<T> architecture, SoftSplatOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SoftSplatOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SupportsArbitraryTimestep = true;
        InitializeLayers();
    }

    #endregion

    #region Frame Interpolation

    /// <summary>
    /// Interpolates between two frames using softmax splatting at timestep t.
    /// Note: In native mode, the default layer stack produces midpoint (t=0.5) interpolation.
    /// In ONNX mode, arbitrary timestep support depends on the loaded model weights.
    /// </summary>
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
            Layers.AddRange(Architecture.Layers);
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
            Name = _useNativeMode ? "SoftSplat-Native" : "SoftSplat-ONNX",
            Description = $"SoftSplat {_options.Variant} softmax splatting interpolation (Niklaus & Liu, CVPR 2020)",
            ModelType = ModelType.FrameInterpolation,
            Complexity = _options.NumGridNetLevels * _options.NumResBlocksPerRow
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumGridNetLevels"] = _options.NumGridNetLevels.ToString();
        m.AdditionalInfo["NumResBlocksPerRow"] = _options.NumResBlocksPerRow.ToString();
        m.AdditionalInfo["NumFeatureBlocks"] = _options.NumFeatureBlocks.ToString();
        m.AdditionalInfo["UseImportanceMetric"] = _options.UseImportanceMetric.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumGridNetLevels);
        w.Write(_options.NumResBlocksPerRow);
        w.Write(_options.NumFeatureBlocks);
        w.Write(_options.UseImportanceMetric);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumGridNetLevels = r.ReadInt32();
        _options.NumResBlocksPerRow = r.ReadInt32();
        _options.NumFeatureBlocks = r.ReadInt32();
        _options.UseImportanceMetric = r.ReadBoolean();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (IsOnnxMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new SoftSplat<T>(Architecture, mp, _options);
        return new SoftSplat<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SoftSplat<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
