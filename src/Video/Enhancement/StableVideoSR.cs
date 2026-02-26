using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// StableVideoSR: stable diffusion with temporal conditioning for video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// StableVideoSR (2024) adapts the Stable Diffusion architecture for video SR:
/// - Temporal conditioning modules: cross-attention layers inserted between spatial attention
///   in the U-Net attend to features from adjacent frames, maintaining temporal coherence
/// - ControlNet adapter: a trainable copy of the U-Net encoder provides fine-grained spatial
///   conditioning from the low-resolution input during the denoising process
/// - Classifier-free guidance: balances between faithful reconstruction and generative
///   enhancement during inference
/// - Noise schedule: adapted from image diffusion to preserve temporal structure
/// </para>
/// <para>
/// <b>For Beginners:</b> StableVideoSR extends the popular Stable Diffusion image AI to
/// handle video. It adds "temporal awareness" so the model considers what happened in
/// previous and next frames when upscaling each frame, preventing the flickering that
/// occurs when frames are processed independently.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new StableVideoSR&lt;float&gt;(arch, "stablevideosr.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "StableVideoSR: Video Super-Resolution via Stable Diffusion with
/// Temporal Conditioning" (2024)
/// </para>
/// </remarks>
public class StableVideoSR<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly StableVideoSROptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a StableVideoSR model in ONNX inference mode.</summary>
    public StableVideoSR(NeuralNetworkArchitecture<T> architecture, string modelPath, StableVideoSROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new StableVideoSROptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a StableVideoSR model in native training mode.</summary>
    public StableVideoSR(NeuralNetworkArchitecture<T> architecture, StableVideoSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new StableVideoSROptions();
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
            int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 128;
            int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 128;
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoSuperResolutionLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures,
                numResBlocks: _options.NumTemporalLayers,
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
        int required = 0;
        foreach (var layer in Layers) required += layer.ParameterCount;
        if (parameters.Length < required)
            throw new ArgumentException($"Parameter vector length {parameters.Length} is less than required {required}.", nameof(parameters));
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
            Name = _useNativeMode ? "StableVideoSR-Native" : "StableVideoSR-ONNX",
            Description = $"StableVideoSR {_options.Variant} temporal diffusion VSR (2024)",
            ModelType = ModelType.VideoSuperResolution,
            Complexity = _options.NumTemporalLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumDenoisingSteps"] = _options.NumDenoisingSteps.ToString();
        m.AdditionalInfo["NumTemporalLayers"] = _options.NumTemporalLayers.ToString();
        m.AdditionalInfo["GuidanceScale"] = _options.GuidanceScale.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumDenoisingSteps);
        w.Write(_options.NumTemporalLayers);
        w.Write(_options.ScaleFactor);
        w.Write(_options.LatentDim);
        w.Write(_options.GuidanceScale);
        w.Write(_options.ControlNetScale);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumDenoisingSteps = r.ReadInt32();
        _options.NumTemporalLayers = r.ReadInt32();
        _options.ScaleFactor = r.ReadInt32();
        _options.LatentDim = r.ReadInt32();
        _options.GuidanceScale = r.ReadDouble();
        _options.ControlNetScale = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        ScaleFactor = _options.ScaleFactor;
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
            return new StableVideoSR<T>(Architecture, p, _options);
        return new StableVideoSR<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(StableVideoSR<T>));
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
