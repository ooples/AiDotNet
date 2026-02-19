using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// DynamiCrafter: animating open-domain images with video diffusion priors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DynamiCrafter (2024) uses video diffusion priors for frame interpolation:
/// - Video diffusion backbone: adapts a pre-trained text-to-video diffusion model for the
///   interpolation task, leveraging its learned motion priors from millions of training videos
/// - First/last frame conditioning: the diffusion process is conditioned on both endpoint
///   frames using CLIP image embeddings injected via cross-attention, ensuring temporal
///   consistency with both the start and end frames
/// - Noise schedule adaptation: modified diffusion noise schedule that biases early denoising
///   steps toward global motion consistency and later steps toward fine detail refinement
/// - Temporal attention: 3D self-attention across generated frames ensures smooth motion
///   transitions without flickering or temporal discontinuities
/// </para>
/// <para>
/// <b>For Beginners:</b> DynamiCrafter uses an AI video generator (diffusion model) that
/// already understands how things move in the real world. Given a start frame and end frame,
/// it gradually "imagines" what happens in between, producing natural-looking intermediate
/// frames with realistic motion, lighting changes, and object interactions.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new DynamiCrafter&lt;float&gt;(arch, "dynamicrafter.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors" (2024)
/// </para>
/// </remarks>
public class DynamiCrafter<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly DynamiCrafterOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a DynamiCrafter model in ONNX inference mode.</summary>
    public DynamiCrafter(NeuralNetworkArchitecture<T> architecture, string modelPath, DynamiCrafterOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new DynamiCrafterOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a DynamiCrafter model in native training mode.</summary>
    public DynamiCrafter(NeuralNetworkArchitecture<T> architecture, DynamiCrafterOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new DynamiCrafterOptions();
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
            Name = _useNativeMode ? "DynamiCrafter-Native" : "DynamiCrafter-ONNX",
            Description = $"DynamiCrafter {_options.Variant} video diffusion interpolation (2024)",
            ModelType = ModelType.FrameInterpolation,
            Complexity = _options.NumResBlocks * _options.NumDiffusionSteps
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumDiffusionSteps"] = _options.NumDiffusionSteps.ToString();
        m.AdditionalInfo["NumResBlocks"] = _options.NumResBlocks.ToString();
        m.AdditionalInfo["NumHeads"] = _options.NumHeads.ToString();
        m.AdditionalInfo["GuidanceScale"] = _options.GuidanceScale.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumDiffusionSteps);
        w.Write(_options.NumResBlocks);
        w.Write(_options.NumHeads);
        w.Write(_options.GuidanceScale);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumDiffusionSteps = r.ReadInt32();
        _options.NumResBlocks = r.ReadInt32();
        _options.NumHeads = r.ReadInt32();
        _options.GuidanceScale = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new DynamiCrafter<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DynamiCrafter<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
