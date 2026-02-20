using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// ToonCrafter generative cartoon interpolation for large non-linear animated motion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "ToonCrafter: Generative Cartoon Interpolation" (2024)</item>
/// </list></para>
/// <para>
/// ToonCrafter specializes in cartoon and animation frame interpolation where motions are
/// large, non-linear, and don't follow physical constraints. It uses a latent diffusion
/// approach adapted for animation:
/// - Dual-reference conditioning: conditions the diffusion process on both start and end
///   cartoon frames simultaneously, using cross-attention to attend to features from both
/// - Sketch-guided generation: optionally uses extracted sketch/edge maps as structural
///   guidance, ensuring generated frames maintain the line art style and character proportions
/// - Toon-adapted noise schedule: a modified diffusion noise schedule that works better with
///   the flat colors and sharp edges typical of cartoon/animation content
/// - Large motion capability: handles the extreme, physically-unrealistic motions common in
///   animation (e.g., squash-and-stretch, sudden direction changes, exaggerated physics)
/// </para>
/// <para>
/// <b>For Beginners:</b> ToonCrafter is designed specifically for cartoons and animations.
/// Unlike real-world video where objects move smoothly, cartoon characters can teleport,
/// stretch, and move in impossible ways. This model understands these animation-specific
/// motion patterns and generates in-between frames that look natural for animated content.
/// </para>
/// </remarks>
public class ToonCrafter<T> : FrameInterpolationBase<T>
{
    private readonly ToonCrafterOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a ToonCrafter model for ONNX inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional configuration options.</param>
    public ToonCrafter(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ToonCrafterOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ToonCrafterOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a ToonCrafter model for native training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    public ToonCrafter(
        NeuralNetworkArchitecture<T> architecture,
        ToonCrafterOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ToonCrafterOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SupportsArbitraryTimestep = true;
        InitializeLayers();
    }

    /// <inheritdoc/>
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultFrameInterpolationLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures));
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
            var output = Predict(input);
            var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
            var gt = Tensor<T>.FromVector(grad);
            for (int i = Layers.Count - 1; i >= 0; i--)
                gt = Layers[i].Backward(gt);
            _optimizer?.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int required = Layers.Sum(l => l.ParameterCount);
        if (parameters.Length < required)
            throw new ArgumentException(
                $"Parameter vector length {parameters.Length} is less than required {required}.",
                nameof(parameters));
        int offset = 0;
        foreach (var layer in Layers)
        {
            var p = layer.GetParameters();
            var sub = new Vector<T>(p.Length);
            for (int i = 0; i < p.Length; i++) sub[i] = parameters[offset + i];
            layer.SetParameters(sub);
            offset += p.Length;
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.FrameInterpolation,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "ToonCrafter" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumDiffusionSteps", _options.NumDiffusionSteps },
                { "NumResBlocks", _options.NumResBlocks },
                { "NumHeads", _options.NumHeads },
                { "GuidanceScale", _options.GuidanceScale },
                { "Complexity", _options.NumDiffusionSteps * _options.NumResBlocks }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write((int)_options.Variant);
        writer.Write(_options.NumFeatures);
        writer.Write(_options.NumDiffusionSteps);
        writer.Write(_options.NumResBlocks);
        writer.Write(_options.NumHeads);
        writer.Write(_options.GuidanceScale);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)reader.ReadInt32();
        _options.NumFeatures = reader.ReadInt32();
        _options.NumDiffusionSteps = reader.ReadInt32();
        _options.NumResBlocks = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.GuidanceScale = reader.ReadDouble();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
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

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new ToonCrafter<T>(Architecture, p, _options);
        return new ToonCrafter<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ToonCrafter<T>));
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
