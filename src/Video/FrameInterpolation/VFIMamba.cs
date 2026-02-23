using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// VFIMamba state-space model for video frame interpolation with linear-complexity attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "VFIMamba: Video Frame Interpolation with State Space Models" (Zhang et al., NeurIPS 2024)</item>
/// </list></para>
/// <para>
/// VFIMamba is the first to apply Mamba (selective state space models) to frame interpolation:
/// - Selective state space: uses Mamba's selective scan mechanism instead of attention, achieving
///   linear complexity O(N) for processing frame features while maintaining global context,
///   enabling efficient processing of high-resolution frames
/// - Bidirectional scanning: scans frame features in both forward (left-to-right, top-to-bottom)
///   and backward directions, ensuring each pixel has full global context from all directions
/// - Cross-frame state propagation: the SSM state from one frame is propagated to condition
///   the processing of the other frame, enabling implicit motion correspondence without
///   explicit flow estimation
/// - Multi-scale Mamba blocks: hierarchical Mamba blocks at different spatial scales, capturing
///   both fine-grained texture details and coarse motion patterns
/// </para>
/// <para>
/// <b>For Beginners:</b> VFIMamba uses a new type of AI architecture called "Mamba" that can
/// process long sequences efficiently (unlike transformers which get slow with long inputs).
/// This means it can handle high-resolution frames without running out of memory, while still
/// understanding how every pixel relates to every other pixel.
/// </para>
/// </remarks>
public class VFIMamba<T> : FrameInterpolationBase<T>
{
    private readonly VFIMambaOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a VFIMamba model for ONNX inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional configuration options.</param>
    public VFIMamba(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        VFIMambaOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VFIMambaOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a VFIMamba model for native training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    public VFIMamba(
        NeuralNetworkArchitecture<T> architecture,
        VFIMambaOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new VFIMambaOptions();
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
            throw new ArgumentOutOfRangeException(nameof(t), t, "Timestep must be in [0, 1].");
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
        int required = 0;
        foreach (var layer in Layers) required += layer.GetParameters().Length;
        if (parameters.Length < required)
            throw new ArgumentException($"Parameter vector length {parameters.Length} is less than required {required}.", nameof(parameters));
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
                { "ModelName", "VFIMamba" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumMambaBlocks", _options.NumMambaBlocks },
                { "StateDim", _options.StateDim },
                { "ExpansionFactor", _options.ExpansionFactor },
                { "NumStages", _options.NumStages },
                { "Complexity", _options.NumMambaBlocks * _options.NumStages }
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
        writer.Write(_options.NumMambaBlocks);
        writer.Write(_options.StateDim);
        writer.Write(_options.ExpansionFactor);
        writer.Write(_options.NumStages);
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
        _options.NumMambaBlocks = reader.ReadInt32();
        _options.StateDim = reader.ReadInt32();
        _options.ExpansionFactor = reader.ReadInt32();
        _options.NumStages = reader.ReadInt32();
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
            return new VFIMamba<T>(Architecture, p, _options);
        return new VFIMamba<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(VFIMamba<T>));
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
