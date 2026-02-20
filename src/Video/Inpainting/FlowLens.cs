using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Inpainting;

/// <summary>
/// FlowLens optical-flow-guided video inpainting with flow completion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "FlowLens: Seeing Beyond the FoV via Optical Flow Completion" (Xu et al., ECCV 2022)</item>
/// </list></para>
/// <para>
/// FlowLens decouples motion estimation from pixel synthesis by first completing optical flow
/// in masked regions, then using the completed flow for temporal propagation of known pixels,
/// followed by a refinement network for remaining holes, achieving sharp and temporally
/// consistent inpainting.
/// </para>
/// </remarks>
public class FlowLens<T> : VideoInpaintingBase<T>
{
    private readonly FlowLensOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a FlowLens model for ONNX inference.
    /// </summary>
    public FlowLens(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        FlowLensOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new FlowLensOptions();
        _useNativeMode = false;
        SupportsTemporalPropagation = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FlowLens model for native training and inference.
    /// </summary>
    public FlowLens(
        NeuralNetworkArchitecture<T> architecture,
        FlowLensOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new FlowLensOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SupportsTemporalPropagation = true;
        InitializeLayers();
    }

    /// <inheritdoc/>
    public override Tensor<T> Inpaint(Tensor<T> frames, Tensor<T> masks)
    {
        ThrowIfDisposed();
        var combined = ConcatFramesAndMasks(frames, masks);
        var output = IsOnnxMode ? RunOnnxInference(combined) : Forward(combined);
        return output;
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoInpaintingLayers(
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
        if (!_useNativeMode) throw new NotSupportedException("Parameter updates are not supported in ONNX mode.");
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
            ModelType = ModelType.VideoInpainting,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "FlowLens" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumFlowIters", _options.NumFlowIters },
                { "NumLevels", _options.NumLevels },
                { "NumResBlocks", _options.NumResBlocks }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_options.Variant);
        writer.Write(_options.NumFeatures);
        writer.Write(_options.NumFlowIters);
        writer.Write(_options.NumLevels);
        writer.Write(_options.NumResBlocks);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.Variant = (VideoModelVariant)reader.ReadInt32();
        _options.NumFeatures = reader.ReadInt32();
        _options.NumFlowIters = reader.ReadInt32();
        _options.NumLevels = reader.ReadInt32();
        _options.NumResBlocks = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new FlowLens<T>(Architecture, p, _options);
        return new FlowLens<T>(Architecture, _options);
    }

    private static Tensor<T> ConcatFramesAndMasks(Tensor<T> frames, Tensor<T> masks)
    {
        if (frames.Rank != 4)
            throw new ArgumentException($"Frames must be rank 4 [N, C, H, W], got rank {frames.Rank}.", nameof(frames));
        if (masks.Rank != 4)
            throw new ArgumentException($"Masks must be rank 4 [N, 1, H, W], got rank {masks.Rank}.", nameof(masks));
        int n = frames.Shape[0];
        int c = frames.Shape[1];
        int h = frames.Shape[2];
        int w = frames.Shape[3];
        if (masks.Shape[0] != n || masks.Shape[2] != h || masks.Shape[3] != w)
            throw new ArgumentException($"Masks spatial dimensions must match frames. Frames: [{n},{c},{h},{w}], Masks: [{masks.Shape[0]},{masks.Shape[1]},{masks.Shape[2]},{masks.Shape[3]}].", nameof(masks));
        var combined = new Tensor<T>([n, c + 1, h, w]);
        int frameSize = c * h * w;
        int maskSize = h * w;
        int combinedSize = (c + 1) * h * w;
        for (int f = 0; f < n; f++)
        {
            for (int i = 0; i < frameSize; i++)
                combined.Data.Span[f * combinedSize + i] = frames.Data.Span[f * frameSize + i];
            for (int i = 0; i < maskSize; i++)
                combined.Data.Span[f * combinedSize + frameSize + i] = masks.Data.Span[f * maskSize + i];
        }
        return combined;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FlowLens<T>));
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }
}
