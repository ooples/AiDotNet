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
/// STTN spatial-temporal transformer network for video inpainting with multi-scale attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Learning Joint Spatial-Temporal Transformations for Video Inpainting" (Zeng et al., ECCV 2020)</item>
/// </list></para>
/// <para>
/// STTN uses multi-scale spatial-temporal transformers that jointly search for and attend to
/// relevant patches across both space and time dimensions. Multi-head attention at multiple
/// feature scales enables both fine-grained texture transfer and large-scale structure completion.
/// </para>
/// </remarks>
public class STTN<T> : VideoInpaintingBase<T>
{
    private readonly STTNOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a STTN model for ONNX inference.
    /// </summary>
    public STTN(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        STTNOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new STTNOptions();
        _useNativeMode = false;
        SupportsTemporalPropagation = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a STTN model for native training and inference.
    /// </summary>
    public STTN(
        NeuralNetworkArchitecture<T> architecture,
        STTNOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new STTNOptions();
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
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--)
            gt = Layers[i].Backward(gt);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var p = layer.GetParameters();
            if (offset + p.Length > parameters.Length) break;
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
                { "ModelName", "STTN" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumTransformerLayers", _options.NumTransformerLayers },
                { "NumHeads", _options.NumHeads },
                { "NumScales", _options.NumScales }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_options.Variant);
        writer.Write(_options.NumFeatures);
        writer.Write(_options.NumTransformerLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.NumScales);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.Variant = (VideoModelVariant)reader.ReadInt32();
        _options.NumFeatures = reader.ReadInt32();
        _options.NumTransformerLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.NumScales = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new STTN<T>(Architecture, _options);
    }

    private static Tensor<T> ConcatFramesAndMasks(Tensor<T> frames, Tensor<T> masks)
    {
        int n = frames.Shape[0];
        int c = frames.Shape[1];
        int h = frames.Shape[2];
        int w = frames.Shape[3];
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
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(STTN<T>));
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
