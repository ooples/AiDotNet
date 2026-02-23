using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Stabilization;

/// <summary>
/// FuSta hybrid full-frame video stabilization with warping and outpainting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "FuSta: Hybrid Approach for Full-frame Video Stabilization" (Liu et al., 2021)</item>
/// </list></para>
/// <para><b>For Beginners:</b> FuSta (Fusion Stabilization) stabilizes video by fusing multiple stabilization strategies including trajectory smoothing and homography warping for robust results.</para>
/// <para>
/// FuSta achieves full-frame stabilization through a two-stage approach: first warping frames
/// using optical-flow-based motion compensation, then using a neural outpainting network
/// to fill missing border regions, avoiding the field-of-view loss of traditional cropping.
/// </para>
/// </remarks>
public class FuSta<T> : VideoStabilizationBase<T>
{
    private readonly FuStaOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a FuSta model for ONNX inference.
    /// </summary>
    public FuSta(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        FuStaOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new FuStaOptions();
        _useNativeMode = false;
        SupportsFullFrame = true;
        CropRatio = 0.0;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FuSta model for native training and inference.
    /// </summary>
    public FuSta(
        NeuralNetworkArchitecture<T> architecture,
        FuStaOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new FuStaOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SupportsFullFrame = true;
        CropRatio = 0.0;
        InitializeLayers();
    }

    /// <inheritdoc/>
    public override Tensor<T> Stabilize(Tensor<T> unstableFrames)
    {
        ThrowIfDisposed();
        var output = IsOnnxMode ? RunOnnxInference(unstableFrames) : Forward(unstableFrames);
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoStabilizationLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w));
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
            ModelType = ModelType.VideoStabilization,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "FuSta" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumLevels", _options.NumLevels },
                { "NumResBlocks", _options.NumResBlocks },
                { "NumHeads", _options.NumHeads }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_options.Variant);
        writer.Write(_options.NumFeatures);
        writer.Write(_options.NumLevels);
        writer.Write(_options.NumResBlocks);
        writer.Write(_options.NumHeads);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.Variant = (VideoModelVariant)reader.ReadInt32();
        _options.NumFeatures = reader.ReadInt32();
        _options.NumLevels = reader.ReadInt32();
        _options.NumResBlocks = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new FuSta<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FuSta<T>));
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
