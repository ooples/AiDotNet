using System.IO;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Denoising;

/// <summary>
/// FloRNN optical-flow-guided recurrent video denoising.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Flowing Recurrent Network for Video Denoising" (Li et al., AAAI 2022)</item>
/// </list></para>
/// <para><b>For Beginners:</b> FloRNN (Flow-guided Recurrent Neural Network) denoises video frames by using optical flow to align neighboring frames before applying recurrent processing. This flow-guided approach preserves temporal consistency.</para>
/// <para>
/// FloRNN uses optical flow to guide recurrent denoising, warping previous hidden states
/// for temporal alignment before feeding them to ConvLSTM/ConvGRU units, with occlusion-aware
/// gating to suppress unreliable aligned features.
/// </para>
/// </remarks>
public class FloRNN<T> : VideoDenoisingBase<T>
{
    private readonly FloRNNOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a FloRNN model for ONNX inference.
    /// </summary>
    public FloRNN(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        FloRNNOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new FloRNNOptions();
        _useNativeMode = false;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FloRNN model for native training and inference.
    /// </summary>
    public FloRNN(
        NeuralNetworkArchitecture<T> architecture,
        FloRNNOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new FloRNNOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <inheritdoc/>
    public override Tensor<T> Denoise(Tensor<T> noisyFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(noisyFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoDenoisingLayers(
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
            ModelType = ModelType.VideoDenoising,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "FloRNN" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumRecurrentLayers", _options.NumRecurrentLayers },
                { "HiddenDim", _options.HiddenDim },
                { "NumFlowScales", _options.NumFlowScales }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_options.Variant);
        writer.Write(_options.NumFeatures);
        writer.Write(_options.NumRecurrentLayers);
        writer.Write(_options.HiddenDim);
        writer.Write(_options.NumFlowScales);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.Variant = (VideoModelVariant)reader.ReadInt32();
        _options.NumFeatures = reader.ReadInt32();
        _options.NumRecurrentLayers = reader.ReadInt32();
        _options.HiddenDim = reader.ReadInt32();
        _options.NumFlowScales = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new FloRNN<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FloRNN<T>));
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
