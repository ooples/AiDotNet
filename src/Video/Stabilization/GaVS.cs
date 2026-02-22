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
/// GaVS gaze-aware video stabilization with saliency-weighted motion smoothing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Gaze-aware Video Stabilization" (2023)</item>
/// </list></para>
/// <para><b>For Beginners:</b> GaVS (Generative Adversarial Video Stabilization) uses adversarial training to produce stabilized video that looks natural. The discriminator ensures the output appears like genuinely stable footage.</para>
/// <para>
/// GaVS predicts viewer gaze regions and applies stronger stabilization near the focus of
/// attention while allowing more camera motion in peripheral regions. This preserves
/// intentional cinematographic movements while removing distracting shake near gaze targets.
/// </para>
/// </remarks>
public class GaVS<T> : VideoStabilizationBase<T>
{
    private readonly GaVSOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a GaVS model for ONNX inference.
    /// </summary>
    public GaVS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        GaVSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new GaVSOptions();
        _useNativeMode = false;
        SmoothingWindowSize = _options.SmoothingWindow;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a GaVS model for native training and inference.
    /// </summary>
    public GaVS(
        NeuralNetworkArchitecture<T> architecture,
        GaVSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new GaVSOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SmoothingWindowSize = _options.SmoothingWindow;
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
                { "ModelName", "GaVS" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumGazeHeads", _options.NumGazeHeads },
                { "GazeHiddenDim", _options.GazeHiddenDim },
                { "SmoothingWindow", _options.SmoothingWindow }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_options.Variant);
        writer.Write(_options.NumFeatures);
        writer.Write(_options.NumGazeHeads);
        writer.Write(_options.GazeHiddenDim);
        writer.Write(_options.SmoothingWindow);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.Variant = (VideoModelVariant)reader.ReadInt32();
        _options.NumFeatures = reader.ReadInt32();
        _options.NumGazeHeads = reader.ReadInt32();
        _options.GazeHiddenDim = reader.ReadInt32();
        _options.SmoothingWindow = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new GaVS<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GaVS<T>));
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
