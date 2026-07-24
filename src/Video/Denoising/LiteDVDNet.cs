using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Denoising;

/// <summary>
/// LiteDVDNet lightweight deep video denoising with depthwise separable convolutions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "LiteDVDNet: A Lightweight Deep Video Denoising Network" (2020)</item>
/// </list></para>
/// <para><b>For Beginners:</b> LiteDVDNet is a lightweight video denoiser designed for real-time performance. It achieves good denoising quality with significantly fewer parameters than full-scale models like DVDNet.</para>
/// <para>
/// LiteDVDNet is an efficient two-stage denoiser that first processes frames independently
/// then fuses temporal information, using depthwise separable convolutions for 8-10x
/// parameter reduction while maintaining quality.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a lightweight LiteDVDNet model for efficient video denoising
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3);
/// var options = new LiteDVDNetOptions();
/// var liteDvdNet = new LiteDVDNet&lt;double&gt;(architecture, options);
///
/// // Or load a pre-trained ONNX model for real-time inference
/// var liteDvdNetOnnx = new LiteDVDNet&lt;double&gt;(architecture, "litedvdnet_model.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("LiteDVDNet: A Lightweight Deep Video Denoising Network",
    "https://arxiv.org/abs/2004.08569",
    Year = 2020,
    Authors = "Matias Tassano, Julie Delon, Thomas Veit")]
public class LiteDVDNet<T> : VideoDenoisingBase<T>
{
    private readonly LiteDVDNetOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a LiteDVDNet model for ONNX inference.
    /// </summary>
    public LiteDVDNet(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        LiteDVDNetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new LiteDVDNetOptions();
        _useNativeMode = false;
        TemporalRadius = (_options.TemporalWindowSize - 1) / 2;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a LiteDVDNet model for native training and inference.
    /// </summary>
    public LiteDVDNet(
        NeuralNetworkArchitecture<T> architecture,
        LiteDVDNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new LiteDVDNetOptions();
        _useNativeMode = true;
        // Default optimizer honors the model's configured LearningRate — the bare AdamWOptimizer(this) ignored
        // it and ran at Adam's 0.001, which diverged (Training_ShouldReduceLoss saw loss explode 0.28 -> 150) —
        // and enables gradient clipping. Fully user-overridable via the optimizer parameter and
        // LiteDVDNetOptions.LearningRate (default lowered to the standard 1e-4 used for image/video denoisers,
        // since the model's [ResearchPaper] URL is a mis-citation to an unrelated paper). (#1789)
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            { InitialLearningRate = _options.LearningRate, EnableGradientClipping = true, MaxGradientNorm = 1.0 });
        TemporalRadius = (_options.TemporalWindowSize - 1) / 2;
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
                numFeatures: _options.NumFeatures,
                temporalFrames: _options.TemporalWindowSize));
        }
    }

    /// <summary>
    /// Routes TrainWithTape through the model's configured optimizer (default: AdamW at the denoiser-standard
    /// <see cref="LiteDVDNetOptions.LearningRate"/> = 1e-4 with gradient clipping) instead of the base Adam 1e-3
    /// default. 1e-3 explodes this deep conv architecture's loss (Training_ShouldReduceLoss saw 0.28 -> 150 even
    /// with the base global gradient-norm clip); the 10x-smaller step converges. Simply setting the private
    /// <c>_optimizer</c> field was inert until this override — the base trainer only consults
    /// GetOrCreateBaseOptimizer(). Fully user-overridable via the constructor's optimizer parameter. (#1789)
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? base.GetOrCreateBaseOptimizer();

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
            TrainWithTape(input, expected);
        }
        finally
        {
            SetTrainingMode(false);
        }
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
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "LiteDVDNet" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumBlocks", _options.NumBlocks },
                { "TemporalWindowSize", _options.TemporalWindowSize },
                { "ExpansionFactor", _options.ExpansionFactor }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_options.Variant);
        writer.Write(_options.NumFeatures);
        writer.Write(_options.NumBlocks);
        writer.Write(_options.TemporalWindowSize);
        writer.Write(_options.ExpansionFactor);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.Variant = (VideoModelVariant)reader.ReadInt32();
        _options.NumFeatures = reader.ReadInt32();
        _options.NumBlocks = reader.ReadInt32();
        _options.TemporalWindowSize = reader.ReadInt32();
        _options.ExpansionFactor = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new LiteDVDNet<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(LiteDVDNet<T>));
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
