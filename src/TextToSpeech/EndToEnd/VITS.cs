using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// VITS: end-to-end TTS with conditional VAE, normalizing flows, and adversarial training for parallel high-quality synthesis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" (Kim et al., 2021)</item></list></para>
/// <para><b>For Beginners:</b> VITS (Variational Inference with adversarial learning for end-to-end
/// Text-to-Speech) was a breakthrough model that unified the entire TTS pipeline into a single
/// end-to-end architecture. Previous systems had separate components (text encoder, acoustic model,
/// vocoder) that were trained independently and could introduce errors at each boundary.
/// VITS combines three key techniques:
/// (1) A Conditional Variational Autoencoder (CVAE) that learns a latent representation of speech,
/// (2) Normalizing flows that transform simple distributions into complex speech distributions,
/// (3) Adversarial training (like GANs) that ensures the generated speech sounds natural.
/// During training, it uses Monotonic Alignment Search (MAS) to learn text-to-speech alignment
/// without external alignment tools. At inference, it generates high-quality speech in parallel
/// (all at once), making it fast while maintaining natural prosody.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech", "https://arxiv.org/abs/2106.06103", Year = 2021, Authors = "Kim et al.")]
public class VITS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly VITSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="VITS{T}"/> class in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional model configuration options.</param>
    public VITS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        VITSOptions? options = null) : base(architecture)
    {
        _options = options ?? new VITSOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="VITS{T}"/> class in native training/inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional model configuration options.</param>
    /// <param name="optimizer">Optional gradient-based optimizer for training.</param>
    public VITS(
        NeuralNetworkArchitecture<T> architecture,
        VITSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new VITSOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;

    /// <summary>
    /// Gets the hidden dimension size. Intentionally hides base HiddenDim to expose options-driven value.
    /// </summary>
    public new int HiddenDim => _options.HiddenDim;

    /// <summary>
    /// Gets the number of normalizing flow steps used to transform between prior and posterior distributions.
    /// </summary>
    public int NumFlowSteps => _options.NumFlowSteps;

    /// <summary>
    /// Synthesizes speech from text using VITS' VAE + normalizing flow + HiFi-GAN decoder pipeline.
    /// </summary>
    /// <param name="text">The input text to synthesize.</param>
    /// <returns>A tensor containing the generated waveform.</returns>
    /// <remarks>
    /// <para>Per the paper (Kim et al., 2021), the inference pipeline:</para>
    /// <para>(1) Text encoder: transformer encoder produces text hidden states h_text.</para>
    /// <para>(2) Stochastic duration predictor: predicts phoneme durations via a flow-based model.</para>
    /// <para>(3) Expand: h_text is expanded according to predicted durations to match audio frame rate.</para>
    /// <para>(4) Prior normalizing flow: transforms the expanded text features into latent variable z.</para>
    /// <para>(5) HiFi-GAN decoder: converts z into a raw audio waveform.</para>
    /// <para><b>For Beginners:</b> At inference time, VITS works in five steps: (1) encode the text into
    /// hidden representations, (2) predict how long each sound should last, (3) stretch the text
    /// representations to match those durations, (4) use normalizing flows to transform these into
    /// a rich latent representation that captures the nuances of speech, and (5) use the HiFi-GAN
    /// neural vocoder to convert that latent representation directly into an audio waveform.
    /// All of this happens in parallel (not one sample at a time), making VITS fast at inference.</para>
    /// </remarks>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var output = Predict(input);
        return PostprocessAudio(output);
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        int len = Math.Min(text.Length, _options.MaxTextLength);
        var t = new Tensor<T>([len]);
        for (int i = 0; i < len; i++)
            t[i] = NumOps.FromDouble(text[i] / 128.0);
        return t;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(LayerHelper<T>.CreateDefaultVITSLayers(
                _options.HiddenDim, _options.InterChannels, _options.FilterChannels,
                _options.NumEncoderLayers, _options.NumFlowSteps,
                _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate));
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            var o = Predict(input);
            var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector());
            var gt = Tensor<T>.FromVector(g);
            for (int i = Layers.Count - 1; i >= 0; i--)
                gt = Layers[i].Backward(gt);
            _optimizer?.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers)
        {
            int c = l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "VITS-Native" : "VITS-ONNX",
            Description = "VITS: Conditional VAE with Adversarial Learning for End-to-End TTS (Kim et al., 2021)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.HiddenDim
        };
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.SampleRate);
        writer.Write(_options.MelChannels);
        writer.Write(_options.HopSize);
        writer.Write(_options.HiddenDim);
        writer.Write(_options.NumFlowSteps);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.FilterChannels);
        writer.Write(_options.InterChannels);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumHeads);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.SampleRate = reader.ReadInt32();
        _options.MelChannels = reader.ReadInt32();
        _options.HopSize = reader.ReadInt32();
        _options.HiddenDim = reader.ReadInt32();
        _options.NumFlowSteps = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.FilterChannels = reader.ReadInt32();
        _options.InterChannels = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumEncoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new VITS<T>(Architecture, mp, _options);
        return new VITS<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(VITS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
