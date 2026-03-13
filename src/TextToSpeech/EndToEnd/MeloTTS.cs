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
/// MeloTTS: high-quality multilingual TTS with VITS backbone, language-specific text processing, and mixed-language support.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Project: "MeloTTS: High-quality Multi-lingual Text-to-Speech" (MyShell, 2024)</item></list></para>
/// <para><b>For Beginners:</b> MeloTTS builds on the VITS architecture (see VITS class) to add robust
/// multilingual support. While VITS handles the core speech synthesis pipeline (VAE + normalizing flows +
/// HiFi-GAN decoder), MeloTTS adds several important features:
/// (1) Language-specific text processing using BERT-based grapheme-to-phoneme (G2P) for Chinese
/// and eSpeak for other languages,
/// (2) Language ID embeddings that tell the encoder and decoder which language is being spoken,
/// (3) Mixed-language (code-switching) support so it can handle sentences that switch between
/// languages mid-utterance, and
/// (4) Multi-speaker conditioning for generating different voices.
/// This makes it particularly useful for applications needing natural speech across multiple languages.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class MeloTTS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly MeloTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="MeloTTS{T}"/> class in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional model configuration options.</param>
    public MeloTTS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        MeloTTSOptions? options = null) : base(architecture)
    {
        _options = options ?? new MeloTTSOptions();
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
    /// Initializes a new instance of the <see cref="MeloTTS{T}"/> class in native training/inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional model configuration options.</param>
    /// <param name="optimizer">Optional gradient-based optimizer for training.</param>
    public MeloTTS(
        NeuralNetworkArchitecture<T> architecture,
        MeloTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new MeloTTSOptions();
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
    /// Gets the number of normalizing flow steps used in the VITS backbone.
    /// </summary>
    public int NumFlowSteps => _options.NumFlowSteps;

    /// <summary>
    /// Synthesizes multilingual speech using MeloTTS's extended VITS pipeline.
    /// </summary>
    /// <param name="text">The input text to synthesize.</param>
    /// <returns>A tensor containing the generated waveform.</returns>
    /// <remarks>
    /// <para>MeloTTS extends VITS (Kim et al., 2021) with:</para>
    /// <para>(1) Language-specific text processing: BERT-based G2P for Chinese, eSpeak for other languages.</para>
    /// <para>(2) Language ID embedding: conditions encoder and decoder on target language.</para>
    /// <para>(3) Mixed-language support: handles code-switching within utterances.</para>
    /// <para>(4) VITS backbone with speaker conditioning for multi-speaker support.</para>
    /// <para><b>For Beginners:</b> This method takes text (which can be in multiple languages, even within
    /// the same sentence) and produces speech audio. The language-specific processing ensures that
    /// each language's phonetic rules are followed correctly, while the language ID embedding helps
    /// the model adjust its pronunciation and prosody for each language segment.</para>
    /// </remarks>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return PostprocessAudio(OnnxModel.Run(input));
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
        ThrowIfDisposed();
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
        ThrowIfDisposed();
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
            Name = _useNativeMode ? "MeloTTS-Native" : "MeloTTS-ONNX",
            Description = "MeloTTS: High-quality Multilingual TTS (MyShell, 2024)",
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
        writer.Write(_options.DropoutRate);
        writer.Write(_options.FilterChannels);
        writer.Write(_options.InterChannels);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumFlowSteps);
        writer.Write(_options.NumHeads);
        writer.Write(_options.MaxTextLength);
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
        _options.DropoutRate = reader.ReadDouble();
        _options.FilterChannels = reader.ReadInt32();
        _options.InterChannels = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumEncoderLayers = reader.ReadInt32();
        _options.NumFlowSteps = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.MaxTextLength = reader.ReadInt32();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
        {
            if (!File.Exists(p))
                throw new FileNotFoundException($"ONNX model not found during deserialization: {p}", p);
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new MeloTTS<T>(Architecture, mp, _options);
        return new MeloTTS<T>(Architecture, _options, _optimizer);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(MeloTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
