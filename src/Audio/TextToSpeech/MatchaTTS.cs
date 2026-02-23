using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Matcha-TTS fast text-to-speech model using conditional flow matching (Mehta et al., 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Matcha-TTS uses Optimal Transport Conditional Flow Matching (OT-CFM) to generate
/// mel-spectrograms from text in just 2-4 synthesis steps, achieving 10x speedup over
/// Grad-TTS with comparable quality (MOS 4.04 on LJSpeech). The model combines a
/// text encoder with a U-Net-based flow matching decoder and a duration predictor.
/// </para>
/// <para>
/// <b>For Beginners:</b> Matcha-TTS is a fast, lightweight text-to-speech model.
/// While other models need many steps to gradually refine audio (like slowly developing
/// a photograph), Matcha-TTS takes a shortcut - it finds the most direct path from
/// random noise to a mel-spectrogram in just a few steps.
///
/// Pipeline:
/// 1. Text encoder: Analyzes the input text
/// 2. Duration predictor: Decides how long each sound should be
/// 3. Flow matching decoder: Generates the mel-spectrogram in 2-4 steps
/// 4. Vocoder (separate): Converts mel-spectrogram to audio waveform
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 1);
/// var model = new MatchaTTS&lt;float&gt;(arch, "matcha_tts_base.onnx");
/// var audio = model.Synthesize("Hello, how are you today?");
/// </code>
/// </para>
/// </remarks>
public class MatchaTTS<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
{
    #region Fields

    private readonly MatchaTTSOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region ITextToSpeech Properties

    /// <inheritdoc />
    public IReadOnlyList<VoiceInfo<T>> AvailableVoices { get; } = new[]
    {
        new VoiceInfo<T> { Id = "default", Name = "Default", Language = "en", Gender = VoiceGender.Neutral, Style = "neutral" }
    };

    /// <inheritdoc />
    public bool SupportsVoiceCloning => false;

    /// <inheritdoc />
    public bool SupportsEmotionControl => false;

    /// <inheritdoc />
    public bool SupportsStreaming => false;

    #endregion

    #region Constructors

    /// <summary>Creates a Matcha-TTS model in ONNX inference mode.</summary>
    public MatchaTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, MatchaTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MatchaTTSOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        InitializeLayers();
    }

    /// <summary>Creates a Matcha-TTS model in native training mode.</summary>
    public MatchaTTS(NeuralNetworkArchitecture<T> architecture, MatchaTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MatchaTTSOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<MatchaTTS<T>> CreateAsync(MatchaTTSOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new MatchaTTSOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("matcha_tts", $"matcha_tts_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: 1);
        return new MatchaTTS<T>(arch, mp, options);
    }

    #endregion

    #region ITextToSpeech

    /// <inheritdoc />
    public Tensor<T> Synthesize(string text, string? voiceId = null, double speakingRate = 1.0, double pitch = 0.0)
    {
        ThrowIfDisposed();
        var encoded = EncodeText(text);
        var withRate = ApplyProsody(encoded, speakingRate, pitch);
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(withRate) : Predict(withRate);
    }

    /// <inheritdoc />
    public Task<Tensor<T>> SynthesizeAsync(string text, string? voiceId = null, double speakingRate = 1.0,
        double pitch = 0.0, CancellationToken cancellationToken = default)
        => Task.Run(() => Synthesize(text, voiceId, speakingRate, pitch), cancellationToken);

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithVoiceCloning(string text, Tensor<T> referenceAudio, double speakingRate = 1.0, double pitch = 0.0)
        => throw new NotSupportedException("Matcha-TTS does not support voice cloning. Use StyleTTS 2 or VALL-E instead.");

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithEmotion(string text, string emotion, double emotionIntensity = 0.5,
        string? voiceId = null, double speakingRate = 1.0)
        => throw new NotSupportedException("Matcha-TTS does not support emotion control. Use StyleTTS 2 instead.");

    /// <inheritdoc />
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
        => throw new NotSupportedException("Matcha-TTS does not support speaker embedding extraction.");

    /// <inheritdoc />
    public IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0)
        => throw new NotSupportedException("Matcha-TTS does not support streaming synthesis.");

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultMatchaTTSLayers(
            textEncoderDim: _options.TextEncoderDim, numTextEncoderLayers: _options.NumTextEncoderLayers,
            numTextEncoderHeads: _options.NumTextEncoderHeads, decoderDim: _options.DecoderDim,
            numDecoderLayers: _options.NumDecoderLayers, numMels: _options.NumMels,
            dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MatchaTTS-Native" : "MatchaTTS-ONNX",
            Description = $"Matcha-TTS {_options.Variant} flow-matching TTS (Mehta et al., 2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumTextEncoderLayers + _options.NumDecoderLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["NumSynthesisSteps"] = _options.NumSynthesisSteps.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.TextEncoderDim); w.Write(_options.NumTextEncoderLayers);
        w.Write(_options.NumTextEncoderHeads); w.Write(_options.DecoderDim);
        w.Write(_options.NumDecoderLayers); w.Write(_options.NumSynthesisSteps);
        w.Write(_options.Temperature); w.Write(_options.NumMels);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.TextEncoderDim = r.ReadInt32(); _options.NumTextEncoderLayers = r.ReadInt32();
        _options.NumTextEncoderHeads = r.ReadInt32(); _options.DecoderDim = r.ReadInt32();
        _options.NumDecoderLayers = r.ReadInt32(); _options.NumSynthesisSteps = r.ReadInt32();
        _options.Temperature = r.ReadDouble(); _options.NumMels = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new MatchaTTS<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> EncodeText(string text)
    {
        var encoding = _tokenizer.Encode(text);
        int dim = _options.TextEncoderDim;
        var tokens = new Tensor<T>([dim]);
        int copyCount = Math.Min(encoding.TokenIds.Count, dim);
        for (int i = 0; i < copyCount; i++)
            tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    private Tensor<T> ApplyProsody(Tensor<T> encoded, double speakingRate, double pitch)
    {
        int prosodyDim = 4;
        var result = new Tensor<T>(new[] { encoded.Length + prosodyDim });
        for (int i = 0; i < encoded.Length; i++)
            result[i] = NumOps.FromDouble(NumOps.ToDouble(encoded[i]) * speakingRate);
        result[encoded.Length] = NumOps.FromDouble(speakingRate);
        result[encoded.Length + 1] = NumOps.FromDouble(pitch);
        result[encoded.Length + 2] = NumOps.FromDouble(_options.Temperature);
        result[encoded.Length + 3] = NumOps.FromDouble(_options.NumSynthesisSteps);
        return result;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MatchaTTS<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
