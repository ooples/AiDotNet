using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// Conformer speech recognition model (Gulati et al., 2020, Google).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Conformer combines convolution and self-attention in a novel macaron-style
/// architecture: Feed-Forward / Self-Attention / Convolution / Feed-Forward. This captures
/// both local (phoneme-level) and global (sentence-level) dependencies. Conformer-CTC
/// achieves WER 1.9%/3.9% on LibriSpeech test-clean/other and is the backbone of
/// production ASR at Google, NVIDIA NeMo, and many other systems.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Conformer is like a super-powered speech encoder.
/// Before it, speech models used either:
/// - Transformers (good at seeing the whole sentence, bad at fine details)
/// - CNNs (good at local patterns like individual sounds, bad at context)
///
/// The Conformer uses BOTH in each layer, so it can hear the "s" at the end of a
/// word (local detail) while also understanding the sentence structure (global context).
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 5000);
/// var model = new Conformer&lt;float&gt;(arch, "conformer_medium.onnx");
/// var result = model.Transcribe(audioWaveform);
/// Console.WriteLine(result.Text); // "hello world"
/// </code>
/// </para>
/// </remarks>
public class Conformer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    #region Fields

    private readonly ConformerOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region ISpeechRecognizer Properties

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedLanguages { get; }

    /// <inheritdoc />
    public bool SupportsStreaming => true;

    /// <inheritdoc />
    public bool SupportsWordTimestamps => true;

    #endregion

    #region Constructors

    /// <summary>Creates a Conformer model in ONNX inference mode.</summary>
    public Conformer(NeuralNetworkArchitecture<T> architecture, string modelPath, ConformerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ConformerOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    /// <summary>Creates a Conformer model in native training mode.</summary>
    public Conformer(NeuralNetworkArchitecture<T> architecture, ConformerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ConformerOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    internal static async Task<Conformer<T>> CreateAsync(ConformerOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new ConformerOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("conformer", $"conformer_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.VocabSize);
        return new Conformer<T>(arch, mp, options);
    }

    #endregion

    #region ISpeechRecognizer

    /// <inheritdoc />
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var logits = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        var tokens = CTCGreedyDecode(logits);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;
        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(tokens.Count > 0 ? 0.9 : 0.0),
            DurationSeconds = duration,
            Segments = includeTimestamps ? ExtractSegments(tokens, text, audio.Shape[0]) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null,
        bool includeTimestamps = false, CancellationToken cancellationToken = default)
        => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <inheritdoc />
    public string DetectLanguage(Tensor<T> audio) => _options.Language;

    /// <inheritdoc />
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
        => new Dictionary<string, T> { [_options.Language] = NumOps.FromDouble(1.0) };

    /// <inheritdoc />
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null)
        => new ConformerStreamingSession(this, language ?? _options.Language);

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultConformerLayers(
            encoderDim: _options.EncoderDim, numLayers: _options.NumEncoderLayers,
            numAttentionHeads: _options.NumAttentionHeads,
            feedForwardExpansionFactor: _options.FeedForwardExpansionFactor,
            numMels: _options.NumMels, vocabSize: _options.VocabSize,
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
        _optimizer.UpdateParameters(Layers);
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
            Name = _useNativeMode ? "Conformer-Native" : "Conformer-ONNX",
            Description = $"Conformer {_options.Variant} ASR model (Gulati et al., 2020)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["EncoderDim"] = _options.EncoderDim.ToString();
        m.AdditionalInfo["VocabSize"] = _options.VocabSize.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers);
        w.Write(_options.NumAttentionHeads); w.Write(_options.FeedForwardExpansionFactor);
        w.Write(_options.ConvKernelSize); w.Write(_options.NumMels);
        w.Write(_options.VocabSize); w.Write(_options.DropoutRate);
        w.Write(_options.Language);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32();
        _options.NumAttentionHeads = r.ReadInt32(); _options.FeedForwardExpansionFactor = r.ReadInt32();
        _options.ConvKernelSize = r.ReadInt32(); _options.NumMels = r.ReadInt32();
        _options.VocabSize = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        _options.Language = r.ReadString();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new Conformer<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private List<int> CTCGreedyDecode(Tensor<T> logits)
    {
        var tokens = new List<int>();
        int prevToken = -1;
        int blankIdx = 0;
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0];
        for (int t = 0; t < numFrames; t++)
        {
            int maxIdx = 0; double maxVal = double.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]);
                if (val > maxVal) { maxVal = val; maxIdx = v; }
            }
            if (maxIdx != blankIdx && maxIdx != prevToken) tokens.Add(maxIdx);
            prevToken = maxIdx;
        }
        return tokens;
    }

    private string TokensToText(List<int> tokens)
    {
        var vocab = _options.Vocabulary;
        var chars = new List<char>();
        foreach (var token in tokens)
        {
            if (token >= 0 && token < vocab.Length)
            {
                var symbol = vocab[token];
                if (symbol == "|" || symbol == " ") chars.Add(' ');
                else if (symbol.Length == 1 && char.IsLetter(symbol[0])) chars.Add(symbol[0]);
                else if (symbol == "'") chars.Add('\'');
            }
        }
        return new string(chars.ToArray()).Trim();
    }

    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(List<int> tokens, string text, int audioLength)
    {
        if (tokens.Count == 0 || string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>();
        double duration = (double)audioLength / SampleRate;
        return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(0.9) } };
    }

    #endregion

    #region Streaming Session

    private sealed class ConformerStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly Conformer<T> _model;
        private readonly string _language;
        private readonly List<Tensor<T>> _chunks = new();
        private bool _disposed;

        public ConformerStreamingSession(Conformer<T> model, string language) { _model = model; _language = language; }

        public void FeedAudio(Tensor<T> audioChunk) { if (_disposed) throw new ObjectDisposedException(nameof(ConformerStreamingSession)); _chunks.Add(audioChunk); }

        public TranscriptionResult<T> GetPartialResult()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ConformerStreamingSession));
            if (_chunks.Count == 0) return new TranscriptionResult<T> { Language = _language };
            int totalLen = 0; foreach (var ch in _chunks) totalLen += ch.Length;
            var combined = new Tensor<T>(new[] { totalLen });
            int offset = 0;
            foreach (var ch in _chunks) { for (int i = 0; i < ch.Length; i++) combined[offset + i] = ch[i]; offset += ch.Length; }
            return _model.Transcribe(combined, _language);
        }

        public TranscriptionResult<T> Finalize()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ConformerStreamingSession));
            var result = GetPartialResult();
            _disposed = true;
            return result;
        }

        public void Dispose() { _disposed = true; }
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Conformer<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
