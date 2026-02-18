using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// Zipformer speech recognition model (Yao et al., 2023, Next-gen Kaldi).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Zipformer (Yao et al., 2023) is a more efficient variant of the Conformer with a U-Net-like
/// structure that processes speech at different temporal resolutions. It uses BiasNorm instead
/// of LayerNorm, SwooshR/SwooshL activations, and temporal downsampling across encoder stacks.
/// This achieves better accuracy with fewer parameters than standard Conformer, with WER 2.0%/4.4%
/// on LibriSpeech test-clean/other without an external language model.
/// </para>
/// <para>
/// <b>For Beginners:</b> Zipformer is an improved version of the Conformer that's both faster
/// and more accurate. It processes speech at different "zoom levels":
///
/// - Some layers look at fine details (individual sounds at full resolution)
/// - Middle layers zoom out (every 2nd, 4th, or 8th frame) for bigger-picture patterns
/// - Then it zooms back in to combine everything
///
/// This U-Net-like structure (like looking through a microscope at different magnifications)
/// makes it one of the most efficient speech encoders available today.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 5000);
/// var model = new Zipformer&lt;float&gt;(arch, "zipformer_medium.onnx");
/// var result = model.Transcribe(audioWaveform);
/// Console.WriteLine(result.Text); // "hello world"
/// </code>
/// </para>
/// </remarks>
public class Zipformer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    #region Fields

    private readonly ZipformerOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
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

    /// <summary>Creates a Zipformer model in ONNX inference mode.</summary>
    public Zipformer(NeuralNetworkArchitecture<T> architecture, string modelPath, ZipformerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ZipformerOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    /// <summary>Creates a Zipformer model in native training mode.</summary>
    public Zipformer(NeuralNetworkArchitecture<T> architecture, ZipformerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ZipformerOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    internal static async Task<Zipformer<T>> CreateAsync(ZipformerOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new ZipformerOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("zipformer", $"zipformer_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.VocabSize);
        return new Zipformer<T>(arch, mp, options);
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
        => new ZipformerStreamingSession(this, language ?? _options.Language);

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultZipformerLayers(
            encoderDims: _options.EncoderDims, numLayersPerStack: _options.NumLayersPerStack,
            numHeadsPerStack: _options.NumHeadsPerStack, numMels: _options.NumMels,
            vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate));
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
        int totalLayers = 0; foreach (int n in _options.NumLayersPerStack) totalLayers += n;
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Zipformer-Native" : "Zipformer-ONNX",
            Description = $"Zipformer {_options.Variant} U-Net ASR (Yao et al., 2023, Next-gen Kaldi)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = totalLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["NumStacks"] = _options.EncoderDims.Length.ToString();
        m.AdditionalInfo["MaxEncoderDim"] = (_options.EncoderDims.Length > 0 ? _options.EncoderDims.Max() : 0).ToString();
        m.AdditionalInfo["VocabSize"] = _options.VocabSize.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.NumMels); w.Write(_options.VocabSize);
        w.Write(_options.EncoderDims.Length);
        foreach (int d in _options.EncoderDims) w.Write(d);
        w.Write(_options.NumLayersPerStack.Length);
        foreach (int n in _options.NumLayersPerStack) w.Write(n);
        w.Write(_options.NumHeadsPerStack.Length);
        foreach (int h in _options.NumHeadsPerStack) w.Write(h);
        w.Write(_options.DownsampleFactors.Length);
        foreach (int f in _options.DownsampleFactors) w.Write(f);
        w.Write(_options.DropoutRate); w.Write(_options.Language);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.NumMels = r.ReadInt32(); _options.VocabSize = r.ReadInt32();
        int dimsLen = r.ReadInt32(); _options.EncoderDims = new int[dimsLen]; for (int i = 0; i < dimsLen; i++) _options.EncoderDims[i] = r.ReadInt32();
        int layersLen = r.ReadInt32(); _options.NumLayersPerStack = new int[layersLen]; for (int i = 0; i < layersLen; i++) _options.NumLayersPerStack[i] = r.ReadInt32();
        int headsLen = r.ReadInt32(); _options.NumHeadsPerStack = new int[headsLen]; for (int i = 0; i < headsLen; i++) _options.NumHeadsPerStack[i] = r.ReadInt32();
        int dsLen = r.ReadInt32(); _options.DownsampleFactors = new int[dsLen]; for (int i = 0; i < dsLen; i++) _options.DownsampleFactors[i] = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble(); _options.Language = r.ReadString();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new Zipformer<T>(Architecture, _options);

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

    private sealed class ZipformerStreamingSession : IStreamingTranscriptionSession<T>
    {
        private readonly Zipformer<T> _model;
        private readonly string _language;
        private readonly List<Tensor<T>> _chunks = new();
        private bool _disposed;

        public ZipformerStreamingSession(Zipformer<T> model, string language) { _model = model; _language = language; }

        public void FeedAudio(Tensor<T> audioChunk) { if (_disposed) throw new ObjectDisposedException(nameof(ZipformerStreamingSession)); _chunks.Add(audioChunk); }

        public TranscriptionResult<T> GetPartialResult()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ZipformerStreamingSession));
            if (_chunks.Count == 0) return new TranscriptionResult<T> { Language = _language };
            int totalLen = 0; foreach (var ch in _chunks) totalLen += ch.Length;
            var combined = new Tensor<T>(new[] { totalLen });
            int offset = 0;
            foreach (var ch in _chunks) { for (int i = 0; i < ch.Length; i++) combined[offset + i] = ch[i]; offset += ch.Length; }
            return _model.Transcribe(combined, _language);
        }

        public TranscriptionResult<T> Finalize()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(ZipformerStreamingSession));
            var result = GetPartialResult();
            _disposed = true;
            return result;
        }

        public void Dispose() { _disposed = true; }
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Zipformer<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
