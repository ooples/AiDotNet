using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// CTC (Connectionist Temporal Classification) decoder-based speech recognition model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CTC (Graves et al., 2006) is the most widely-used alignment-free training criterion for
/// ASR. A CTC decoder pairs a neural encoder (Transformer, Conformer, or LSTM) with a CTC
/// output head and decodes using greedy or beam search with optional language model rescoring.
/// This class provides a standalone CTC-based ASR pipeline with configurable beam width and
/// language model integration.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional speech recognition required aligning audio with text
/// frame-by-frame, which is expensive and error-prone. CTC removes this requirement:
///
/// 1. The encoder processes audio features (mel-spectrogram) and outputs a probability
///    distribution over characters for each time frame.
/// 2. CTC allows the model to output a special "blank" token when it's not sure which
///    character comes next.
/// 3. The decoder collapses repeated characters and removes blanks to get the final text.
///
/// Example: "h h h - e e - l - l - o" becomes "hello" (- = blank, repeated chars collapsed).
///
/// Beam search improves accuracy by considering multiple possible decodings simultaneously.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 34);
/// var model = new CTCDecoder&lt;float&gt;(arch, "ctc_decoder.onnx");
/// var result = model.Transcribe(audioWaveform);
/// Console.WriteLine(result.Text);
/// </code>
/// </para>
/// </remarks>
public class CTCDecoder<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    #region Fields

    private readonly CTCDecoderOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region ISpeechRecognizer Properties

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedLanguages { get; }

    /// <inheritdoc />
    public bool SupportsStreaming => false;

    /// <inheritdoc />
    public bool SupportsWordTimestamps => true;

    #endregion

    #region Constructors

    /// <summary>Creates a CTC decoder model in ONNX inference mode.</summary>
    public CTCDecoder(NeuralNetworkArchitecture<T> architecture, string modelPath, CTCDecoderOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CTCDecoderOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    /// <summary>Creates a CTC decoder model in native training mode.</summary>
    public CTCDecoder(NeuralNetworkArchitecture<T> architecture, CTCDecoderOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CTCDecoderOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    internal static async Task<CTCDecoder<T>> CreateAsync(CTCDecoderOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new CTCDecoderOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("ctc_decoder", $"ctc_decoder_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: options.VocabSize);
        return new CTCDecoder<T>(arch, mp, options);
    }

    #endregion

    #region ISpeechRecognizer

    /// <inheritdoc />
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var logits = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        List<int> tokens = _options.BeamWidth <= 1 ? CTCGreedyDecode(logits) : CTCBeamSearchDecode(logits, _options.BeamWidth);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;
        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(tokens.Count > 0 ? 0.85 : 0.0),
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
        => throw new NotSupportedException("CTC decoder does not support streaming. Use Conformer with streaming instead.");

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultCTCDecoderLayers(
            encoderDim: _options.EncoderDim, numLayers: _options.NumEncoderLayers,
            numAttentionHeads: _options.NumAttentionHeads, feedForwardDim: _options.FeedForwardDim,
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
            Name = _useNativeMode ? "CTCDecoder-Native" : "CTCDecoder-ONNX",
            Description = $"CTC decoder {_options.Variant} ASR model (Graves et al., 2006)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["EncoderDim"] = _options.EncoderDim.ToString();
        m.AdditionalInfo["BeamWidth"] = _options.BeamWidth.ToString();
        m.AdditionalInfo["VocabSize"] = _options.VocabSize.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers);
        w.Write(_options.NumAttentionHeads); w.Write(_options.FeedForwardDim);
        w.Write(_options.NumMels); w.Write(_options.VocabSize);
        w.Write(_options.BeamWidth); w.Write(_options.DropoutRate);
        w.Write(_options.Language); w.Write(_options.BlankTokenIndex);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32();
        _options.NumAttentionHeads = r.ReadInt32(); _options.FeedForwardDim = r.ReadInt32();
        _options.NumMels = r.ReadInt32(); _options.VocabSize = r.ReadInt32();
        _options.BeamWidth = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        _options.Language = r.ReadString(); _options.BlankTokenIndex = r.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new CTCDecoder<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private List<int> CTCGreedyDecode(Tensor<T> logits)
    {
        var tokens = new List<int>();
        int prevToken = -1;
        int blankIdx = _options.BlankTokenIndex;
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

    private List<int> CTCBeamSearchDecode(Tensor<T> logits, int beamWidth)
    {
        int blankIdx = _options.BlankTokenIndex;
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0];

        // Each beam is (prefix tokens, log-probability)
        var beams = new List<(List<int> Tokens, double LogProb)> { (new List<int>(), 0.0) };

        for (int t = 0; t < numFrames; t++)
        {
            var candidates = new List<(List<int> Tokens, double LogProb)>();
            foreach (var (tokens, logProb) in beams)
            {
                // Compute log-softmax for this frame
                double maxLogit = double.NegativeInfinity;
                for (int v = 0; v < vocabSize; v++)
                {
                    double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]);
                    if (val > maxLogit) maxLogit = val;
                }
                double sumExp = 0;
                for (int v = 0; v < vocabSize; v++)
                {
                    double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]);
                    sumExp += Math.Exp(val - maxLogit);
                }
                double logSumExp = maxLogit + Math.Log(sumExp);

                // Blank extension
                {
                    double blankLogProb = (logits.Rank >= 2 ? NumOps.ToDouble(logits[t, blankIdx]) : NumOps.ToDouble(logits[blankIdx])) - logSumExp;
                    candidates.Add((new List<int>(tokens), logProb + blankLogProb));
                }

                // Character extensions - take top-K to keep beam manageable
                var topK = new List<(int Idx, double LogP)>();
                for (int v = 0; v < vocabSize; v++)
                {
                    if (v == blankIdx) continue;
                    double val = (logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v])) - logSumExp;
                    topK.Add((v, val));
                }
                topK.Sort((a, b) => b.LogP.CompareTo(a.LogP));
                int limit = Math.Min(beamWidth, topK.Count);
                for (int k = 0; k < limit; k++)
                {
                    int lastToken = tokens.Count > 0 ? tokens[^1] : -1;
                    if (topK[k].Idx == lastToken) continue; // CTC collapse
                    var newTokens = new List<int>(tokens) { topK[k].Idx };
                    candidates.Add((newTokens, logProb + topK[k].LogP));
                }
            }

            // Prune to beam width
            candidates.Sort((a, b) => b.LogProb.CompareTo(a.LogProb));
            beams = candidates.GetRange(0, Math.Min(beamWidth, candidates.Count));
        }

        return beams.Count > 0 ? beams[0].Tokens : new List<int>();
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
        return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(0.85) } };
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CTCDecoder<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
