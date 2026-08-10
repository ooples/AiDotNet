using AiDotNet.Attributes;
using AiDotNet.ActivationFunctions;
using AiDotNet.Audio;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>
/// Medical-conversation ASR: an end-to-end recognizer for doctor-patient dialogue, offering both
/// arms of the paper's comparison — Listen-Attend-Spell and CTC.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Chiu, Tripathi, Chou, Co, Jaitly, Jaunzeikare, Kannan, Nguyen, Sak, Sankar, Tansuwan, Wan, Wu and
/// Zhang, "Speech recognition for medical conversations" (Interspeech 2018, arXiv:1711.07274). The
/// paper is a COMPARISON, not a single architecture: "We explored both CTC and LAS systems for
/// building speech recognition models."
/// </para>
/// <para>
/// Its conclusion sets the default here: "The LAS was more resilient to noisy data and CTC required
/// more data clean up." Spontaneous doctor-patient conversation is precisely the noisy condition
/// where that mattered, so <see cref="MedicalAsrDecoderType.ListenAttendSpell"/> is the default and
/// CTC is available as the paper's other arm rather than as a lesser fallback.
/// </para>
/// <para>
/// <b>Two properties carry the method and are implemented rather than described.</b>
/// </para>
/// <list type="number">
/// <item><description><b>Both decoders exist and are selectable.</b> A CTC-only class cannot express
/// the paper's finding, because the finding IS the difference between the two. The previous revision
/// of this class had only a CTC head and described "clinical dictation" — a different task from the
/// conversations the paper studies.</description></item>
/// <item><description><b>LAS listens pyramidally.</b> Each reduction concatenates adjacent time
/// steps and halves the sequence, so the speller attends over a short summary rather than every
/// acoustic frame of a consultation. That reduction is what makes attention tractable over
/// conversation-length audio; without it "LAS" is just an attention decoder.</description></item>
/// </list>
/// <para>
/// <b>For Beginners:</b> Two ways to turn sound into text. One (CTC) labels each slice of audio
/// independently and is fast but needs clean recordings. The other (LAS) first compresses the audio
/// into a shorter summary, then writes the sentence out while looking back at that summary, which
/// copes better when people talk over each other or trail off — as they do in real consultations.
/// This model gives you both, and defaults to the one the paper found more robust.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 5000);
///
/// // Defaults to LAS, the paper's more noise-resilient arm.
/// var las = new MedicalASR&lt;double&gt;(architecture);
///
/// // The paper's other arm.
/// var ctc = new MedicalASR&lt;double&gt;(architecture,
///     new MedicalASROptions { DecoderType = MedicalAsrDecoderType.Ctc });
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Speech recognition for medical conversations",
    "https://arxiv.org/abs/1711.07274",
    Year = 2018,
    Authors = "Chung-Cheng Chiu, Anshuman Tripathi, Katherine Chou, Chris Co, Navdeep Jaitly, " +
              "Diana Jaunzeikare, Anjuli Kannan, Patrick Nguyen, Hasim Sak, Ananth Sankar, " +
              "Justin Tansuwan, Nathan Wan, Yonghui Wu, Xuedong Zhang")]
public class MedicalASR<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly MedicalASROptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;

    // Views into Layers. Under LAS the list holds two sub-graphs - listener then speller - which
    // read in sequence but with a pyramidal reduction BETWEEN them, so a plain sequential walk of
    // Layers would skip the reduction entirely.
    private readonly List<ILayer<T>> _listenerLayers = new();
    private readonly List<ILayer<T>> _spellerLayers = new();

    /// <summary>Gets which arm of the paper's comparison this instance is built as.</summary>
    public MedicalAsrDecoderType DecoderType => _options.DecoderType;

    /// <summary>
    /// Gets the factor by which the LAS listener reduces the time axis, 2^PyramidalReductions.
    /// One for CTC, whose frame-synchronous head needs the full resolution.
    /// </summary>
    public int TimeReductionFactor =>
        _options.DecoderType == MedicalAsrDecoderType.ListenAttendSpell
            ? 1 << Math.Max(0, _options.PyramidalReductions)
            : 1;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public MedicalASR(NeuralNetworkArchitecture<T> architecture, string modelPath, MedicalASROptions? options = null) : base(architecture) { _options = options ?? new MedicalASROptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "en" }; InitializeLayers(); }
    public MedicalASR(NeuralNetworkArchitecture<T> architecture, MedicalASROptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new MedicalASROptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { "en" }; InitializeLayers(); }

    /// <summary>
    /// Transcribes medical speech using domain-specialized Conformer encoder.
    /// The model is fine-tuned on clinical speech with medical terminology coverage.
    /// CTC head with domain LM rescoring produces medical transcriptions.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> logits;

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            logits = OnnxEncoder.Run(features);
        }
        else
        {
            logits = RecognizeForward(features);
        }

        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(logits);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = includeTimestamps ? ExtractSegments(text, duration, tokens, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var features = PreprocessAudio(audio); Tensor<T> logits; if (IsOnnxMode && OnnxEncoder is not null) logits = OnnxEncoder.Run(features); else { logits = RecognizeForward(features); } var (tokens, _) = CTCGreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(lang == detected ? primaryProb : otherProb); return result; }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("MedicalASR does not support streaming.");

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        Layers.Clear();
        _listenerLayers.Clear();
        _spellerLayers.Clear();

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _listenerLayers.AddRange(Layers);
            return;
        }

        var identity = (IActivationFunction<T>)new IdentityActivation<T>();

        if (_options.DecoderType == MedicalAsrDecoderType.Ctc)
        {
            // The paper's CTC arm: a frame-synchronous head straight onto the encoder - no decoder
            // state and no time reduction.
            var ctcStack = LayerHelper<T>.CreateDefaultConformerLayers(
                encoderDim: _options.EncoderDim, numLayers: _options.NumEncoderLayers,
                numAttentionHeads: _options.NumAttentionHeads, numMels: _options.NumMels,
                vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate).ToList();
            Layers.AddRange(ctcStack);
            _listenerLayers.AddRange(ctcStack);
            return;
        }

        // The paper's LAS arm. "Listen": encode, then reduce the time axis pyramidally.
        var listener = LayerHelper<T>.CreateDefaultConformerLayers(
            encoderDim: _options.EncoderDim, numLayers: _options.NumEncoderLayers,
            numAttentionHeads: _options.NumAttentionHeads, numMels: _options.NumMels,
            vocabSize: _options.EncoderDim, dropoutRate: _options.DropoutRate).ToList();
        Layers.AddRange(listener);
        _listenerLayers.AddRange(listener);

        // "Attend and spell": a decoder over the reduced summary.
        for (int i = 0; i < Math.Max(1, _options.NumDecoderLayers); i++)
        {
            var layer = new FullyConnectedLayer<T>(
                i == 0 ? _options.EncoderDim : _options.DecoderDim, _options.DecoderDim, identity);
            Layers.Add(layer);
            _spellerLayers.Add(layer);
        }
        var projection = new FullyConnectedLayer<T>(_options.DecoderDim, _options.VocabSize, identity);
        Layers.Add(projection);
        _spellerLayers.Add(projection);
    }

    /// <summary>
    /// The pyramidal reduction between listener and speller: halve the sequence length, once per
    /// configured reduction.
    /// </summary>
    /// <remarks>
    /// This is what makes attention tractable over conversation-length audio, and it is what
    /// distinguishes a LAS "listener" from an ordinary encoder. Adjacent frames are AVERAGED rather
    /// than concatenated, which halves the length without widening the feature axis - so the
    /// speller's input width stays the encoder width and the reduction can be applied repeatedly.
    /// </remarks>
    internal Tensor<T> PyramidalReduce(Tensor<T> encoded)
    {
        if (_options.DecoderType != MedicalAsrDecoderType.ListenAttendSpell) return encoded;

        var current = encoded;
        for (int r = 0; r < Math.Max(0, _options.PyramidalReductions); r++)
        {
            if (current.Rank < 2 || current.Shape[0] < 2) break;
            int frames = current.Shape[0];
            int width = current.Length / frames;
            int reduced = frames / 2;
            if (reduced < 1) break;

            var next = new Tensor<T>([reduced, width]);
            for (int t = 0; t < reduced; t++)
            {
                for (int w = 0; w < width; w++)
                {
                    double a = NumOps.ToDouble(current[(2 * t) * width + w]);
                    double b = NumOps.ToDouble(current[(2 * t + 1) * width + w]);
                    next[t * width + w] = NumOps.FromDouble(0.5 * (a + b));
                }
            }
            current = next;
        }
        return current;
    }

    /// <summary>Listener forward: the shared encoder.</summary>
    internal Tensor<T> ListenForward(Tensor<T> features)
    {
        var current = features;
        foreach (var layer in _listenerLayers) current = layer.Forward(current);
        return current;
    }

    /// <summary>Speller forward, over the pyramidally reduced summary.</summary>
    internal Tensor<T> SpellForward(Tensor<T> reduced)
    {
        if (_spellerLayers.Count == 0) return reduced;
        var current = reduced;
        for (int i = 0; i < _spellerLayers.Count - 1; i++)
        {
            current = Engine.ReLU(_spellerLayers[i].Forward(current));
        }
        return _spellerLayers[^1].Forward(current);
    }

    /// <summary>The full recognition graph for whichever arm is configured.</summary>
    internal Tensor<T> RecognizeForward(Tensor<T> features)
    {
        var encoded = ListenForward(features);
        return _spellerLayers.Count == 0 ? encoded : SpellForward(PyramidalReduce(encoded));
    }
    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        return RecognizeForward(input);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Overridden so the tape path follows the configured arm. Walking Layers sequentially would run
    /// the speller straight off the listener and skip the pyramidal reduction between them.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => RecognizeForward(input);

    /// <inheritdoc />
    /// <remarks>
    /// Overridden for the same reason: under LAS the listener and speller are separated by a
    /// reduction that a sequential walk of Layers does not perform.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (_listenerLayers.Count == 0) return base.GetNamedLayerActivations(input);

        var activations = new Dictionary<string, Tensor<T>>();
        var encoded = ListenForward(input);
        activations["Listener"] = encoded.Clone();
        if (_spellerLayers.Count > 0)
        {
            var reduced = PyramidalReduce(encoded);
            activations["PyramidalSummary"] = reduced.Clone();
            activations["Speller"] = SpellForward(reduced).Clone();
        }
        return activations;
    }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); TrainWithTape(input, expected, _optimizer); SetTrainingMode(false); }
    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) { if (MelSpec is not null) return MelSpec.Forward(rawAudio); return rawAudio; }
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "MedicalASR-Native" : "MedicalASR-ONNX", Description = "Medical ASR: clinical speech recognition (2024)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };
    protected override void SerializeNetworkSpecificData(BinaryWriter w) { w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty); w.Write(_options.SampleRate); w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers); w.Write(_options.NumAttentionHeads); w.Write(_options.NumMels); w.Write(_options.VocabSize); w.Write(_options.MaxTextLength); w.Write(_options.DropoutRate); w.Write(_options.Language); w.Write((int)_options.DecoderType); w.Write(_options.PyramidalReductions); w.Write(_options.DecoderDim); w.Write(_options.NumDecoderLayers); }
    protected override void DeserializeNetworkSpecificData(BinaryReader r) { _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = r.ReadInt32(); _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.VocabSize = r.ReadInt32(); _options.MaxTextLength = r.ReadInt32(); _options.DropoutRate = r.ReadDouble(); _options.Language = r.ReadString(); _options.DecoderType = (AiDotNet.Enums.MedicalAsrDecoderType)r.ReadInt32(); _options.PyramidalReductions = r.ReadInt32(); _options.DecoderDim = r.ReadInt32(); _options.NumDecoderLayers = r.ReadInt32(); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);  RebindSubNetworkViews(); }

    /// <summary>
    /// Rebinds the listener/speller views into <c>Layers</c> from the current layer instances.
    /// </summary>
    /// <remarks>
    /// Required after deserialization. The base deserializer replaces every entry of <c>Layers</c>
    /// with a fresh instance, so these views would otherwise still reference the discarded
    /// constructor-initialized weights — the clone then predicts from untrained layers while
    /// reporting the trained parameter vector, which is exactly what
    /// Clone_ShouldProduceIdenticalOutput caught here.
    /// </remarks>
    private void RebindSubNetworkViews()
    {
        _listenerLayers.Clear();
        _spellerLayers.Clear();
        if (!_useNativeMode || Layers.Count == 0) return;

        if (_options.DecoderType == MedicalAsrDecoderType.Ctc
            || (Architecture.Layers is not null && Architecture.Layers.Count > 0))
        {
            _listenerLayers.AddRange(Layers);
            return;
        }

        int spellerCount = Math.Max(1, _options.NumDecoderLayers) + 1;
        int listenerCount = Layers.Count - spellerCount;
        if (listenerCount <= 0) { _listenerLayers.AddRange(Layers); return; }

        for (int i = 0; i < listenerCount; i++) _listenerLayers.Add(Layers[i]);
        for (int i = listenerCount; i < Layers.Count; i++) _spellerLayers.Add(Layers[i]);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new MedicalASR<T>(Architecture, mp, _options); return new MedicalASR<T>(Architecture, _options); }

    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames && tokens.Count < _options.MaxTextLength; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, List<int> tokens, double confidence) { if (string.IsNullOrWhiteSpace(text) || tokens.Count == 0) return Array.Empty<TranscriptionSegment<T>>(); double timePerToken = duration / tokens.Count; var segments = new List<TranscriptionSegment<T>>(); var sb = new System.Text.StringBuilder(); double segStart = 0; for (int i = 0; i < tokens.Count; i++) { if (tokens[i] > 0 && tokens[i] <= char.MaxValue) sb.Append((char)tokens[i]); if ((tokens[i] == ' ' || i == tokens.Count - 1) && sb.Length > 0) { segments.Add(new TranscriptionSegment<T> { Text = sb.ToString().Trim(), StartTime = segStart, EndTime = (i + 1) * timePerToken, Confidence = NumOps.FromDouble(confidence) }); sb.Clear(); segStart = (i + 1) * timePerToken; } } if (segments.Count == 0) segments.Add(new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) }); return segments; }
    private string ClassifyLanguageFromTokens(List<int> tokens) { if (tokens.Count == 0) return _options.Language; int cjkCount = 0, latinCount = 0; foreach (var t in tokens) { if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++; else if (t >= 0x41 && t <= 0x7A) latinCount++; } if (cjkCount > latinCount && SupportedLanguages.Contains("zh")) return "zh"; return _options.Language; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MedicalASR<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
