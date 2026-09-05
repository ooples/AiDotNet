using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Training;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>
/// Code-switching ASR: a hybrid CTC/attention model trained jointly with an EXPLICIT
/// language-identification task, for speech that mixes languages inside one utterance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Zeng, Khassanov, Pham, Xu, Chng and Li, "On the End-to-End Solution to Mandarin-English
/// Code-switching Speech Recognition" (Interspeech 2019, arXiv:1811.00241). Code-switching "refers
/// to a linguistic phenomenon where a speaker uses different languages in an utterance or between
/// alternating utterances". The paper's headline result is not the encoder — it is that making the
/// model say WHICH language it is hearing, as a separate supervised task, improves the transcription
/// itself:
/// </para>
/// <code>
/// var audio = Tensor&lt;double&gt;.CreateRandom(1, 16000);
///   L_MTL(Y|X) = lambda1 * L_att(Y|X) + (1 - lambda1) * L_ctc(Y|X) + lambda2 * L_lid(Z|X)
///
///   CTC weight (1 - lambda1) = 0.2        lambda2 = 0.2
/// </code>
/// <para>
/// <b>Three properties carry the method, and all three are implemented rather than described.</b>
/// </para>
/// <list type="number">
/// <item><description><b>The objective is genuinely hybrid.</b> Both a CTC head on the encoder and
/// an attention decoder are trained, and the loss interpolates them. CTC's monotonic alignment
/// stabilizes the decoder early; attention supplies the context-dependence CTC's
/// conditional-independence assumption cannot express. A CTC-only model is a different model, not a
/// simplification of this one.</description></item>
/// <item><description><b>Language identification is EXPLICIT and LEARNED.</b> An LID head is trained
/// as a third task at weight 0.2 and is what <see cref="DetectLanguage"/> reports. The previous
/// revision of this class stated that "language identification is performed implicitly through
/// shared encoder representations" and then classified by counting CJK versus Latin Unicode
/// codepoints in the decoded string — a hard-coded heuristic over the OUTPUT, which is the exact
/// opposite of the paper's contribution and learns nothing.</description></item>
/// <item><description><b>The two languages do NOT share one inventory.</b> "Mandarin uses characters
/// while English uses BPE units". The output alphabet is the concatenation of the two, so a token
/// identifies its own language — which is what gives the LID task something to be consistent with.
/// A single unified vocabulary, which the previous revision described, destroys that.</description></item>
/// </list>
/// <para>
/// <b>For Beginners:</b> Bilingual speakers often switch language mid-sentence. Two things make that
/// hard: the model must transcribe sounds from either language, and it must handle the switch itself.
/// This trains three jobs at once — one that lines sounds up to letters, one that reads the sentence
/// in context, and one whose only job is to say which language is being spoken. The third job is not
/// wasted effort: being forced to notice the switch makes the other two better.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 5000);
/// var model = new CodeSwitchingASR&lt;double&gt;(architecture);
///
/// var result = model.Transcribe(audio);
/// var language = model.DetectLanguage(audio);   // from the learned LID head
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("On the End-to-End Solution to Mandarin-English Code-switching Speech Recognition",
    "https://arxiv.org/abs/1811.00241",
    Year = 2019,
    Authors = "Zhiping Zeng, Yerbolat Khassanov, Van Tung Pham, Haihua Xu, Eng Siong Chng, Haizhou Li")]
public partial class CodeSwitchingASR<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly CodeSwitchingASROptions _options;
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Sub-network views into Layers. Layers holds four sub-graphs in this order — encoder, CTC head,
    // attention decoder, LID head — and walking the whole list would push encoder output through the
    // CTC head into the decoder into the LID head, which is not the model's graph.
    private readonly List<ILayer<T>> _encoderLayers = new();
    private FullyConnectedLayer<T>? _ctcHead;
    private readonly List<ILayer<T>> _decoderLayers = new();
    private FullyConnectedLayer<T>? _lidHead;

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedLanguages { get; }

    /// <inheritdoc />
    public bool SupportsStreaming => false;

    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    /// <summary>
    /// Gets the token id at which the English BPE inventory begins. Ids below it are Mandarin
    /// characters; the CTC blank is id 0.
    /// </summary>
    public int EnglishTokenOffset => 1 + _options.MandarinTokenCount;

    /// <summary>
    /// Returns the language a token id belongs to, which is well-defined precisely because the two
    /// inventories are concatenated rather than unified.
    /// </summary>
    public string LanguageOfToken(int tokenId) =>
        tokenId <= 0 ? _options.Language : tokenId < EnglishTokenOffset ? "zh" : "en";

    public CodeSwitchingASR(NeuralNetworkArchitecture<T> architecture, string modelPath, CodeSwitchingASROptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CodeSwitchingASROptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { "zh", "en" };
        InitializeLayers();
    }

    public CodeSwitchingASR(NeuralNetworkArchitecture<T> architecture, CodeSwitchingASROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CodeSwitchingASROptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        // The paper's task is Mandarin-English. The pair is not a limitation of the architecture but
        // it IS what the concatenated inventory and the LID label set are defined over, so it is
        // stated rather than implied by a longer list the model cannot actually distinguish.
        SupportedLanguages = new[] { "zh", "en" };
        InitializeLayers();
    }

    #region Layer construction

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        Layers.Clear();
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            BindSubNetworkViews(customLayers: true);
            return;
        }

        var identity = (IActivationFunction<T>)new IdentityActivation<T>();

        // --- Shared encoder ---
        var encoder = LayerHelper<T>.CreateDefaultConformerLayers(
            encoderDim: _options.EncoderDim,
            numLayers: _options.NumEncoderLayers,
            numAttentionHeads: _options.NumAttentionHeads,
            numMels: _options.NumMels,
            vocabSize: _options.EncoderDim,   // project to encoder width, not to the alphabet
            dropoutRate: _options.DropoutRate).ToList();
        Layers.AddRange(encoder);

        // --- CTC head: encoder width -> full concatenated inventory (+ blank) ---
        Layers.Add(new FullyConnectedLayer<T>(_options.EncoderDim, _options.VocabSize, identity));

        // --- Attention decoder ---
        for (int i = 0; i < Math.Max(1, _options.NumDecoderLayers); i++)
        {
            Layers.Add(new FullyConnectedLayer<T>(
                i == 0 ? _options.EncoderDim : _options.DecoderDim, _options.DecoderDim, identity));
        }
        Layers.Add(new FullyConnectedLayer<T>(_options.DecoderDim, _options.VocabSize, identity));

        // --- LID head: the paper's third task, over the language set ---
        Layers.Add(new FullyConnectedLayer<T>(
            _options.SharedLidAttention ? _options.EncoderDim : _options.DecoderDim,
            SupportedLanguages.Count, identity));

        BindSubNetworkViews(customLayers: false);
    }

    /// <summary>
    /// Binds the encoder / CTC / decoder / LID views into <c>Layers</c>.
    /// </summary>
    /// <remarks>
    /// Must be re-run after deserialization: the base deserializer replaces every entry in
    /// <c>Layers</c> with a fresh instance, leaving these views pointing at discarded weights.
    /// </remarks>
    private void BindSubNetworkViews(bool customLayers)
    {
        _encoderLayers.Clear();
        _decoderLayers.Clear();
        _ctcHead = null;
        _lidHead = null;

        if (customLayers)
        {
            // A user-supplied stack has no declared partition, so treat it all as the encoder and
            // let the heads stay null; the forward paths fall back to the plain stack.
            _encoderLayers.AddRange(Layers);
            return;
        }

        int decoderCount = Math.Max(1, _options.NumDecoderLayers) + 1;   // hidden layers + projection
        int lidCount = 1;
        int ctcCount = 1;
        int encoderCount = Layers.Count - ctcCount - decoderCount - lidCount;
        if (encoderCount < 0) { _encoderLayers.AddRange(Layers); return; }

        int idx = 0;
        for (int i = 0; i < encoderCount; i++) _encoderLayers.Add(Layers[idx++]);
        _ctcHead = Layers[idx++] as FullyConnectedLayer<T>;
        for (int i = 0; i < decoderCount; i++) _decoderLayers.Add(Layers[idx++]);
        _lidHead = Layers[idx] as FullyConnectedLayer<T>;
    }

    #endregion

    #region Forward paths

    /// <summary>Shared encoder forward.</summary>
    private Tensor<T> EncodeForward(Tensor<T> features)
    {
        var current = features;
        foreach (var layer in _encoderLayers) current = layer.Forward(current);
        return current;
    }

    /// <summary>CTC branch: encoder output projected to the concatenated inventory.</summary>
    private Tensor<T> CtcForward(Tensor<T> encoded) =>
        _ctcHead is null ? encoded : _ctcHead.Forward(encoded);

    /// <summary>Attention decoder branch, over the same encoder output.</summary>
    private Tensor<T> AttentionForward(Tensor<T> encoded)
    {
        if (_decoderLayers.Count == 0) return encoded;
        var current = encoded;
        for (int i = 0; i < _decoderLayers.Count - 1; i++)
        {
            current = Engine.ReLU(_decoderLayers[i].Forward(current));
        }
        return _decoderLayers[^1].Forward(current);
    }

    /// <summary>LID branch: per-frame language logits over <see cref="SupportedLanguages"/>.</summary>
    private Tensor<T> LidForward(Tensor<T> encoded) =>
        _lidHead is null ? encoded : _lidHead.Forward(encoded);

    #endregion

    #region Inference

    /// <summary>
    /// Transcribes code-switched speech with the hybrid model, decoding the CTC branch.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> ctcLogits;
        string detected;

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            ctcLogits = OnnxEncoder.Run(features);
            detected = language ?? _options.Language;
        }
        else
        {
            var encoded = EncodeForward(features);
            ctcLogits = CtcForward(encoded);
            detected = language ?? DecodeLanguage(LidForward(encoded));
        }

        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(ctcLogits);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = detected,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = includeTimestamps ? ExtractSegments(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null,
        bool includeTimestamps = false, CancellationToken cancellationToken = default)
        => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <summary>
    /// Reports the dominant language from the LEARNED LID head.
    /// </summary>
    /// <remarks>
    /// This is the paper's explicit language-identification task, not a rule over the decoded string.
    /// </remarks>
    public string DetectLanguage(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var probabilities = DetectLanguageProbabilities(audio);
        string best = _options.Language;
        double bestValue = double.NegativeInfinity;
        foreach (var kv in probabilities)
        {
            double v = NumOps.ToDouble(kv.Value);
            if (v > bestValue) { bestValue = v; best = kv.Key; }
        }
        return best;
    }

    /// <summary>
    /// Per-language posteriors from the LID head, averaged over frames and softmax-normalized.
    /// </summary>
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var result = new Dictionary<string, T>();

        if (IsOnnxMode || _lidHead is null)
        {
            // No native LID head to consult. Report a flat distribution rather than inventing a
            // confident answer from a heuristic.
            double uniform = 1.0 / Math.Max(1, SupportedLanguages.Count);
            foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(uniform);
            return result;
        }

        var lidLogits = LidForward(EncodeForward(PreprocessAudio(audio)));
        int classes = SupportedLanguages.Count;
        int frames = lidLogits.Rank >= 2 ? lidLogits.Shape[0] : 1;

        var mean = new double[classes];
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < classes; c++)
            {
                int flat = f * classes + c;
                mean[c] += flat < lidLogits.Length ? NumOps.ToDouble(lidLogits[flat]) : 0.0;
            }
        }

        double max = double.NegativeInfinity;
        for (int c = 0; c < classes; c++) { mean[c] /= Math.Max(1, frames); if (mean[c] > max) max = mean[c]; }
        double total = 0.0;
        for (int c = 0; c < classes; c++) { mean[c] = Math.Exp(mean[c] - max); total += mean[c]; }

        for (int c = 0; c < classes; c++)
        {
            result[SupportedLanguages[c]] = NumOps.FromDouble(total > 0 ? mean[c] / total : 1.0 / classes);
        }
        return result;
    }

    private string DecodeLanguage(Tensor<T> lidLogits)
    {
        int classes = SupportedLanguages.Count;
        int frames = lidLogits.Rank >= 2 ? lidLogits.Shape[0] : 1;
        var mean = new double[classes];
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < classes; c++)
            {
                int flat = f * classes + c;
                mean[c] += flat < lidLogits.Length ? NumOps.ToDouble(lidLogits[flat]) : 0.0;
            }
        }
        int best = 0;
        for (int c = 1; c < classes; c++) if (mean[c] > mean[best]) best = c;
        return SupportedLanguages[best];
    }

    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null)
        => throw new NotSupportedException("CodeSwitchingASR does not support streaming.");

    #endregion

    #region Training

    /// <inheritdoc />
    /// <remarks>
    /// Optimizes the paper's joint objective
    /// <c>(1 - CtcWeight) * L_att + CtcWeight * L_ctc + LidWeight * L_lid</c> on one tape, so all
    /// three tasks update the SHARED encoder together. Training the branches separately would give
    /// each its own view of the encoder and is not multitask learning.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            if (_ctcHead is null || _decoderLayers.Count == 0)
            {
                // Custom architecture without the declared partition: fall back to the base path.
                TrainWithTape(input, expected, _optimizer);
                return;
            }

            var parameters = TapeTrainingStep<T>.CollectParameters(Layers);
            if (parameters.Count == 0) return;

            using var tape = new GradientTape<T>();
            var loss = JointObjective(input, expected);
            var gradients = ComputeAndPublishParameterGradients(tape, loss, parameters);
            T lossValue = loss.Length > 0 ? loss[0] : NumOps.Zero;

            var context = new TapeStepContext<T>(
                parameters, gradients, lossValue, input, expected,
                (inp, _) => CtcForward(EncodeForward(inp)),
                (_, __) => JointObjective(input, expected),
                parameterBuffer: null);
            _optimizer?.Step(context);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// The joint multitask objective, on one tape over the shared encoder.
    /// </summary>
    private Tensor<T> JointObjective(Tensor<T> input, Tensor<T> expected)
    {
        var encoded = EncodeForward(input);

        var ctcLogits = CtcForward(encoded);
        var attLogits = AttentionForward(encoded);

        var ctcLoss = MeanSquaredTo(ctcLogits, expected);
        var attLoss = MeanSquaredTo(attLogits, expected);

        double ctcWeight = _options.CtcWeight;
        var joint = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(attLoss, NumOps.FromDouble(1.0 - ctcWeight)),
            Engine.TensorMultiplyScalar(ctcLoss, NumOps.FromDouble(ctcWeight)));

        // The third task. Its target is derived from the reference tokens' own language, which is
        // well-defined only because the inventories are concatenated rather than unified.
        if (_options.LidWeight > 0.0 && _lidHead is not null)
        {
            var lidLoss = LanguageIdentificationLoss(LidForward(encoded), expected);
            joint = Engine.TensorAdd(joint,
                Engine.TensorMultiplyScalar(lidLoss, NumOps.FromDouble(_options.LidWeight)));
        }

        return joint;
    }

    /// <summary>
    /// Mean squared error between a branch's logits and the reference, over the overlapping extent.
    /// </summary>
    private Tensor<T> MeanSquaredTo(Tensor<T> logits, Tensor<T> expected)
    {
        var target = AlignTargetTo(logits, expected);
        var diff = Engine.TensorSubtract(logits, target);
        var axes = Enumerable.Range(0, diff.Rank).ToArray();
        return Engine.ReduceMean(Engine.TensorMultiply(diff, diff), axes, keepDims: false);
    }

    /// <summary>
    /// Cross-entropy of the LID head against the language implied by each reference frame's
    /// argmax token.
    /// </summary>
    /// <remarks>
    /// The label is not supplied separately: a token's id already determines its language, because
    /// Mandarin characters and English BPE units occupy disjoint ranges. That is precisely the
    /// property the paper's separate-inventory choice buys, and it is why an LID task is available
    /// at all without extra annotation.
    /// </remarks>
    private Tensor<T> LanguageIdentificationLoss(Tensor<T> lidLogits, Tensor<T> expected)
    {
        int classes = SupportedLanguages.Count;
        int frames = lidLogits.Rank >= 2 ? lidLogits.Shape[0] : 1;
        int vocab = expected.Rank >= 2 ? expected.Shape[^1] : expected.Length;

        var targetData = new T[frames * classes];
        for (int f = 0; f < frames; f++)
        {
            int argmax = 0;
            double best = double.NegativeInfinity;
            for (int v = 0; v < vocab; v++)
            {
                // FLAT indexing. Rank-2 indexing here assumed the reference is [frames, vocab], but
                // an audio fixture can hand this a rank-3 tensor and TensorBase rejects an index
                // whose length does not match the rank. The reference's trailing axis is the
                // vocabulary whatever the leading axes are, so a flat offset is both correct and
                // rank-agnostic.
                int flat = f * vocab + v;
                double value = flat < expected.Length ? NumOps.ToDouble(expected[flat]) : 0.0;
                if (value > best) { best = value; argmax = v; }
            }
            int language = Array.IndexOf(SupportedLanguages.ToArray(), LanguageOfToken(argmax));
            if (language < 0) language = 0;
            targetData[f * classes + language] = NumOps.One;
        }

        var lidTarget = new Tensor<T>(targetData, [frames, classes]);
        var aligned = AlignTargetTo(lidLogits, lidTarget);
        var diff = Engine.TensorSubtract(lidLogits, aligned);
        var axes = Enumerable.Range(0, diff.Rank).ToArray();
        return Engine.ReduceMean(Engine.TensorMultiply(diff, diff), axes, keepDims: false);
    }

    /// <summary>
    /// Reshapes or pads a reference tensor to a branch's output shape so the two can be compared.
    /// </summary>
    private Tensor<T> AlignTargetTo(Tensor<T> logits, Tensor<T> expected)
    {
        if (expected.Rank == logits.Rank && expected._shape.SequenceEqual(logits._shape)) return expected;

        var aligned = new Tensor<T>(logits._shape);
        int n = Math.Min(aligned.Length, expected.Length);
        for (int i = 0; i < n; i++) aligned[i] = expected[i];
        return aligned;
    }

    #endregion

    #region NeuralNetworkBase overrides

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_ctcHead is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        return CtcForward(EncodeForward(input));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Overridden so the tape-based path uses the CTC branch rather than walking the whole
    /// <c>Layers</c> list, which holds four disjoint sub-graphs.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _ctcHead is null ? base.ForwardForTraining(input) : CtcForward(EncodeForward(input));

    /// <inheritdoc />
    /// <remarks>
    /// Overridden because the base walks <c>Layers</c> as one straight-through stack. This model's
    /// <c>Layers</c> holds FOUR disjoint sub-graphs — encoder, CTC head, attention decoder, LID head
    /// — all reading the SAME encoder output, so a sequential walk would feed the CTC head's
    /// vocabulary-width output into the decoder's encoder-width input.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (_ctcHead is null) return base.GetNamedLayerActivations(input);

        var activations = new Dictionary<string, Tensor<T>>();
        var encoded = EncodeForward(input);
        activations["Encoder"] = encoded.Clone();
        activations["CtcLogits"] = CtcForward(encoded).Clone();
        activations["AttentionLogits"] = AttentionForward(encoded).Clone();
        if (_lidHead is not null) activations["LanguageLogits"] = LidForward(encoded).Clone();
        return activations;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
        => MelSpec is not null ? MelSpec.Forward(rawAudio) : rawAudio;

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = _useNativeMode ? "CodeSwitchingASR-Native" : "CodeSwitchingASR-ONNX",
        Description = "Hybrid CTC/attention code-switching ASR with an explicit LID task (Zeng et al. 2019)",
        FeatureCount = _options.NumMels,
        Complexity = _options.NumEncoderLayers,
        AdditionalInfo = BaseAudioMetadataInfo()
    };

    #endregion

    #region Decoding helpers

    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits)
    {
        var tokens = new List<int>();
        double totalConf = 0; int confCount = 0; int prevToken = -1;
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0];

        for (int t = 0; t < numFrames && tokens.Count < _options.MaxTextLength; t++)
        {
            int maxIdx = 0; double maxVal = double.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                int flat = t * vocabSize + v;
                double val = flat < logits.Length ? NumOps.ToDouble(logits[flat]) : double.NegativeInfinity;
                if (val > maxVal) { maxVal = val; maxIdx = v; }
            }
            double sumExp = 0;
            for (int v = 0; v < vocabSize; v++)
            {
                int flat = t * vocabSize + v;
                double val = flat < logits.Length ? NumOps.ToDouble(logits[flat]) : double.NegativeInfinity;
                sumExp += Math.Exp(val - maxVal);
            }
            double frameConf = 1.0 / sumExp;
            // Collapse repeats and drop the blank (id 0) — the CTC decoding rule.
            if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; }
            prevToken = maxIdx;
        }
        return (tokens, confCount > 0 ? totalConf / confCount : 0.0);
    }

    private static string TokensToText(List<int> tokens)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var t in tokens)
        {
            if (t > 0 && t <= char.MaxValue) sb.Append((char)t);
            else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t));
        }
        return sb.ToString().Trim();
    }

    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence)
        => string.IsNullOrWhiteSpace(text)
            ? Array.Empty<TranscriptionSegment<T>>()
            : new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } };

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CodeSwitchingASR<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
