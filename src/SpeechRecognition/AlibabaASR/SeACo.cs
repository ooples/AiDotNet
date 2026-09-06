using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>
/// SeACo-Paraformer: hot-word customizable ASR
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "SeACo-Paraformer: A Non-Autoregressive ASR System with Flexible and Effective Hot-Word Customization Ability" (An et al., Alibaba DAMO, 2023)</item></list></para>
/// <para><b>For Beginners:</b> SeACo-Paraformer extends Paraformer with Semantic-Aware Contextual (SeACo) biasing for hot-word customization. A context encoder processes a list of bias phrases, and cross-attention between the decoder and context embeddings biases recognition to...</para>
/// <para>
/// SeACo-Paraformer extends Paraformer with Semantic-Aware Contextual (SeACo) biasing for hot-word customization. A context encoder processes a list of bias phrases, and cross-attention between the decoder and context embeddings biases recognition toward specified terms. This enables accurate recognition of domain-specific terminology, proper nouns, and rare words without retraining. The biasing is applied at the semantic level rather than shallow fusion.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SeACo-Paraformer model for hot-word customizable ASR
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 5000);
/// var model = new SeACo&lt;double&gt;(architecture);
///
/// // Or load a pre-trained ONNX model for context-biased recognition
/// var onnxModel = new SeACo&lt;double&gt;(architecture, "seaco.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SeACo-Paraformer: A Non-Autoregressive ASR System with Flexible and Effective Hot-Word Customization Ability", "https://arxiv.org/abs/2308.03266", Year = 2023, Authors = "An et al.")]
public partial class SeACo<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{

    /// <inheritdoc />
    public TrainingObjectiveKind TrainingObjectiveKind => TrainingObjectiveKind.MultiTask;

    /// <inheritdoc />
    public Tensor<T> ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    public T EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        EnsureBiasStageIsSupported();

        var savedTarget = _currentTarget;
        var savedHotwords = _activeHotwordIds;
        try
        {
            _currentTarget = target;
            Tensor<T> objectiveTarget = target;
            if (_options.TrainingStage != SeACoTrainingStage.Backbone)
            {
                var positions = CreateHotwordPositions(
                    target,
                    RandomHelper.CreateSeededRandom(_options.Seed ?? HotwordSamplingSeed));
                _activeHotwordIds = SampleHotwordIdsFromTarget(target, positions);
                if (_options.TrainingStage == SeACoTrainingStage.Bias)
                    objectiveTarget = CreateBiasTrainingTarget(target, positions);
            }
            else
            {
                _activeHotwordIds = null;
            }

            var prediction = ForwardForTraining(input);
            var loss = _options.TrainingStage == SeACoTrainingStage.Bias
                ? BiasStageLoss
                : LossFunction;
            return loss.ComputeTapeLoss(prediction, objectiveTarget).Data.Span[0];
        }
        finally
        {
            _currentTarget = savedTarget;
            _activeHotwordIds = savedHotwords;
        }
    }

    private readonly SeACoOptions _options; public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    private static readonly AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T> BiasStageLoss = new();
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public SeACo(NeuralNetworkArchitecture<T> architecture, string modelPath, SeACoOptions? options = null) : base(architecture) { _options = options ?? new SeACoOptions(); _useNativeMode = false; _hotwordRng = RandomHelper.CreateSeededRandom(_options.Seed ?? HotwordSamplingSeed); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "zh", "en" }; InitializeLayers(); }
    // Paraformer / SeACo-Paraformer (Gao et al., arXiv 2206.08317; Shi et al., arXiv 2308.03266) train
    // the token head with CROSS-ENTROPY over the vocabulary (alongside CTC and the predictor's MAE),
    // never a regression loss. AudioNeuralNetworkBase defaults to MeanSquaredErrorLoss, so this model
    // was descending MSE between its [1, 64, 8404] vocab LOGITS and a dense target — an objective the
    // paper never uses and one that cannot be fitted, which is why extra training made the measured
    // loss RISE (0.99 -> 44.38) while parameters barely moved (L2 195.796 -> 195.852). The head emits
    // raw logits, so use the fused log-softmax/NLL form, matching the other logit-head models here.
    public SeACo(NeuralNetworkArchitecture<T> architecture, SeACoOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture, new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>()) { _options = options ?? new SeACoOptions(); _useNativeMode = true; _optimizer = optimizer ?? CreateParaformerOptimizer(); _hotwordRng = RandomHelper.CreateSeededRandom(_options.Seed ?? HotwordSamplingSeed); base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels; SupportedLanguages = new[] { "zh", "en" }; InitializeLayers(); InstallParaformerObjective(); }

    /// <summary>
    /// Builds the optimizer Paraformer / SeACo-Paraformer specify.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Paraformer (Gao et al., arXiv 2206.08317) and SeACo-Paraformer (Shi et al., arXiv 2308.03266)
    /// train with <b>Adam</b> at a peak learning rate of <b>5e-4</b>. This model previously used
    /// <c>AdamWOptimizer</c> with DEFAULT options -- lr 1e-3 (2x the paper's peak) plus AdamW's
    /// decoupled weight decay of 0.01, which the paper does not use at all.
    /// </para>
    /// <para>
    /// The measured consequence was violent divergence rather than training. Loss over 12 steps on a
    /// fixed pair: 1.386 -> 1.780 -> 6.462 -> 1.667 -> 7.809 -> 11.805 -> 11.968 -> 11.061 -> 6.372
    /// -> 3.812 -> 11.534 -> 8.189, with cumulative parameter movement reaching 16 against an L2 of
    /// ~195. That oscillation is the signature of an oversized step, not of first-step overshoot that
    /// settles.
    /// </para>
    /// <para>
    /// No warmup length is invented here. The papers warm up over tens of thousands of iterations,
    /// which at the handful of steps the invariants run would reduce the rate to ~0 and simply
    /// replace divergence with no learning at all -- the trap already measured on SAM, where a
    /// faithful 250-step warmup merely DELAYED the blow-up. Callers who train for real can pass any
    /// scheduler through the constructor's optimizer parameter.
    /// </para>
    /// </remarks>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateParaformerOptimizer()
        => new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
            });

    /// <summary>
    /// Transcribes audio using SeACo's context-biased CIF parallel decoding.
    /// Per An et al. (2023): the CIF encoder processes speech, context encoder encodes
    /// bias phrases, and cross-attention in the decoder biases toward hot words.
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
            logits = features;
            foreach (var l in Layers) logits = l.Forward(logits);
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
            Segments = includeTimestamps ? ExtractSegments(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool includeTimestamps = false, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var features = PreprocessAudio(audio); Tensor<T> logits; if (IsOnnxMode && OnnxEncoder is not null) logits = OnnxEncoder.Run(features); else { logits = features; foreach (var l in Layers) logits = l.Forward(logits); } var (tokens, _) = CTCGreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(lang == detected ? primaryProb : otherProb); return result; }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("SeACo does not support streaming.");

    /// <summary>
    /// Builds the Paraformer ASR backbone followed by SeACo's bias branch, recording where one ends and
    /// the other begins so the paper's stage-2 freeze can be applied to a real parameter partition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SeACo-Paraformer (Shi et al., arXiv 2308.03266, §3) adds a bias encoder, bias decoder and bias
    /// out layer on top of a Paraformer backbone. Those modules were previously MISSING entirely — this
    /// class wired only CreateDefaultParaformerLayers, i.e. plain Paraformer — so the model carried the
    /// SeACo name while none of the paper's hot-word machinery existed, there were no bias parameters to
    /// freeze or train, and a caller's hotword list could not affect decoding at all.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            // A caller-supplied list has an unknown layout, so no backbone/bias split can be asserted.
            _backboneLayerCount = Layers.Count;
            _biasLayerCount = 0;
            return;
        }

        // decoderDim MUST equal encoderDim. SeACo Eq 2 defines the encoded hotwords as Z in R^(n x d) —
        // ONE model width d — and Eq 3 attends over that same Z from both the decoder hidden state D and
        // the CIF acoustic embedding E. Leaving decoderDim at the factory's 512 default while encoderDim
        // was 32 gave D width 512 and E width 32, so a single bias-attention pair could not serve both:
        // it threw "Input embedding dimension (512) does not match weight dimension (32)".
        var backbone = new List<ILayer<T>>(LayerHelper<T>.CreateDefaultParaformerLayers(
            encoderDim: _options.EncoderDim,
            decoderDim: _options.EncoderDim,
            numEncoderLayers: _options.NumEncoderLayers,
            numDecoderLayers: _options.NumDecoderLayers,
            numAttentionHeads: _options.NumAttentionHeads,
            feedForwardDim: _options.FeedForwardDim,
            numMels: _options.NumMels,
            vocabSize: _options.VocabSize,
            dropoutRate: _options.DropoutRate));

        var bias = new List<ILayer<T>>(LayerHelper<T>.CreateSeACoBiasLayers(
            vocabSize: _options.VocabSize,
            encoderDim: _options.EncoderDim,
            numAttentionHeads: _options.NumAttentionHeads,
            dropoutRate: _options.DropoutRate));

        Layers.AddRange(backbone);
        Layers.AddRange(bias);
        _backboneLayerCount = backbone.Count;
        _biasLayerCount = bias.Count;

        ValidateBiasBranchLayout(bias);
    }

    /// <summary>Number of layers <c>CreateSeACoBiasLayers</c> is contracted to emit.</summary>
    /// <remarks>
    /// ApplyHotwordBias indexes the first five layers, casts positions +2 and +3 to
    /// <see cref="MultiHeadAttentionLayer{T}"/>, and requires a final dense projection. Those reads are a contract with the
    /// factory that nothing checked: a factory change, or a layout this model does not support, showed
    /// up as an ArgumentOutOfRangeException or an InvalidCastException thrown from the middle of an
    /// inference call, naming an index rather than the mismatch. Checked once at construction instead.
    /// </remarks>
    private const int SeACoBiasLayerCount = 6;

    /// <summary>
    /// Confirms the bias branch has the shape the hotword forward indexes into, before any inference
    /// can reach it.
    /// </summary>
    private void ValidateBiasBranchLayout(List<ILayer<T>> bias)
    {
        if (bias.Count == 0)
        {
            // No bias branch is a supported configuration: Eq 5 then returns P_ASR unchanged, and
            // HasBiasBranch already gates every hotword path on it.
            return;
        }

        if (bias.Count < SeACoBiasLayerCount)
        {
            throw new InvalidOperationException(
                $"SeACo's bias branch needs at least {SeACoBiasLayerCount} layers (embedding, LSTM, " +
                $"two multi-head attentions, normalization, output projection) but {bias.Count} were supplied. Hotword " +
                "biasing cannot be applied to this layout.");
        }

        if (bias[2] is not MultiHeadAttentionLayer<T> || bias[3] is not MultiHeadAttentionLayer<T>)
        {
            throw new InvalidOperationException(
                "SeACo's bias branch expects multi-head attention at bias positions 2 and 3 (the " +
                $"decoder- and encoder-attending blocks), but found {bias[2].GetType().Name} and " +
                $"{bias[3].GetType().Name}. Hotword biasing cannot be applied to this layout.");
        }

        if (bias[^1] is not DenseLayer<T>)
        {
            throw new InvalidOperationException(
                $"SeACo's bias branch must end in a dense VocabSize + 1 output projection, but found " +
                $"{bias[^1].GetType().Name}. Hotword biasing cannot be applied to this layout.");
        }
    }

    /// <summary>Number of leading layers that form the Paraformer ASR backbone.</summary>
    private int _backboneLayerCount;

    /// <summary>Number of trailing layers that form SeACo's bias branch.</summary>
    private int _biasLayerCount;

    /// <summary>Whether a real backbone/bias partition is available for staged training.</summary>
    internal bool HasBiasBranch => _biasLayerCount > 0;

    /// <summary>Rejects paper stage-2 training when no validated bias branch is present.</summary>
    private void EnsureBiasStageIsSupported()
    {
        if (_options.TrainingStage == SeACoTrainingStage.Bias && !HasBiasBranch)
        {
            throw new InvalidOperationException(
                "Bias-stage training requires the native SeACo bias branch. " +
                "Caller-supplied Architecture.Layers does not define a validated backbone/bias partition.");
        }
    }

    /// <summary>Half-open range of Layers holding the bias branch.</summary>
    internal (int Start, int End) BiasLayerRange => (_backboneLayerCount, _backboneLayerCount + _biasLayerCount);

    /// <summary>Number of flattened parameters owned by the Paraformer backbone.</summary>
    internal long BackboneParameterCount
        => Layers.Take(_backboneLayerCount).Sum(layer => layer.ParameterCount);
    /// <summary>
    /// Runs the ASR backbone. The bias branch is a PARALLEL module, not a continuation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SeACo's bias encoder / decoder / out-layer form a second branch (arXiv 2308.03266, Fig. 1) whose
    /// encoder consumes HOTWORD TOKEN IDS and whose decoder takes the backbone's states as queries with
    /// the encoded hotwords as keys/values. Walking <c>Layers</c> straight through would feed backbone
    /// features into an <c>EmbeddingLayer</c> that expects token ids, so the branch boundary recorded by
    /// <see cref="InitializeLayers"/> is respected here — the same reason SAM2 overrides its forward
    /// instead of letting the base class walk a branching layer list in order.
    /// </para>
    /// <para>
    /// Biasing is applied only when a hotword list is supplied; without one the paper's default
    /// &lt;blank&gt; hotword leaves recognition unchanged, which is exactly the general-ASR parity its
    /// Table 2 reports.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);

        int backboneEnd = _backboneLayerCount > 0 ? _backboneLayerCount : Layers.Count;
        int cifIndex = IndexOfCifLayer();

        // Run the backbone but STOP BEFORE its final vocabulary projection: that last Dense turns the
        // decoder's hidden state D into P_ASR, and Eq 3's attentions operate on D (width = decoderDim),
        // not on the vocabulary logits. Feeding logits in threw
        // "Input embedding dimension (4) does not match weight dimension (32)" — the width was VocabSize.
        // §3.1: "we retain the CIF output E and parallel decoder [hidden state D]".
        int hiddenEnd = backboneEnd - 1;
        var hidden = input;
        Tensor<T>? acoustic = null;
        for (int i = 0; i < hiddenEnd; i++)
        {
            hidden = Layers[i].Forward(hidden);
            if (i == cifIndex) acoustic = hidden;   // E, the CIF acoustic embedding
        }

        // P_ASR: the backbone's own logits, from the final vocabulary projection.
        var pAsr = Layers[hiddenEnd].Forward(hidden);

        // Eq 5: "When there is no hotword incoming or no hotword detected, SeACo-Paraformer uses P_ASRi
        // only." So with no active hotwords the backbone's logits are returned UNCHANGED.
        if (!HasBiasBranch || _activeHotwordIds is null)
        {
            return pAsr;
        }

        var pBias = ApplyHotwordBias(hidden, acoustic ?? hidden, _activeHotwordIds);
        return MergeBiasedProbabilities(pAsr, pBias);
    }

    /// <summary>
    /// Merges backbone and bias logits per arXiv 2308.03266 Eq 5.
    /// </summary>
    /// <param name="pAsr">Backbone logits, [.., V].</param>
    /// <param name="pBias">Bias-branch logits, [.., V + 1]; the extra trailing slot is the
    /// &lt;no-bias&gt; token '#'.</param>
    /// <returns>Merged logits, [.., V].</returns>
    /// <remarks>
    /// Eq 5 is per-position and conditional:
    /// <c>P_m = P_ASR</c> when <c>argmax P_bi = &lt;no-bias&gt;</c>, else
    /// <c>lambda * P_bi + (1 - lambda) * P_ASR</c>. The argmax is a discrete decision and is read
    /// off-tape as a 0/1 gate; the blended values are produced with Engine ops so gradient still
    /// reaches both branches wherever a hotword IS detected.
    /// </remarks>
    private Tensor<T> MergeBiasedProbabilities(Tensor<T> pAsr, Tensor<T> pBias)
    {
        int vocab = _options.VocabSize;
        int noBias = _options.ResolveHotwordMaskTokenId();                       // configured '#' no-bias slot
        int rowsAsr = Math.Max(1, pAsr.Length / Math.Max(1, vocab));
        int rowsBias = Math.Max(1, pBias.Length / Math.Max(1, vocab + 1));

        // The two branches describe the same positions, so a disagreement is a shape bug upstream.
        // Silently blending the shorter prefix would train the bias branch against positions the
        // backbone scored differently.
        if (rowsAsr != rowsBias)
        {
            throw new InvalidOperationException(
                $"The ASR branch produced {rowsAsr} positions but the bias branch produced {rowsBias}. " +
                "Eq 5 merges them position by position, so they must agree.");
        }

        int rows = rowsAsr;
        var eng = Engine;
        var originalShape = pAsr.Shape.ToArray();

        // WHERE the blend applies is a discrete argmax, so it is read off-tape -- that matches the
        // paper, which makes a hard per-position choice. WHAT the blend computes must stay on tape,
        // and that is what this rewrite fixes: the previous body wrote scalars into a Clone()'s
        // Data.Span, and raw span writes are invisible to autodiff. Stage 2 trains the bias branch, so
        // its gradient path ran through a tensor nothing had recorded and the bias parameters got no
        // signal from this blend at all.
        var gate = new Tensor<T>([rows, 1]);
        var biasSpan = pBias.Data.Span;
        for (int r = 0; r < rows; r++)
        {
            int argmax = 0;
            double best = double.NegativeInfinity;
            for (int v = 0; v <= vocab; v++)
            {
                double val = NumOps.ToDouble(biasSpan[(r * (vocab + 1)) + v]);
                if (val > best) { best = val; argmax = v; }
            }

            // "no hotword detected" -> keep P_ASR for this position.
            gate[r, 0] = argmax == noBias ? NumOps.Zero : NumOps.One;
        }

        // Eq 5, written so both branches stay differentiable:
        //   merged = P_ASR + gate * lambda * (P_bias - P_ASR)
        // gate 0 leaves P_ASR exactly; gate 1 gives lambda*P_bias + (1-lambda)*P_ASR. The [rows, 1]
        // gate broadcasts across the vocabulary axis.
        var asr2d = eng.Reshape(pAsr, new[] { rows, vocab });
        var bias2d = eng.Reshape(
            eng.TensorNarrow(pBias, pBias.Rank - 1, 0, vocab), new[] { rows, vocab });

        var delta = eng.TensorSubtract(bias2d, asr2d);
        var scaled = eng.TensorMultiplyScalar(delta, NumOps.FromDouble(_options.BiasMergeLambda));
        var merged = eng.TensorAdd(asr2d, eng.TensorMultiply(scaled, gate));

        return eng.Reshape(merged, originalShape);
    }

    /// <summary>
    /// Hotword token ids [n, maxLen] active for the current forward, or null when none are supplied.
    /// </summary>
    /// <remarks>
    /// Null means "no hotword incoming", which arXiv 2308.03266 Eq 5 answers by returning P_ASR
    /// unchanged. During training this is populated from RANDOMLY SAMPLED hotwords (§3.1, §4.2), which
    /// is how the bias parameters receive gradient without altering inference behaviour.
    /// </remarks>
    [Scratch]
    private Tensor<T>? _activeHotwordIds;

    /// <summary>
    /// The target for the current training step, needed by the GLM sampler.
    /// </summary>
    /// <remarks>
    /// Paraformer's sampler substitutes TARGET embeddings into the acoustic embedding, so the two-pass
    /// forward needs Y. <c>ForwardNativeForTraining</c> receives only the input, so the target is stashed
    /// here by <see cref="Train"/> for the duration of the step and cleared in its finally block.
    /// </remarks>
    [Scratch]
    private Tensor<T>? _currentTarget;
    /// <summary>
    /// Trains the parameter group selected by <see cref="SeACoOptions.TrainingStage"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SeACo-Paraformer (Shi et al., arXiv 2308.03266, §3) trains in two SEPARATE stages: the Paraformer
    /// backbone first, then "with a well-trained Paraformer model freezing" only the bias encoder, bias
    /// decoder and bias out layer, because "the training of bias-related parameters is separate from the
    /// ASR training". Freezing is what stops hot-word supervision from degrading general recognition.
    /// </para>
    /// <para>
    /// The freeze is applied by flipping the excluded layers' training mode off and restoring it
    /// afterwards, so a frozen group receives no gradient for the step regardless of optimizer.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        EnsureBiasStageIsSupported();

        SetTrainingMode(true);
        var frozen = FreezeForStage(_options.TrainingStage);
        try
        {
            // Stage 2 supervises the bias branch with SeACo §3's hotword-position-aware criterion:
            // labels at non-hotword positions are replaced by '#', so the bias parameters receive
            // gradient ONLY where a sampled hotword sits. Training them against an unmasked
            // full-sequence CE (what this model did before the bias branch existed) asks them to
            // re-learn the entire transcription, which is why its loss surface moved 8x while the
            // parameters barely moved at all.
            // The paper's stage-1 backbone training has no bias branch. Sample and activate
            // hotwords only for the bias/joint stages; otherwise the supposedly frozen branch
            // still changes the forward objective even though its parameters cannot update.
            var target = expected;
            if (_options.TrainingStage != SeACoTrainingStage.Backbone)
            {
                var positions = SampleHotwordPositions(expected);
                _activeHotwordIds = SampleHotwordIdsFromTarget(expected, positions);
                if (_options.TrainingStage == SeACoTrainingStage.Bias)
                    target = CreateBiasTrainingTarget(expected, positions);
            }
            else
            {
                _activeHotwordIds = null;
            }

            // The GLM sampler is supervised by the transcription target, not by the bias
            // branch's appended '#' class target.
            _currentTarget = expected;

            var savedLoss = LossFunction;
            try
            {
                if (_options.TrainingStage == SeACoTrainingStage.Bias)
                    LossFunction = BiasStageLoss;
                TrainWithTape(input, target, _optimizer);
            }
            finally
            {
                LossFunction = savedLoss;
            }
        }
        finally
        {
            _activeHotwordIds = null;
            _currentTarget = null;
            foreach (var l in frozen) l.SetTrainingMode(true);
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Puts the layers excluded by <paramref name="stage"/> into inference mode for one step.
    /// </summary>
    /// <returns>The layers that were frozen, so the caller can restore them.</returns>
    /// <remarks>
    /// Returns an empty list for <see cref="SeACoTrainingStage.Joint"/>, and also when no bias branch
    /// exists (a caller-supplied Architecture.Layers list, whose layout is unknown) — freezing an
    /// unidentified partition would silently train the wrong half.
    /// </remarks>
    private List<ILayer<T>> FreezeForStage(SeACoTrainingStage stage)
    {
        var frozen = new List<ILayer<T>>();
        if (stage == SeACoTrainingStage.Joint || !HasBiasBranch)
        {
            return frozen;
        }

        var (biasStart, biasEnd) = BiasLayerRange;
        for (int i = 0; i < Layers.Count; i++)
        {
            bool isBias = i >= biasStart && i < biasEnd;
            bool freeze = stage == SeACoTrainingStage.Bias ? !isBias : isBias;
            if (!freeze) continue;

            Layers[i].SetTrainingMode(false);
            frozen.Add(Layers[i]);
        }

        return frozen;
    }

    /// <summary>
    /// Transcribes audio with a caller-supplied hot-word list, biasing recognition toward those terms.
    /// </summary>
    /// <param name="audio">Raw audio samples.</param>
    /// <param name="hotwords">Terms to bias toward (names, jargon, product names). Empty or null yields
    /// the same result as <see cref="Transcribe(Tensor{T}, string?, bool)"/>.</param>
    /// <param name="language">Optional language override.</param>
    /// <param name="includeTimestamps">Whether to include segment timestamps.</param>
    /// <returns>The transcription, with hot-word biasing applied when a list is supplied.</returns>
    /// <remarks>
    /// <para>
    /// This is SeACo-Paraformer's entire purpose (Shi et al., arXiv 2308.03266): "flexible and effective
    /// hot-word customization" WITHOUT retraining, so the list is supplied per call. The model
    /// previously exposed no way to pass one at all, which made the SeACo name unearned regardless of
    /// what its layers contained.
    /// </para>
    /// <para>
    /// Biasing runs the bias branch: the hotwords are embedded and contextualised by the bias encoder,
    /// the bias decoder attends from the backbone's states over those representations, and the bias out
    /// layer produces vocabulary logits that are combined with the backbone's. With no hotwords the
    /// paper's default &lt;blank&gt; applies and the backbone's own logits are returned unchanged —
    /// matching the general-ASR parity its Table 2 reports.
    /// </para>
    /// </remarks>
    public TranscriptionResult<T> Transcribe(
        Tensor<T> audio,
        IReadOnlyList<string>? hotwords,
        string? language = null,
        bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));

        if (hotwords is null || hotwords.Count == 0 || !HasBiasBranch || IsOnnxMode)
        {
            return Transcribe(audio, language, includeTimestamps);
        }

        // PREPROCESSED, LIKE EVERY OTHER INFERENCE PATH. This passed the raw waveform straight into
        // PredictCore while the sibling Transcribe overload calls PreprocessAudio first, so the biased
        // and unbiased paths read from different input domains -- one mel features, one audio samples --
        // through the same encoder. Whichever of the two does not throw on the shape is the more
        // dangerous outcome: it returns a transcription computed from noise.
        Tensor<T> biased;
        var biasedFeatures = PreprocessAudio(audio);
        _activeHotwordIds = BuildHotwordIds(hotwords);
        try
        {
            biased = PredictCore(biasedFeatures);
        }
        finally
        {
            _activeHotwordIds = null;
        }

        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(biased);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = ExtractSegments(text, duration, confidence),
        };
    }

    /// <summary>
    /// Runs SeACo's bias branch over <paramref name="hotwords"/>, per arXiv 2308.03266 Eq 2-4.
    /// </summary>
    /// <param name="decoderHidden">D, the parallel decoder's hidden state (pre-vocabulary-projection).</param>
    /// <param name="acousticEmbedding">E, the CIF acoustic embedding.</param>
    /// <param name="hotwords">Hotword strings; an empty list still runs the branch under the paper's
    /// default &lt;blank&gt; hotword.</param>
    /// <returns>Biased logits over the ASR vocabulary plus the appended '#' no-bias token.</returns>
    /// <remarks>
    /// <para>
    /// Eq 2: <c>Z = LSTM(EMB(H))</c> encodes the hotword character sequences.
    /// Eq 3: <c>D+ = MHA(D, Z, Z)</c> and <c>E+ = MHA(E, Z, Z)</c> — the paper sends "the CIF output and
    /// parallel decoder output ... to bias decoder separately", so BOTH streams attend over Z.
    /// Eq 4: <c>P_bi = BiasOutLayer(D+ + E+)</c> sums them before the projection.
    /// </para>
    /// <para>
    /// Composed from Engine ops throughout so the branch is differentiable, which stage-2 training
    /// requires.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyHotwordBias(Tensor<T> decoderHidden, Tensor<T> acousticEmbedding, Tensor<T> hotwordIds)
    {
        var (biasStart, _) = BiasLayerRange;

        // Layer contract emitted by CreateSeACoBiasLayers.
        var embedding = Layers[biasStart];
        var lstm = Layers[biasStart + 1];
        // Typed as the concrete attention layer: ILayer<T> only declares Forward(Tensor<T>), and Eq 3
        // needs the (Q, K=V) cross-attention overload that MultiHeadAttentionLayer<T> adds.
        var attendD = (MultiHeadAttentionLayer<T>)Layers[biasStart + 2];
        var attendE = (MultiHeadAttentionLayer<T>)Layers[biasStart + 3];
        var norm = Layers[biasStart + 4];
        int biasEnd = BiasLayerRange.End;
        var outLayer = Layers[biasEnd - 1];

        // Eq 2.
        var z = lstm.Forward(embedding.Forward(hotwordIds));

        // Eq 3 — CROSS-attention, D and E as SEPARATE queries over the same encoded hotwords Z:
        //   D+ = MHA(D, Z, Z)   the parallel decoder's hidden state
        //   E+ = MHA(E, Z, Z)   the CIF acoustic embedding
        // §3.1: "the CIF output and parallel decoder output are sent to bias decoder separately". The
        // two-argument Forward overload is (Q, K=V); the single-argument one is SELF-attention and never
        // looks at the hotwords at all.
        var dPlus = attendD.Forward(decoderHidden, z);
        var ePlus = attendE.Forward(acousticEmbedding, z);

        // Eq 4.
        var summed = Engine.TensorAdd(dPlus, ePlus);
        var biasHidden = norm.Forward(summed);
        for (int i = biasStart + 5; i < biasEnd - 1; i++)
        {
            biasHidden = Layers[i].Forward(biasHidden);
        }

        return outLayer.Forward(biasHidden);
    }

    /// <summary>
    /// Two-pass GLM training forward (Paraformer §2.3 Eq 4-5; SeACo §3.1 Eq 1).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The published procedure is NOT a single gradient pass:
    /// </para>
    /// <list type="number">
    /// <item><b>Pass 1, without gradient.</b> SeACo §3.1: the CIF predictor generates the acoustic
    /// embedding E "which makes up Pass1 in training (w/o gradient)". Its only purpose is to produce a
    /// prediction Yhat to measure against the target.</item>
    /// <item><b>Sampler.</b> Paraformer §2.3 Eq 4: <c>GLM(Y, Yhat) = Sampler(Es | Ea, Ec, ceil(lambda *
    /// d(Y, Yhat)))</c> with <c>d</c> the Hamming distance (Eq 5). It "incorporates target embeddings Ec
    /// by randomly substituting ceil(lambda*d) tokens into acoustic embedding Ea to generate semantic
    /// embedding Es". Because d shrinks as the model improves, the substitution rate anneals with no
    /// explicit schedule.</item>
    /// <item><b>Pass 2, with gradient.</b> The parallel decoder runs on Es and is "trained to predict the
    /// target tokens with semantic context, enabling the model to learn interdependency between output
    /// tokens" — the interdependency a NAR decoder otherwise cannot learn, which §3 Table 1 measures as
    /// the gap between vanilla NAR and Paraformer.</item>
    /// </list>
    /// <para>
    /// Training with a single unsampled pass (what this model did before) trains the parallel decoder
    /// without that semantic context at all, which is why its training-family invariants misbehaved.
    /// </para>
    /// <para>
    /// Falls back to the plain forward when the stack has no CIF layer (a caller-supplied
    /// Architecture.Layers list) or no target is available — there is no Ea boundary to sample at, and
    /// inventing one would train something the paper does not describe.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (_options.TrainingStage == SeACoTrainingStage.Bias && HasBiasBranch)
        {
            return ForwardBiasBranchForTraining(input);
        }

        int cifIndex = IndexOfCifLayer();
        if (cifIndex < 0 || _currentTarget is null || !HasBiasBranch)
        {
            return PredictCore(input);
        }

        // ---- Pass 1: no gradient. InferenceMode is the repo's torch.inference_mode() equivalent, so
        // nothing here is recorded on the tape.
        Tensor<T> pass1Logits;
        using (InferenceMode.Enter())
        {
            pass1Logits = PredictCore(input);
        }

        // Acoustic embedding Ea: the stack up to and including the CIF layer.
        var acoustic = input;
        for (int i = 0; i <= cifIndex; i++)
        {
            acoustic = Layers[i].Forward(acoustic);
        }

        // ---- Sampler: Es = Ea with ceil(lambda * HammingDistance) positions replaced by Ec.
        var semantic = ApplyGlmSampler(acoustic, pass1Logits, _currentTarget);

        // ---- Pass 2: with gradient, from the decoder onward.
        var decoded = semantic;
        int projectionIndex = _backboneLayerCount - 1;
        for (int i = cifIndex + 1; i < projectionIndex; i++)
        {
            decoded = Layers[i].Forward(decoded);
        }

        var pAsr = Layers[projectionIndex].Forward(decoded);
        if (_options.TrainingStage == SeACoTrainingStage.Joint && _activeHotwordIds is not null)
        {
            var pBias = ApplyHotwordBias(decoded, acoustic, _activeHotwordIds);
            return MergeBiasedProbabilities(pAsr, pBias);
        }

        return pAsr;
    }

    /// <summary>
    /// Runs SeACo stage 2 with the trained Paraformer backbone detached from autodiff and
    /// returns the bias branch's V+1 logits, including the appended '#' class.
    /// </summary>
    private Tensor<T> ForwardBiasBranchForTraining(Tensor<T> input)
    {
        if (_activeHotwordIds is null)
            throw new InvalidOperationException("Bias-stage training requires sampled hotword ids.");

        int cifIndex = IndexOfCifLayer();
        if (cifIndex < 0)
            throw new InvalidOperationException("Bias-stage training requires a Paraformer CIF layer.");

        Tensor<T> hidden;
        Tensor<T>? acoustic = null;
        int projectionIndex = _backboneLayerCount - 1;
        using (InferenceMode.Enter())
        {
            hidden = input;
            for (int i = 0; i < projectionIndex; i++)
            {
                hidden = Layers[i].Forward(hidden);
                if (i == cifIndex) acoustic = hidden;
            }
        }

        // InferenceMode prevents new tape records, but its result tensors can still carry
        // creator metadata from prior lazy-shape/evaluation forwards. Materialize fresh leaf
        // tensors so stage-2 gradients end at D and E exactly as the paper's frozen backbone
        // requires; sharing the values is correct, sharing the graph is not.
        var detachedHidden = DetachFromAutodiff(hidden);
        var detachedAcoustic = DetachFromAutodiff(acoustic ?? hidden);
        return ApplyHotwordBias(detachedHidden, detachedAcoustic, _activeHotwordIds);
    }

    /// <summary>Copies a tensor's values into a fresh leaf with no autodiff creator.</summary>
    private static Tensor<T> DetachFromAutodiff(Tensor<T> source)
    {
        var detached = new Tensor<T>(source.Shape.ToArray());
        source.Data.Span.CopyTo(detached.Data.Span);
        return detached;
    }

    /// <summary>Index of the CIF layer in the backbone, or -1 when absent.</summary>
    private int IndexOfCifLayer()
    {
        int end = _backboneLayerCount > 0 ? _backboneLayerCount : Layers.Count;
        for (int i = 0; i < end; i++)
        {
            if (Layers[i] is CifAlignmentLayer<T>) return i;
        }

        return -1;
    }

    /// <summary>
    /// Paraformer's GLM sampler: substitutes target embeddings into the acoustic embedding.
    /// </summary>
    /// <param name="acoustic">Ea, the CIF acoustic embedding.</param>
    /// <param name="pass1Logits">Pass-1 logits, used only to derive Yhat.</param>
    /// <param name="target">Y, the training target.</param>
    /// <returns>Es, the semantic embedding fed to the parallel decoder.</returns>
    /// <remarks>
    /// <para>
    /// Substitution count is <c>ceil(SamplerLambda * d(Y, Yhat))</c> with d the Hamming distance
    /// (Paraformer Eq 5), so a poorly-trained model substitutes many positions and a well-trained one
    /// few. SeACo §3.1 adds that the replaced steps are chosen "according to the correctly recognized
    /// positions", so MISMATCHED positions are substituted first — those are exactly the steps whose
    /// acoustic embedding the model got wrong and where target context helps most.
    /// </para>
    /// <para>
    /// Ec comes from the bias branch's embedding, which SeACo §3.1 specifies is "parameter shared with
    /// ASR embedding" — so one embedding serves both the hotword encoder and the sampler, exactly as
    /// the paper describes rather than as a second copy.
    /// </para>
    /// <para>
    /// The substitution is a write into a cloned tensor: it selects WHICH rows carry target context and
    /// is not itself a differentiable operation, matching Pass 1 being gradient-free. Gradient flows
    /// through Pass 2 from Es onward.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyGlmSampler(Tensor<T> acoustic, Tensor<T> pass1Logits, Tensor<T> target)
    {
        int vocab = _options.VocabSize;
        int positions = acoustic.Rank > 1 ? acoustic.Shape[acoustic.Rank - 2] : 1;
        int width = acoustic.Shape[acoustic.Rank - 1];

        int targetCols = target.Rank > 1 ? target.Shape[^1] : target.Length;
        int targetRows = target.Length / Math.Max(1, targetCols);
        int logitRows = Math.Max(1, pass1Logits.Length / Math.Max(1, vocab));
        int comparable = Math.Min(positions, Math.Min(targetRows, logitRows));

        // Eq 5: d(Y, Yhat) = sum_n [y_n != yhat_n], plus the per-position mismatch flags SeACo uses to
        // decide WHICH steps to replace.
        var mismatched = new List<int>();
        for (int n = 0; n < comparable; n++)
        {
            int predicted = ArgMaxRow(pass1Logits, n, vocab);
            int actual = ArgMaxRow(target, n, targetCols);
            if (predicted != actual) mismatched.Add(n);
        }

        int distance = mismatched.Count;
        if (distance == 0) return acoustic;

        int substitutions = (int)Math.Ceiling(_options.SamplerLambda * distance);
        substitutions = Math.Min(substitutions, comparable);
        if (substitutions <= 0) return acoustic;

        // Ec: target token embeddings through the SHARED embedding (SeACo §3.1).
        var (biasStart, _) = BiasLayerRange;
        var sharedEmbedding = Layers[biasStart];

        var targetIds = new Tensor<T>(new[] { comparable, 1 });
        for (int n = 0; n < comparable; n++)
        {
            targetIds.Data.Span[n] = NumOps.FromDouble(ArgMaxRow(target, n, targetCols) % vocab);
        }

        Tensor<T> targetEmbedding;
        using (InferenceMode.Enter())
        {
            targetEmbedding = sharedEmbedding.Forward(targetIds);
        }

        // Replace mismatched positions first, then any remaining budget over the rest, in a
        // deterministically seeded order — an unseeded draw here would make the objective itself vary
        // run to run, the flaky-RNG failure mode this repository has already been bitten by.
        var order = new List<int>(mismatched);
        if (order.Count < substitutions)
        {
            var rng = RandomHelper.CreateSeededRandom(GlmSamplerSeed);
            for (int n = 0; n < comparable && order.Count < substitutions; n++)
            {
                if (!mismatched.Contains(n) && rng.NextDouble() < 0.5) order.Add(n);
            }

            for (int n = 0; n < comparable && order.Count < substitutions; n++)
            {
                if (!order.Contains(n)) order.Add(n);
            }
        }

        int embedWidth = targetEmbedding.Length / Math.Max(1, comparable);
        var semantic = acoustic.Clone();
        for (int k = 0; k < substitutions && k < order.Count; k++)
        {
            int n = order[k];
            for (int w = 0; w < width && w < embedWidth; w++)
            {
                semantic.Data.Span[(n * width) + w] = targetEmbedding.Data.Span[(n * embedWidth) + w];
            }
        }

        return semantic;
    }

    /// <summary>Argmax over one row of a [rows, cols] tensor.</summary>
    private int ArgMaxRow(Tensor<T> tensor, int row, int cols)
    {
        int best = 0;
        double bestValue = double.NegativeInfinity;
        int offset = row * cols;
        for (int c = 0; c < cols && offset + c < tensor.Length; c++)
        {
            double v = NumOps.ToDouble(tensor.Data.Span[offset + c]);
            if (v > bestValue) { bestValue = v; best = c; }
        }

        return best;
    }

    /// <summary>Deterministic seed for the sampler's substitution order.</summary>
    private const int GlmSamplerSeed = 4242;

    /// <summary>
    /// Installs Paraformer's training objective, <c>gamma * L_CE + L_MAE</c> (arXiv 2206.08317, Eq 6).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Called AFTER <see cref="InitializeLayers"/> because the MAE term reads the CIF predictor's token
    /// count, and the layer must exist first. The lookup is a lazy callback rather than a captured
    /// reference so it keeps working across Clone / Deserialize, which rebuild the layer list.
    /// </para>
    /// <para>
    /// Skipped when no CIF layer is present (a caller-supplied Architecture.Layers list), in which case
    /// there is no token-count head to supervise and plain CE remains correct — rather than adding a
    /// term with nothing behind it.
    /// </para>
    /// </remarks>
    private void InstallParaformerObjective()
    {
        if (!_useNativeMode) return;
        if (!Layers.OfType<CifAlignmentLayer<T>>().Any()) return;

        LossFunction = new ParaformerObjective<T>(
            new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>(),
            () => Layers.OfType<CifAlignmentLayer<T>>().FirstOrDefault()?.LastPredictedTokenCount,
            // NAMED, because ParaformerObjective's third parameter is the optional targetTokenCount
            // Func -- passing the weights positionally put CeWeight there and would not compile once
            // both slices of #1789 are in one assembly. The target count is genuinely absent here:
            // SeACo supervises the predicted count against the label length inside the objective.
            ceWeight: _options.CeWeight,
            maeWeight: _options.MaeWeight);
    }

    /// <summary>
    /// Encodes hotword strings into the [n, l_max] token-id tensor the bias encoder consumes.
    /// </summary>
    private Tensor<T> BuildHotwordIds(IReadOnlyList<string> hotwords)
    {
        int maskId = _options.ResolveHotwordMaskTokenId();
        int rows = Math.Max(1, hotwords.Count);
        int maxLen = Math.Max(1, _options.HotwordMaxLength);

        var ids = new Tensor<T>(new[] { rows, maxLen });
        for (int h = 0; h < rows; h++)
        {
            string word = h < hotwords.Count ? hotwords[h] ?? string.Empty : string.Empty;
            for (int c = 0; c < maxLen; c++)
            {
                int id = c < word.Length ? word[c] % _options.VocabSize : maskId;
                ids.Data.Span[(h * maxLen) + c] = NumOps.FromDouble(id);
            }
        }

        return ids;
    }

    /// <summary>
    /// Samples hotword token ids out of the TRAINING TARGET, as SeACo does during training.
    /// </summary>
    /// <remarks>
    /// §3.1: "n hotwords are randomly sampled out of batches of y_1:L". §4.2 confirms the model "is
    /// trained with ASR backbone freezing" on those sampled hotwords. Sampling from the target is what
    /// keeps the bias branch active — and therefore receiving gradient — during every training step,
    /// while inference stays governed by Eq 5 and the caller's list.
    /// </remarks>
    private Tensor<T> SampleHotwordIdsFromTarget(Tensor<T> expected, Func<int, bool> isHotwordPosition)
    {
        int cols = expected.Rank > 1 ? expected.Shape[^1] : expected.Length;
        int positions = expected.Length / Math.Max(1, cols);
        int maxLen = Math.Max(1, _options.HotwordMaxLength);
        int maskId = _options.ResolveHotwordMaskTokenId();

        var sampled = new List<int>();
        for (int r = 0; r < positions && sampled.Count < maxLen; r++)
        {
            if (!isHotwordPosition(r)) continue;
            // Take the argmax class at this position as the hotword's token.
            int argmax = 0;
            double best = double.NegativeInfinity;
            for (int c = 0; c < cols; c++)
            {
                double v = NumOps.ToDouble(expected.Data.Span[(r * cols) + c]);
                if (v > best) { best = v; argmax = c; }
            }

            sampled.Add(argmax % _options.VocabSize);
        }

        var ids = new Tensor<T>(new[] { 1, maxLen });
        for (int c = 0; c < maxLen; c++)
        {
            ids.Data.Span[c] = NumOps.FromDouble(c < sampled.Count ? sampled[c] : maskId);
        }

        return ids;
    }

    /// <summary>
    /// Selects which output positions count as hotword positions for this step, following SeACo's
    /// random sampling strategy.
    /// </summary>
    /// <param name="expected">The target tensor whose leading axis indexes output positions.</param>
    /// <returns>A predicate over position index; true where a sampled hotword sits.</returns>
    /// <remarks>
    /// <para>
    /// SeACo §3 controls sampling with four hyper-parameters: r_b (ratio of batches to sample at all —
    /// "the forward of the other batches will be conducted with a default hotword &lt;blank&gt;"), r_u
    /// (utterance-level ratio inside an active batch, giving "r_u x bs + 1" hotwords on average, the
    /// +1 being the default hotword), and l_min / l_max bounding sampled hotword length. They are
    /// exposed as <see cref="SeACoOptions.HotwordBatchRatio"/>,
    /// <see cref="SeACoOptions.HotwordUtteranceRatio"/>, <see cref="SeACoOptions.HotwordMinLength"/> and
    /// <see cref="SeACoOptions.HotwordMaxLength"/>.
    /// </para>
    /// <para>
    /// Deterministically seeded so a training step is reproducible: an unseeded draw here would make
    /// the objective itself vary run to run, which is the flaky-RNG failure mode this repository has
    /// already been bitten by.
    /// </para>
    /// </remarks>
    private Func<int, bool> SampleHotwordPositions(Tensor<T> expected)
        => CreateHotwordPositions(expected, _hotwordRng);

    /// <summary>Applies SeACo's r_b/r_u/span sampling recipe using the supplied generator.</summary>
    private Func<int, bool> CreateHotwordPositions(Tensor<T> expected, Random rng)
    {
        int cols = expected.Rank > 1 ? expected.Shape[^1] : expected.Length;
        int positions = expected.Length / Math.Max(1, cols);

        // r_b: this batch may be inactive entirely, in which case the default <blank> hotword applies
        // and NO position is treated as a hotword.
        //
        // ONE GENERATOR PER MODEL, NOT ONE PER CALL. Constructing a fresh seeded generator here meant
        // every training step replayed the identical draw sequence: the r_b gate below always compared
        // the SAME first value against HotwordBatchRatio, so it was not a ratio at all -- it either
        // admitted every step or none of them for the whole run -- and every admitted step then selected
        // the identical spans. SeACo's criterion depends on the sampling varying across steps; frozen,
        // the bias branch sees one fixed masking pattern and the ratio options do nothing.
        if (rng.NextDouble() >= _options.HotwordBatchRatio)
        {
            return _ => false;
        }

        // r_u: average of r_u * bs + 1 sampled spans, each of length in [l_min, l_max].
        int spans = Math.Max(1, (int)Math.Round((_options.HotwordUtteranceRatio * positions) + 1.0));
        int lmin = Math.Max(1, _options.HotwordMinLength);
        int lmax = Math.Max(lmin, _options.HotwordMaxLength);

        var isHotword = new bool[positions];
        for (int s = 0; s < spans; s++)
        {
            int len = rng.Next(lmin, lmax + 1);
            if (len >= positions)
            {
                for (int i = 0; i < positions; i++) isHotword[i] = true;
                break;
            }

            int start = rng.Next(0, positions - len + 1);
            for (int i = start; i < start + len; i++) isHotword[i] = true;
        }

        return i => i >= 0 && i < positions && isHotword[i];
    }

    /// <summary>
    /// The default hotword-sampling seed, used when the caller sets no <see cref="ModelOptions.Seed"/>.
    /// </summary>
    /// <remarks>
    /// Seeding is still the default so a run is reproducible end to end. What changed is the SCOPE: the
    /// generator advances across the whole run instead of being rebuilt per step, so successive steps
    /// draw different spans while the sequence as a whole stays deterministic.
    /// </remarks>
    private const int HotwordSamplingSeed = 1337;

    /// <summary>
    /// The hotword sampler's generator, created once and advanced across training steps.
    /// </summary>
    private readonly Random _hotwordRng;

    /// <summary>
    /// Applies SeACo's hotword-position-aware criterion to a label tensor.
    /// </summary>
    /// <param name="labels">Target labels, one row per output position.</param>
    /// <param name="isHotwordPosition">Predicate marking which positions belong to a sampled hotword.</param>
    /// <returns>A copy whose non-hotword rows are replaced by <see cref="SeACoOptions.HotwordMaskTokenId"/>.</returns>
    /// <remarks>
    /// SeACo §3: the bias parameters "can be updated with the hotword-position-aware criterion in which
    /// labels in non-hotword positions are replaced by #". Masking is what keeps the bias branch focused
    /// on the hotword list instead of re-learning the whole transcription, and it is the reason an
    /// UNMASKED full-sequence CE (what this model used before) produces a loss surface the bias
    /// parameters cannot descend.
    /// </remarks>
    internal Tensor<T> CreateBiasTrainingTarget(Tensor<T> labels, Func<int, bool> isHotwordPosition)
    {
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (isHotwordPosition is null) throw new ArgumentNullException(nameof(isHotwordPosition));

        int sourceCols = labels.Rank > 1 ? labels.Shape[^1] : labels.Length;
        int rows = labels.Length / Math.Max(1, sourceCols);
        int biasClasses = _options.VocabSize + 1;
        var targetShape = labels.Shape.ToArray();
        if (targetShape.Length == 0)
            targetShape = new[] { biasClasses };
        else
            targetShape[^1] = biasClasses;
        var target = new Tensor<T>(targetShape);

        for (int r = 0; r < rows; r++)
        {
            int targetClass = isHotwordPosition(r)
                ? Math.Min(ArgMaxRow(labels, r, sourceCols), _options.VocabSize - 1)
                : _options.ResolveHotwordMaskTokenId();
            target.Data.Span[(r * biasClasses) + targetClass] = NumOps.One;
        }

        return target;
    }
    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "SeACo-Native" : "SeACo-ONNX", Description = "SeACo-Paraformer: hot-word biased CIF ASR (Alibaba, 2023)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };



    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames && tokens.Count < _options.MaxTextLength; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } }; }
    private string ClassifyLanguageFromTokens(List<int> tokens) { if (tokens.Count == 0) return _options.Language; int cjkCount = 0, latinCount = 0; foreach (var t in tokens) { if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++; else if (t >= 0x41 && t <= 0x7A) latinCount++; } if (cjkCount > latinCount && SupportedLanguages.Contains("zh")) return "zh"; return _options.Language; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SeACo<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
