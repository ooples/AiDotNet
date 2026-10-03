using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// AdaSpeech: adaptive TTS for custom voice with acoustic condition modeling and conditional layer normalization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "AdaSpeech: Adaptive Text to Speech for Custom Voice" (Chen et al., ICLR 2021).</para>
/// <para>
/// The backbone is FastSpeech 2 (§2): a phoneme encoder of 4 FFT blocks, the variance adaptor (duration, pitch,
/// energy), and a mel decoder of 4 FFT blocks, hidden 256. Two additions make it adaptable:
/// </para>
/// <list type="bullet">
/// <item><b>Acoustic condition modeling</b> (§2.1): a speaker embedding, an utterance-level vector from a reference
/// speech (two convolutions of kernel 5 and stride 3 and mean pooling) and 4-dimensional phoneme-level vectors (from the
/// target frames averaged per phoneme in training, from a predictor on the phoneme encoder output at inference) are
/// added to the phoneme hidden sequence before the variance adaptor.</item>
/// <item><b>Conditional layer normalization</b> (§2.2): every layer normalization in the mel decoder, plus a final
/// one, takes its scale and bias from the speaker embedding through two linear maps
/// (<see cref="ConditionalLayerNormalizationLayer{T}"/>).</item>
/// </list>
/// <para>
/// Training follows §3 through <see cref="CurrentPhase"/>: source pre-training (everything but the phoneme-level
/// predictor), joint training with the predictor's MSE to the stop-gradient phoneme-level encoder output, and
/// adaptation to a new voice, which updates only the speaker embedding and the conditional layer normalizations'
/// <c>W_γ</c>, <c>W_β</c>. The loss is FastSpeech 2's: mel MAE plus the variance predictors' MSE.
/// </para>
/// <para>
/// Training data: durations from a forced aligner (MFA in the paper) and the speaker's index, so a plain
/// <c>Train(tokens, mel)</c> throws; pass a <see cref="TtsTrainingSample{T}"/> with
/// <see cref="TtsTrainingSample{T}.Durations"/> and <see cref="TtsTrainingSample{T}.SpeakerId"/>. Synthesis reads
/// <see cref="TtsModelBase{T}.Voice"/>: the speaker and a reference mel spectrogram of them.
/// </para>
/// <para><b>For Beginners:</b> AdaSpeech is a FastSpeech 2 that can learn a new voice from a few minutes of speech by
/// adjusting only a few thousand numbers per speaker.</para>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 200, outputSize: 80);
/// var model = new AdaSpeech&lt;double&gt;(architecture, new AdaSpeechOptions());
/// model.Train(new TtsTrainingSample&lt;double&gt; { Tokens = tokens, Mel = mel, Durations = durations, SpeakerId = 3 });
/// model.Voice = new TtsVoice&lt;double&gt; { SpeakerId = 3, Reference = referenceMel };
/// var spectrogram = model.Synthesize("hello world");
/// </code>
/// </example>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "AdaSpeech: Adaptive Text to Speech for Custom Voice",
    "https://arxiv.org/abs/2103.00993",
    Year = 2021,
    Authors = "Chen et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-9,
                Source = "Chen et al. 2021, Sec. 3: the Adam optimizer with beta1 0.9, beta2 0.98 and "
                        + "epsilon 1e-9. The paper states no learning rate in its training description, "
                        + "so none is declared.")]
public partial class AdaSpeech<T> : VarianceAdaptorTtsModelBase<T>, IAcousticModel<T>
{
    private readonly AdaSpeechOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<object, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _phaseOptimizers = new();
    private bool _useNativeMode;

    private EmbeddingLayer<T>? _speakerEmbedding;
    private AcousticConditionEncoderLayer<T>? _utteranceEncoder;
    private AcousticConditionEncoderLayer<T>? _phonemeEncoder;
    private AcousticConditionEncoderLayer<T>? _phonemePredictor;
    private DenseLayer<T>? _phonemeProjection;

    public override ModelOptions GetOptions() => _options;

    /// <summary>The speaker embedding table (null when the architecture supplies its own layers).</summary>
    internal EmbeddingLayer<T>? SpeakerEmbeddingTable => _speakerEmbedding;

    /// <summary>The phoneme-level acoustic predictor (Fig. 2d).</summary>
    internal AcousticConditionEncoderLayer<T>? PhonemePredictor => _phonemePredictor;

    /// <summary>The model's options.</summary>
    private protected AdaSpeechOptions AdaOptions => _options;

    /// <summary>The utterance-level acoustic encoder (Fig. 2b).</summary>
    private protected AcousticConditionEncoderLayer<T> UtteranceEncoder =>
        _utteranceEncoder ?? throw new InvalidOperationException("The model was built from caller-supplied layers.");

    /// <summary>Which optimizer a training step uses: each phase trains its own parameter set, so each gets its own.</summary>
    private protected virtual object TrainingPhaseKey => CurrentPhase;

    public AdaSpeech(NeuralNetworkArchitecture<T> architecture, string modelPath, AdaSpeechOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new AdaSpeechOptions();
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

    /// <summary>Creates a trainable AdaSpeech.</summary>
    /// <param name="architecture">The network architecture; supply layers only to replace the paper's stack.</param>
    /// <param name="options">Model options; defaults are the paper's.</param>
    /// <param name="optimizer">An optimizer used for every training phase; by default each phase gets its own
    /// paper-configured Adam, since each phase trains a different parameter set.</param>
    public AdaSpeech(
        NeuralNetworkArchitecture<T> architecture,
        AdaSpeechOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new AdaSpeechOptions();
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>
    /// The training phase (§3): source-model pre-training without the phoneme-level predictor, then joint training
    /// with it, then per-voice adaptation of the conditional layer normalizations and the speaker embedding.
    /// </summary>
    public AdaSpeechTrainingPhase CurrentPhase { get; set; } = AdaSpeechTrainingPhase.SourcePretraining;

    /// <summary>Generates a mel spectrogram from text in <see cref="TtsModelBase{T}.Voice"/> (pair it with a vocoder;
    /// the paper uses MelGAN).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <inheritdoc />
    /// <remarks>The FastSpeech 2 backbone scores the mel spectrogram with mean absolute error (Ren et al. 2021, §3.1).</remarks>
    protected override bool UsesL1MelLoss => true;

    /// <inheritdoc />
    /// <remarks>Forced-alignment durations (MFA, §3) and the speaker's index in the speaker embedding table.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Durations | TtsSupervision.SpeakerId;

    /// <inheritdoc />
    /// <remarks>Inference reads the speaker embedding and "the utterance-level acoustic conditions ... extracted from
    /// another reference speech of the speaker" (§3).</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.SpeakerId | TtsSupervision.SpeakerReference;

    /// <inheritdoc />
    protected override bool HasAcousticConditioning => _speakerEmbedding is not null;

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer
    {
        get
        {
            if (_suppliedOptimizer is not null)
                return _suppliedOptimizer;
            if (!_phaseOptimizers.TryGetValue(TrainingPhaseKey, out var optimizer))
            {
                // Adam with beta1 0.9, beta2 0.98, epsilon 1e-9 (§3). The paper states no learning rate, so the
                // options' rate (1e-4 by default, the fine-tuning rate of its FastSpeech 2 lineage) stands.
                optimizer = PaperOptimizerFactory.VerifyHandBuilt(this,
                    new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                        new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                        {
                            InitialLearningRate = _options.LearningRate,
                            Beta1 = 0.9,
                            Beta2 = 0.98,
                            Epsilon = 1e-9
                        }));
                _phaseOptimizers[TrainingPhaseKey] = optimizer;
            }
            return optimizer;
        }
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        int hidden = _options.HiddenDim;
        AddEncoderDecoderLayers(
            LayerHelper<T>.CreateDefaultFastSpeechEncoderLayers(
                _options.VocabSize, _options.EncoderDim, hidden, _options.NumEncoderLayers, _options.NumHeads,
                _options.FftFilterSize, _options.FftKernelSizes[0], _options.FftKernelSizes[1], _options.DropoutRate,
                _options.MaxTextLength),
            LayerHelper<T>.CreateDefaultAdaSpeechDecoderLayers(
                hidden, hidden, _options.MelChannels, _options.NumDecoderLayers, _options.NumHeads,
                _options.FftFilterSize, _options.FftKernelSizes[0], _options.FftKernelSizes[1], _options.DropoutRate,
                _options.MaxMelLength, _options.VariancePredictorFilterSize, _options.VariancePredictorKernelSize,
                _options.VariancePredictorDropout, _options.NumPitchBins, _options.PitchMinHz, _options.PitchMaxHz,
                _options.NumEnergyBins, _options.FftSize, _options.UsePitchPredictor, _options.UseEnergyPredictor));

        // Acoustic condition modeling (§2.1, Fig. 2): speaker embedding (hidden-wide), utterance-level encoder
        // (kernel 5, stride 3, mean pooling), phoneme-level encoder and predictor (kernel 3, stride 1, linear to 4).
        // The 4-dimensional phoneme-level vectors are "added element-wisely into the hidden sequence" (Fig. 2a), which
        // needs a projection back to the hidden width; the paper does not name one, so a linear layer does it.
        _speakerEmbedding = new EmbeddingLayer<T>(_options.NumSpeakers, hidden);
        _utteranceEncoder = new AcousticConditionEncoderLayer<T>(_options.MelChannels, hidden,
            _options.UtteranceEncoderKernelSize, _options.UtteranceEncoderStride, 0, meanPool: true,
            _options.AcousticConditionDropout);
        _phonemeEncoder = new AcousticConditionEncoderLayer<T>(_options.MelChannels, _options.AcousticConditionFilterSize,
            _options.PhonemeEncoderKernelSize, 1, _options.PhonemeConditionDim, meanPool: false,
            _options.AcousticConditionDropout);
        _phonemePredictor = new AcousticConditionEncoderLayer<T>(hidden, _options.AcousticConditionFilterSize,
            _options.PhonemeEncoderKernelSize, 1, _options.PhonemeConditionDim, meanPool: false,
            _options.AcousticConditionDropout);
        _phonemeProjection = new DenseLayer<T>(hidden, new IdentityActivation<T>() as IActivationFunction<T>);
        ComponentLayers.Add(_speakerEmbedding);
        ComponentLayers.Add(_utteranceEncoder);
        ComponentLayers.Add(_phonemeEncoder);
        ComponentLayers.Add(_phonemePredictor);
        ComponentLayers.Add(_phonemeProjection);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Training conditions come from the utterance itself (§2.1): the utterance-level vector from the target speech,
    /// the phoneme-level vectors from its frames averaged per phoneme by the alignment. In joint training the
    /// phoneme-level predictor learns, with MSE, to reproduce the phoneme-level encoder's output, with the gradient
    /// stopped at that label (§3).
    /// </remarks>
    protected override AcousticConditioning<T> ConditionForTraining(
        Tensor<T> hidden, TtsTrainingSample<T> sample, Tensor<T> mel, int[] durations)
    {
        int speaker = sample.SpeakerId ?? throw new ArgumentException(
            $"{nameof(AdaSpeech<T>)} trains a speaker table; set {nameof(sample.SpeakerId)}.", nameof(sample));
        var speakerEmbedding = SpeakerEmbedding(speaker);
        var utterance = _utteranceEncoder!.Forward(mel);
        var phonemeVectors = _phonemeEncoder!.Forward(PhonemeLevelMel(mel, durations));

        var conditioned = AddConditions(hidden, speakerEmbedding, utterance, phonemeVectors);

        Tensor<T>? predictorLoss = null;
        if (CurrentPhase == AdaSpeechTrainingPhase.SourceJoint)
        {
            // Stop-gradient label: a detached copy of the phoneme-level encoder output.
            var label = new Tensor<T>(phonemeVectors._shape, phonemeVectors.ToVector());
            predictorLoss = MeanSquaredError(_phonemePredictor!.Forward(hidden), label);
        }
        return new AcousticConditioning<T>(conditioned, speakerEmbedding, predictorLoss);
    }

    /// <inheritdoc />
    /// <remarks>Inference conditions (§3): the speaker embedding and the utterance-level vector of the voice's
    /// reference speech, and phoneme-level vectors from the predictor.</remarks>
    protected override AcousticConditioning<T> ConditionForInference(Tensor<T> hidden)
    {
        var voice = RequireVoice();
        var reference = voice.Reference!;
        if (reference.Rank != 2 || reference.Shape[1] != _options.MelChannels)
            throw new ArgumentException(
                $"The voice's reference must be a mel spectrogram [frames, {_options.MelChannels}], " +
                $"got [{string.Join(", ", reference.Shape)}].");
        var speakerEmbedding = SpeakerEmbedding(voice.SpeakerId);
        var utterance = _utteranceEncoder!.Forward(reference);
        var phonemeVectors = _phonemePredictor!.Forward(hidden);
        var conditioned = AddConditions(hidden, speakerEmbedding, utterance, phonemeVectors);

        var decoderCondition = hidden.Rank == 3
            ? Engine.TensorTile(Engine.Reshape(speakerEmbedding, new[] { 1, speakerEmbedding.Length }), new[] { hidden.Shape[0], 1 })
            : speakerEmbedding;
        return new AcousticConditioning<T>(conditioned, decoderCondition, null);
    }

    /// <summary>
    /// Adds the speaker embedding and the utterance-level vector (expanded over the sequence) and the projected
    /// phoneme-level vectors (element-wise) to the phoneme hidden sequence (Fig. 2a).
    /// </summary>
    private Tensor<T> AddConditions(Tensor<T> hidden, Tensor<T> speakerEmbedding, Tensor<T> utterance, Tensor<T> phonemeVectors)
    {
        var x = Engine.TensorAdd(hidden, Expand(speakerEmbedding, hidden));
        x = Engine.TensorAdd(x, Expand(utterance, hidden));
        return Engine.TensorAdd(x, Engine.Reshape(_phonemeProjection!.Forward(phonemeVectors), hidden._shape));
    }

    /// <summary>Repeats a <c>[hidden]</c> vector over every leading axis of <paramref name="like"/>.</summary>
    private protected Tensor<T> Expand(Tensor<T> vector, Tensor<T> like)
    {
        var shape = new int[like.Rank];
        var multiples = new int[like.Rank];
        for (int i = 0; i < like.Rank; i++)
        {
            shape[i] = i == like.Rank - 1 ? vector.Length : 1;
            multiples[i] = i == like.Rank - 1 ? 1 : like.Shape[i];
        }
        return Engine.TensorTile(Engine.Reshape(vector, shape), multiples);
    }

    /// <summary>The speaker's row of the speaker embedding table, <c>[hidden]</c>.</summary>
    private protected Tensor<T> SpeakerEmbedding(int speaker)
    {
        if (speaker < 0 || speaker >= _options.NumSpeakers)
            throw new ArgumentOutOfRangeException(nameof(speaker),
                $"Speaker {speaker} is outside the speaker table (0..{_options.NumSpeakers - 1}).");
        var id = new Tensor<T>(new[] { 1 });
        id[0] = NumOps.FromDouble(speaker);
        var row = _speakerEmbedding!.Forward(id);
        return Engine.Reshape(row, new[] { _options.HiddenDim });
    }

    /// <summary>
    /// The phoneme-level mel (Fig. 2c): the frames aligned to each phoneme averaged, <c>[phonemes, melChannels]</c>.
    /// A phoneme with no frames gets a zero row.
    /// </summary>
    private Tensor<T> PhonemeLevelMel(Tensor<T> mel, int[] durations)
    {
        int frames = mel.Shape[0];
        var average = new Tensor<T>(new[] { durations.Length, frames });
        int start = 0;
        for (int p = 0; p < durations.Length; p++)
        {
            for (int f = start; f < start + durations[p]; f++)
                average[p, f] = NumOps.FromDouble(1.0 / durations[p]);
            start += durations[p];
        }
        return Engine.TensorMatMul(average, Engine.Reshape(mel, new[] { frames, mel.Length / frames }));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Each phase trains what the paper trains in it (§2.3, §3): pre-training, every parameter except the
    /// phoneme-level predictor's; joint training, every parameter; adaptation, only the speaker embedding and the
    /// two matrices <c>W_γ</c>, <c>W_β</c> of every conditional layer normalization in the decoder.
    /// </remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (_speakerEmbedding is null)
            return parameters;
        IEnumerable<Tensor<T>> selected;
        switch (CurrentPhase)
        {
            case AdaSpeechTrainingPhase.SourcePretraining:
            {
                var predictor = new HashSet<Tensor<T>>(
                    Training.TapeTrainingStep<T>.CollectParameters(new ILayer<T>[] { _phonemePredictor! }, -1),
                    Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                return parameters.Where(p => !predictor.Contains(p)).ToList();
            }
            case AdaSpeechTrainingPhase.SourceJoint:
                return parameters;
            default:
                selected = Training.TapeTrainingStep<T>.CollectParameters(AdaptationLayers(), -1);
                break;
        }
        var keep = new HashSet<Tensor<T>>(selected, Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(keep.Contains).ToList();
    }

    /// <summary>What voice adaptation trains (§2.3): the speaker embedding and every conditional layer normalization's
    /// <c>W_γ</c> and <c>W_β</c> in the decoder.</summary>
    private protected List<ILayer<T>> AdaptationLayers()
    {
        var adaptive = new List<ILayer<T>> { _speakerEmbedding! };
        foreach (var layer in Layers.Skip(EncoderLayerCount))
            CollectConditionalProjections(layer, adaptive);
        return adaptive;
    }

    private static void CollectConditionalProjections(ILayer<T> layer, List<ILayer<T>> into)
    {
        if (layer is ConditionalLayerNormalizationLayer<T> conditional)
        {
            into.Add(conditional.ScaleProjection);
            into.Add(conditional.BiasProjection);
            return;
        }
        if (layer is LayerBase<T> composite)
            foreach (var child in composite.GetSubLayers())
                CollectConditionalProjections(child, into);
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "AdaSpeech-Native" : "AdaSpeech-ONNX",
            Description = "AdaSpeech: Adaptive TTS for Custom Voice (Chen et al., 2021)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "AdaSpeech";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }
}

/// <summary>The training phases of AdaSpeech (Chen et al. 2021, §2.3, §3), run in order.</summary>
public enum AdaSpeechTrainingPhase
{
    /// <summary>Source-model training of every parameter except the phoneme-level acoustic predictor (60k steps).</summary>
    SourcePretraining = 1,

    /// <summary>Source-model training together with the phoneme-level acoustic predictor, whose MSE to the
    /// stop-gradient phoneme-level encoder output joins the loss (40k steps).</summary>
    SourceJoint = 2,

    /// <summary>Per-voice adaptation: only the speaker embedding and the conditional layer normalizations' W_γ and W_β
    /// (2000 steps).</summary>
    Adaptation = 3,
}
