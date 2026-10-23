using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.DescriptionBased;

/// <summary>
/// PromptTTS: speech whose style a plain-language prompt describes (Guo et al. 2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The paper's model (§2): a <b>style encoder</b>, BERT-base whose [CLS] state is the style representation, fine-tuned
/// with P-tuning v2 on an auxiliary task that predicts the prompt's gender, pitch, speaking speed, volume and emotion
/// classes; a <b>content encoder</b>, FastSpeech 2's phoneme encoder of 4 Transformer blocks topped by its variance
/// adaptor (duration, pitch, energy), with the style representation prepended to the input of every block; and a
/// <b>speech decoder</b> of the same blocks, with the style prepended to each block's input, that produces the mel
/// spectrogram (a HiFi-GAN vocoder turns it into audio). Style and content representations are 256 wide (§4.1), so a
/// linear map takes BERT's 768-wide [CLS] state to 256.
/// </para>
/// <para>
/// Training is in two stages (<see cref="CurrentStage"/>): <see cref="PromptTTSTrainingStage.StyleClassification"/>
/// tunes the style encoder on the style-factor labels (<see cref="TrainStyle"/>), then
/// <see cref="PromptTTSTrainingStage.Synthesis"/> trains the rest on FastSpeech 2's objective from samples carrying the
/// phonemes, the forced-alignment durations and the style prompt (<see cref="TtsTrainingSample{T}.Description"/>).
/// </para>
/// <para><b>For Beginners:</b> You describe how the speech should sound ("A lady whispers to her friend slowly") and give
/// the text to speak; the model produces a mel spectrogram in that style.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "PromptTTS: Controllable Text-to-Speech with Text Descriptions",
    "https://arxiv.org/abs/2211.12171",
    Year = 2023,
    Authors = "Guo, Leng, Wu, Zhao and Tan"
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-9,
    Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 4000, ReferenceBatchSize = 48,
    Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "PromptTTS states no optimizer; its content encoder, variance adaptor and decoder follow FastSpeech 2 "
             + "(Guo et al. 2023, §2.3, §4.1), which trains with Adam (0.9, 0.98, eps 1e-9) on the Transformer's "
             + "schedule with 4000 warm-up steps (Ren et al. 2021, §4.1).")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.0,
    MaxGradientNorm = 1.0, Component = "style-encoder", Provenance = RecipeProvenance.NamedReproduction,
    Source = "The paper fine-tunes BERT with P-tuning v2 (§4.1) and states no settings; THUDM/P-tuning-v2 trains with the "
             + "Hugging Face Trainer's defaults (AdamW at 5e-5, betas 0.9 and 0.999, eps 1e-8, no weight decay, "
             + "gradient norm 1, linear decay to zero).")]
public partial class PromptTTS<T> : VarianceAdaptorTtsModelBase<T>, IAcousticModel<T>
{
    private readonly PromptTTSOptions _options;
    private readonly bool _useNativeMode;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _styleOptimizer;
    private readonly Random _trainingRandom;
    private readonly WordPieceTokenizer _styleTokenizer = WordPieceTokenizer.BertBaseUncased();
    private bool _disposed;

    private BertNetwork<T>? _bert;
    private readonly List<LayerBase<T>> _bertLayers = new();
    private readonly List<LayerBase<T>> _prefixLayers = new();
    private readonly List<LayerBase<T>> _classifierLayers = new();
    private TiedEmbeddingLayer<T>? _prefixEmbedding;
    private DenseLayer<T>? _prefixHidden, _prefixOutput;
    private DenseLayer<T>? _pooler;
    private readonly List<DenseLayer<T>> _factorHeads = new();
    private DenseLayer<T>? _styleProjection;
    private bool _styleStep;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed PromptTTS for inference.</summary>
    public PromptTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, PromptTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new PromptTTSOptions();
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable PromptTTS.</summary>
    public PromptTTS(NeuralNetworkArchitecture<T> architecture, PromptTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new PromptTTSOptions();
        _useNativeMode = true;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        if (o.EncoderDim != o.HiddenDim)
            throw new ArgumentException("PromptTTS's style and content representations share one width (256, §4.1); set EncoderDim = HiddenDim.");
        if (o.StyleFactorClasses is null || o.StyleFactorClasses.Length == 0 || o.StyleFactorClasses.Any(c => c < 2))
            throw new ArgumentException("Each style factor needs at least two classes.");
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public new int MelChannels => _options.MelChannels;

    /// <inheritdoc />
    public new int HopSize => _options.HopSize;

    /// <inheritdoc />
    public int FftSize => _options.FftSize;

    /// <summary>Generates a mel spectrogram from text in the voice's style (pair it with a vocoder).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <summary>The network <see cref="TtsModelBase{T}.Train(TtsTrainingSample{T})"/> updates; the style encoder trains
    /// through <see cref="TrainStyle"/>.</summary>
    public PromptTTSTrainingStage CurrentStage => _styleStep ? PromptTTSTrainingStage.StyleClassification : PromptTTSTrainingStage.Synthesis;

    /// <inheritdoc />
    /// <remarks>FastSpeech 2's mel loss is the mean absolute error (Ren et al. 2021, §3.1).</remarks>
    protected override bool UsesL1MelLoss => true;

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <inheritdoc />
    /// <remarks>FastSpeech 2's forced-alignment durations, and the style prompt.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Durations | TtsSupervision.Description;

    /// <inheritdoc />
    protected override TtsSupervision RequiredVoice => TtsSupervision.Description;

    /// <inheritdoc />
    protected override bool HasAcousticConditioning => _bert is not null;

    private BertNetwork<T> StyleEncoder => _bert ?? throw new InvalidOperationException("PromptTTS's style encoder exists in native mode only.");

    // The networks' layers, for tests.
    internal IReadOnlyList<LayerBase<T>> StyleEncoderLayers => _bertLayers;
    internal IReadOnlyList<LayerBase<T>> PrefixLayers => _prefixLayers;
    internal IReadOnlyList<LayerBase<T>> ClassifierLayers => _classifierLayers;

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        int d = o.HiddenDim, k1 = o.FftKernelSizes[0], k2 = o.FftKernelSizes[1];
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        // Content encoder: FastSpeech 2's phoneme embedding, positions and blocks, each block reading the style first.
        var encoder = new List<ILayer<T>>
        {
            new EmbeddingLayer<T>(o.VocabSize, d),
            new PositionalEncodingLayer<T>(o.MaxTextLength, d),
        };
        for (int i = 0; i < o.NumEncoderLayers; i++)
            encoder.Add(new FeedForwardTransformerBlock<T>(d, o.NumHeads, o.FftFilterSize, k1, k2, o.DropoutRate, prependCondition: true));
        // Variance adaptor, then the speech decoder of the same blocks.
        var decoder = new List<ILayer<T>>
        {
            new VarianceAdaptorLayer<T>(d, o.VariancePredictorFilterSize, o.VariancePredictorKernelSize, o.VariancePredictorDropout,
                o.NumPitchBins, o.PitchMinHz, o.PitchMaxHz, o.NumEnergyBins, energyMin: 0.0,
                energyMax: VarianceAdaptorLayer<T>.MaxStftEnergy(o.FftSize), usePitch: o.UsePitchPredictor, useEnergy: o.UseEnergyPredictor),
            new PositionalEncodingLayer<T>(o.MaxMelLength, d),
        };
        for (int i = 0; i < o.NumDecoderLayers; i++)
            decoder.Add(new FeedForwardTransformerBlock<T>(d, o.NumHeads, o.FftFilterSize, k1, k2, o.DropoutRate, prependCondition: true));
        decoder.Add(new DenseLayer<T>(o.MelChannels, identity));
        AddEncoderDecoderLayers(encoder, decoder);

        // Style encoder: BERT-base, P-tuning v2's prefixes, the pooler and one classifier per style factor.
        _bert = new BertNetwork<T>(Engine, _bertLayers, new BertConfiguration
        {
            VocabSize = o.StyleVocabularySize, Hidden = o.StyleHiddenDim, Layers = o.StyleLayers, Heads = o.StyleHeads,
            Intermediate = o.StyleIntermediateDim, MaxPositions = o.StyleMaxPositions, Dropout = o.StyleDropout,
        });
        _bert.InitializeLikeReference(random);
        int h = o.StyleHiddenDim, prefixWidth = o.StyleLayers * 2 * h;
        void Normal(DenseLayer<T> layer, int inputs, int outputs) =>
            TorchParameters.Linear(Engine, layer, inputs, outputs, TorchParameters.Gaussian<double>(random, inputs * outputs, 0.02).ToArray(),
                new double[outputs]);
        if (o.PrefixProjection)
        {
            _prefixEmbedding = new TiedEmbeddingLayer<T>(o.PrefixLength, h);
            _prefixHidden = new DenseLayer<T>(o.PrefixHiddenSize, (IActivationFunction<T>)new TanhActivation<T>());
            _prefixOutput = new DenseLayer<T>(prefixWidth, identity);
            _prefixLayers.AddRange(new LayerBase<T>[] { _prefixEmbedding, _prefixHidden, _prefixOutput });
            Normal(_prefixHidden, h, o.PrefixHiddenSize);
            Normal(_prefixOutput, o.PrefixHiddenSize, prefixWidth);
        }
        else
        {
            _prefixEmbedding = new TiedEmbeddingLayer<T>(o.PrefixLength, prefixWidth);
            _prefixLayers.Add(_prefixEmbedding);
        }
        _prefixEmbedding.Reinitialize(() => 0.02 * StandardNormal(random));
        _pooler = new DenseLayer<T>(h, (IActivationFunction<T>)new TanhActivation<T>());
        Normal(_pooler, h, h);
        _classifierLayers.Add(_pooler);
        foreach (int classes in o.StyleFactorClasses)
        {
            var head = new DenseLayer<T>(classes, identity);
            Normal(head, h, classes);
            _factorHeads.Add(head);
            _classifierLayers.Add(head);
        }
        _styleProjection = new DenseLayer<T>(d, identity);
        using (new NoGradScope<T>()) _styleProjection.Forward(new Tensor<T>(new[] { 1, h }));
        ComponentLayers.AddRange(_bertLayers);
        ComponentLayers.AddRange(_prefixLayers);
        ComponentLayers.AddRange(_classifierLayers);
        ComponentLayers.Add(_styleProjection);
    }

    private static double StandardNormal(Random random) =>
        Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());

    // ---------------------------------------------------------------- style prompt

    /// <summary>The style prompt's bert-base-uncased token ids, <c>[CLS] … [SEP]</c>.</summary>
    public int[] TokenizeStyle(string stylePrompt) =>
        _styleTokenizer.Encode(stylePrompt ?? throw new ArgumentNullException(nameof(stylePrompt))).TokenIds.ToArray();

    /// <summary>A voice for <see cref="TtsModelBase{T}.Voice"/>: the style prompt ("A lady whispers to her friend
    /// slowly").</summary>
    public TtsVoice<T> CreateVoice(string stylePrompt)
    {
        if (string.IsNullOrWhiteSpace(stylePrompt)) throw new ArgumentException("Describe the style.", nameof(stylePrompt));
        var ids = TokenizeStyle(stylePrompt);
        var tensor = new Tensor<T>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return new TtsVoice<T> { Description = tensor };
    }

    private int[] StyleIds(Tensor<T> prompt, string name)
    {
        if (prompt.Length == 0) throw new ArgumentException("The style prompt is empty.", name);
        var ids = new int[prompt.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(NumOps.ToDouble(prompt[i]));
            if (ids[i] < 0 || ids[i] >= _options.StyleVocabularySize)
                throw new ArgumentOutOfRangeException(name, $"Style prompt ids lie in [0, {_options.StyleVocabularySize}).");
        }
        return ids;
    }

    /// <summary>P-tuning v2's key and value prefixes of every layer (<c>PrefixEncoder</c> and <c>get_prompt</c>):
    /// the prefix table (or its MLP), dropout, split into each layer's key and value.</summary>
    private IReadOnlyList<(Tensor<T> Key, Tensor<T> Value)>? Prefixes(bool training, Random random)
    {
        if (_options.StyleFineTuning != PromptStyleFineTuning.PTuningV2) return null;
        var o = _options;
        int h = o.StyleHiddenDim;
        var positions = new Tensor<T>(new[] { o.PrefixLength });
        for (int i = 0; i < o.PrefixLength; i++) positions[i] = NumOps.FromDouble(i);
        var table = _prefixEmbedding!.Forward(positions);                                                 // [P, ·]
        if (_prefixHidden is not null) table = _prefixOutput!.Forward(_prefixHidden.Forward(table));      // [P, 2·L·h]
        if (training && o.StyleDropout > 0) table = T5Seq2Seq<T>.Dropout(Engine, table, o.StyleDropout, random);
        var prefixes = new List<(Tensor<T>, Tensor<T>)>();
        for (int layer = 0; layer < o.StyleLayers; layer++)
            prefixes.Add((Engine.TensorSlice(table, new[] { 0, 2 * layer * h }, new[] { o.PrefixLength, h }),
                Engine.TensorSlice(table, new[] { 0, (2 * layer + 1) * h }, new[] { o.PrefixLength, h })));
        return prefixes;
    }

    /// <summary>BERT's last hidden state at [CLS], <c>[1, 768]</c>.</summary>
    private Tensor<T> ClsState(int[] ids, bool training, Random random)
    {
        var bert = StyleEncoder;
        var input = new int[1, ids.Length];
        for (int i = 0; i < ids.Length; i++) input[0, i] = ids[i];
        bert.Training = training;
        var hidden = bert.Encode(input, prefix: Prefixes(training, random));                              // [1, L, h]
        return Engine.Reshape(Engine.TensorSlice(hidden, new[] { 0, 0, 0 }, new[] { 1, 1, _options.StyleHiddenDim }),
            new[] { 1, _options.StyleHiddenDim });
    }

    /// <summary>The style representation <c>[1, HiddenDim]</c> of a style prompt.</summary>
    private Tensor<T> Style(int[] ids, bool training, Random random)
    {
        Tensor<T> cls;
        if (training && _options.TrainStyleEncoderWithSynthesis)
        {
            cls = ClsState(ids, training: true, random);
        }
        else
        {
            using (new NoGradScope<T>()) cls = ClsState(ids, training: false, random);
        }
        return _styleProjection!.Forward(cls);
    }

    // The content encoder with the style before every block.
    private Tensor<T> RunStyledEncoder(Tensor<T> tokens, Tensor<T> style)
    {
        var x = tokens;
        for (int i = 0; i < EncoderLayerCount; i++)
            x = Layers[i] is LayerBase<T> { RequiresMultipleInputs: true } block ? block.Forward(x, style) : Layers[i].Forward(x);
        return x;
    }

    /// <inheritdoc />
    protected override AcousticConditioning<T> EncodeForTraining(Tensor<T> tokens, TtsTrainingSample<T> sample, Tensor<T> mel, int[] durations)
    {
        var prompt = sample.Description ?? throw new ArgumentException("PromptTTS trains on a style prompt; set Description.", nameof(sample));
        var style = Style(StyleIds(prompt, nameof(sample)), IsTrainingMode, _trainingRandom);
        return new AcousticConditioning<T>(RunStyledEncoder(tokens, style), style, null);
    }

    /// <inheritdoc />
    protected override AcousticConditioning<T> EncodeForInference(Tensor<T> input)
    {
        var voice = RequireVoice();
        var style = Style(StyleIds(voice.Description!, nameof(input)), false, _trainingRandom);
        return new AcousticConditioning<T>(RunStyledEncoder(input, style), style, null);
    }

    // ---------------------------------------------------------------- style classification (§2.2)

    /// <summary>
    /// The style-factor logits of a style prompt, one row <c>[1, classes]</c> per factor: P-tuning v2's (and BERT's)
    /// sequence classifier, the pooled [CLS] state (dense and tanh) through dropout and a linear head.
    /// </summary>
    private IReadOnlyList<Tensor<T>> FactorLogits(int[] ids, bool training, Random random)
    {
        var pooled = _pooler!.Forward(ClsState(ids, training, random));
        if (training && _options.StyleDropout > 0) pooled = T5Seq2Seq<T>.Dropout(Engine, pooled, _options.StyleDropout, random);
        return _factorHeads.Select(head => head.Forward(pooled)).ToList();
    }

    private Tensor<T> StyleLoss(int[] ids, int[] labels, bool training, Random random)
    {
        if (labels.Length != _factorHeads.Count)
            throw new ArgumentException($"Expected a label for each of the {_factorHeads.Count} style factors.", nameof(labels));
        var logits = FactorLogits(ids, training, random);
        Tensor<T>? total = null;
        for (int f = 0; f < logits.Count; f++)
        {
            int classes = _options.StyleFactorClasses[f];
            if (labels[f] < 0 || labels[f] >= classes)
                throw new ArgumentOutOfRangeException(nameof(labels), $"Factor {f} has {classes} classes.");
            var selection = new Tensor<T>(new[] { 1, classes });
            selection[0, labels[f]] = NumOps.FromDouble(-1.0);
            var term = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorLogSoftmax(logits[f], axis: -1), selection), new[] { 0, 1 }, keepDims: false);
            total = total is null ? term : Engine.TensorAdd(total, term);
        }
        return total!;
    }

    /// <summary>
    /// One step of the auxiliary style-classification task (§2.2): the summed cross-entropy of each style factor's label
    /// (<see cref="PromptTTSOptions.StyleFactorClasses"/> order) given the style prompt's ids. Under P-tuning v2 only the
    /// prefixes and the classifiers learn; under standard fine-tuning BERT learns too.
    /// </summary>
    /// <returns>The step's loss.</returns>
    public T TrainStyle(Tensor<T> stylePrompt, int[] factorLabels)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (_options.StyleFineTuning == PromptStyleFineTuning.None)
            throw new InvalidOperationException("The style encoder is not fine-tuned (StyleFineTuning = None).");
        Guard.NotNull(stylePrompt);
        Guard.NotNull(factorLabels);
        var ids = StyleIds(stylePrompt, nameof(stylePrompt));
        _styleStep = true;
        try
        {
            return TrainWithCustomObjective(stylePrompt, stylePrompt,
                (_, _) => StyleLoss(ids, factorLabels, training: true, _trainingRandom), StyleOptimizer());
        }
        finally
        {
            _styleStep = false;
        }
    }

    /// <summary>The style-classification loss of a prompt and its labels, without dropout or an update.</summary>
    public T EvaluateStyleLoss(Tensor<T> stylePrompt, int[] factorLabels)
    {
        Guard.NotNull(stylePrompt);
        Guard.NotNull(factorLabels);
        using var _ = new NoGradScope<T>();
        return StyleLoss(StyleIds(stylePrompt, nameof(stylePrompt)), factorLabels, false, _trainingRandom)[0];
    }

    /// <summary>The most likely class of each style factor for a style prompt.</summary>
    public int[] PredictStyleFactors(Tensor<T> stylePrompt)
    {
        Guard.NotNull(stylePrompt);
        using var _ = new NoGradScope<T>();
        var logits = FactorLogits(StyleIds(stylePrompt, nameof(stylePrompt)), false, _trainingRandom);
        return logits.Select(row =>
        {
            int best = 0;
            for (int c = 1; c < row.Shape[1]; c++)
                if (NumOps.ToDouble(row[0, c]) > NumOps.ToDouble(row[0, best])) best = c;
            return best;
        }).ToArray();
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> StyleOptimizer()
    {
        if (_styleOptimizer is null)
        {
            var o = _options;
            int total = o.StyleTrainingSteps;
            var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.StyleLearningRate,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8,
                WeightDecay = 0.0,
                EnableGradientClipping = true,
                MaxGradientNorm = 1.0,
                // The Trainer's linear schedule without warm-up: from the full rate to zero over the run.
                LearningRateScheduler = total > 0
                    ? new LambdaLRScheduler(o.StyleLearningRate, step => Math.Max(0.0, (double)(total - step) / total))
                    : null,
            };
            _styleOptimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options), "style-encoder");
        }
        return _styleOptimizer;
    }

    /// <inheritdoc />
    /// <remarks>The style stage trains the prefixes and classifiers (and BERT under standard fine-tuning); the synthesis
    /// stage trains the content encoder, adaptor, decoder and style projection (and the style encoder only when
    /// <see cref="PromptTTSOptions.TrainStyleEncoderWithSynthesis"/>).</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (_bert is null) return parameters;
        var o = _options;
        IEnumerable<LayerBase<T>> layers;
        if (_styleStep)
        {
            layers = o.StyleFineTuning == PromptStyleFineTuning.Standard
                ? _bertLayers.Concat(_classifierLayers)
                : _prefixLayers.Concat(_factorHeads);
        }
        else
        {
            layers = Layers.OfType<LayerBase<T>>().Append(_styleProjection!);
            if (o.TrainStyleEncoderWithSynthesis)
                layers = layers.Concat(o.StyleFineTuning == PromptStyleFineTuning.PTuningV2 ? _prefixLayers : _bertLayers);
        }
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    // ---------------------------------------------------------------- pretrained weights

    /// <summary>Loads a Hugging Face BERT checkpoint (bert-base-uncased's <c>BertModel</c> or a <c>bert.</c>-prefixed
    /// one) into the style encoder, with its pooler.</summary>
    public void LoadStyleEncoderWeights(string path)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        string prefix = file.ContainsKey("bert.embeddings.word_embeddings.weight") ? "bert." : "";
        double[] Read(string name, int[] shape)
        {
            if (!file.TryGetValue(name, out var tensor))
                throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            if (!tensor.Shape.ToArray().SequenceEqual(shape))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", tensor.Shape.ToArray())}] in the checkpoint but [{string.Join(", ", shape)}] in this model.");
            return tensor.ToVector().Select(v => (double)v).ToArray();
        }
        StyleEncoder.LoadTorchWeights(Read, prefix);
        int h = _options.StyleHiddenDim;
        TorchParameters.Linear(Engine, _pooler!, h, h, Read(prefix + "pooler.dense.weight", new[] { h, h }), Read(prefix + "pooler.dense.bias", new[] { h }));
    }

    /// <summary>
    /// Loads a THUDM/P-tuning-v2 <c>BertPrefixForSequenceClassification</c> state dict: BERT and its pooler
    /// (<c>bert.*</c>), the prefix encoder (<c>prefix_encoder.*</c>) and its classifier (<c>classifier.*</c>) as the one
    /// style factor's head. <paramref name="read"/> returns a named tensor's row-major values after checking its shape.
    /// </summary>
    internal void LoadPTuningWeights(Func<string, int[], double[]> read)
    {
        var o = _options;
        if (_factorHeads.Count != 1)
            throw new InvalidOperationException("A P-tuning v2 classifier has one head; build the model with one style factor.");
        int h = o.StyleHiddenDim, prefixWidth = o.StyleLayers * 2 * h, classes = o.StyleFactorClasses[0];
        StyleEncoder.LoadTorchWeights(read, "bert.");
        TorchParameters.Linear(Engine, _pooler!, h, h, read("bert.pooler.dense.weight", new[] { h, h }), read("bert.pooler.dense.bias", new[] { h }));
        if (_prefixHidden is null)
        {
            _prefixEmbedding!.LoadTable(read("prefix_encoder.embedding.weight", new[] { o.PrefixLength, prefixWidth }));
        }
        else
        {
            _prefixEmbedding!.LoadTable(read("prefix_encoder.embedding.weight", new[] { o.PrefixLength, h }));
            TorchParameters.Linear(Engine, _prefixHidden, h, o.PrefixHiddenSize, read("prefix_encoder.trans.0.weight", new[] { o.PrefixHiddenSize, h }),
                read("prefix_encoder.trans.0.bias", new[] { o.PrefixHiddenSize }));
            TorchParameters.Linear(Engine, _prefixOutput!, o.PrefixHiddenSize, prefixWidth,
                read("prefix_encoder.trans.2.weight", new[] { prefixWidth, o.PrefixHiddenSize }), read("prefix_encoder.trans.2.bias", new[] { prefixWidth }));
        }
        TorchParameters.Linear(Engine, _factorHeads[0], h, classes, read("classifier.weight", new[] { classes, h }), read("classifier.bias", new[] { classes }));
    }

    /// <summary>BERT's [CLS] state <c>[1, StyleHiddenDim]</c> for a style prompt, without dropout.</summary>
    internal Tensor<T> EvaluateClsState(int[] ids)
    {
        using var _ = new NoGradScope<T>();
        return ClsState(ids, false, _trainingRandom);
    }

    /// <summary>The style factors' logits for a style prompt, without dropout.</summary>
    internal IReadOnlyList<Tensor<T>> EvaluateFactorLogits(int[] ids)
    {
        using var _ = new NoGradScope<T>();
        return FactorLogits(ids, false, _trainingRandom);
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "PromptTTS-Native" : "PromptTTS-ONNX",
            Description = "PromptTTS: Controllable Text-to-Speech with Text Descriptions (Guo et al., 2023)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers + _options.StyleLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "PromptTTS";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(PromptTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
