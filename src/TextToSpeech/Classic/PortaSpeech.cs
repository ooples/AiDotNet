using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>The two training stages of <see cref="PortaSpeech{T}"/>.</summary>
public enum PortaSpeechTrainingPhase
{
    /// <summary>The linguistic encoder and variational generator train on the duration, reconstruction and KL losses.</summary>
    VariationalGenerator = 0,

    /// <summary>The post-net trains on its negative log-likelihood with everything else frozen; synthesis runs it.</summary>
    PostNet = 1,
}

/// <summary>
/// PortaSpeech: a portable text-to-speech model with a mixture-alignment linguistic encoder, a VAE variational
/// generator with a normalizing-flow prior, and a flow-based post-net with grouped parameter sharing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "PortaSpeech: Portable and High-Quality Generative Text-to-Speech" (Ren et al., NeurIPS
/// 2021) and its reference implementation (NATSpeech) for what the paper leaves unstated.</para>
/// <para>
/// The linguistic encoder (§3.1, App. A.1) encodes phonemes with relative-position FFT layers, averages them per word
/// for a word encoder, predicts phoneme durations that sum to word durations, expands the word states over the frames
/// and attends from each frame to the phonemes of its own word, with learnable positional encodings scaled by the
/// relative position inside the word. The variational generator (§3.2, App. A.2) is a stride-4 VAE with WaveNet encoder
/// and decoder conditioned on those linguistic features, whose prior is a volume-preserving coupling flow; its KL is the
/// Monte-Carlo estimate of Eq. 3. The post-net (§3.3, App. A.3) is a conditional Glow whose WaveNets are shared within
/// groups of steps. Training (§3.4) minimizes the word duration loss, the reconstruction MAE and the KL, then the
/// post-net negative log-likelihood; see <see cref="CurrentPhase"/>.
/// </para>
/// <para>Supervision: phoneme durations (<see cref="TtsTrainingSample{T}.Durations"/>), summed per word. Words are
/// <see cref="TtsTrainingSample{T}.WordLengths"/> when given, otherwise runs of non-space tokens with every space its
/// own boundary word.</para>
/// <para><b>For Beginners:</b> PortaSpeech plans speech word by word, sketches a spectrogram with a small variational
/// autoencoder, and then sharpens the details with a normalizing flow, keeping the whole model small.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 200, outputSize: 80);
/// var model = new PortaSpeech&lt;double&gt;(architecture, new PortaSpeechOptions());
/// model.Train(new TtsTrainingSample&lt;double&gt; { Tokens = tokens, Mel = mel, Durations = durations });
/// model.CurrentPhase = PortaSpeechTrainingPhase.PostNet;   // after the generator has converged
/// var mel = model.TextToMel("hello world");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "PortaSpeech: Portable and High-Quality Generative Text-to-Speech",
    "https://arxiv.org/abs/2109.15166",
    Year = 2021,
    Authors = "Ren et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-9,
                Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 4000,
                ReferenceBatchSize = 64,
                Provenance = RecipeProvenance.DerivedFromCitedWork,
                Source = "Ren et al. 2021, Sec. 4.1: Adam with beta1 0.9, beta2 0.98 and epsilon 1e-9, "
                        + "following the learning rate schedule of its reference [35], Vaswani et al. "
                        + "2017 (inverse-square-root with 4000 warmup steps, declared here as Noam), batch "
                        + "size 64 sentences. Training runs 320k steps to convergence.")]
public partial class PortaSpeech<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly PortaSpeechOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<PortaSpeechTrainingPhase, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _phaseOptimizers = new();
    private bool _useNativeMode;
    private bool _disposed;
    private Random _trainingRandom;

    private EmbeddingLayer<T>? _embedding;
    private ResidualConvReluNormLayer<T>? _phonemePrenet;
    private readonly List<RelativePositionTransformerBlock<T>> _phonemeBlocks = new();
    private ResidualConvReluNormLayer<T>? _wordPrenet;
    private readonly List<RelativePositionTransformerBlock<T>> _wordBlocks = new();
    private VariancePredictorLayer<T>? _durationPredictor;
    private BiasFreeLinearLayer<T>? _queryPosition;
    private BiasFreeLinearLayer<T>? _keyValuePosition;
    private BiasFreeLinearLayer<T>? _query;
    private BiasFreeLinearLayer<T>? _key;
    private BiasFreeLinearLayer<T>? _value;
    private BiasFreeLinearLayer<T>? _attentionOutput;
    private PortaSpeechVariationalGenerator<T>? _generator;
    private PortaSpeechPostNet<T>? _postNet;

    public override ModelOptions GetOptions() => _options;

    public PortaSpeech(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        PortaSpeechOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new PortaSpeechOptions();
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
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

    public PortaSpeech(
        NeuralNetworkArchitecture<T> architecture,
        PortaSpeechOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new PortaSpeechOptions();
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        MaxGradNorm = NumOps.FromDouble(_options.GradientClipNorm);
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>
    /// The training stage (reference two-stage schedule): <see cref="PortaSpeechTrainingPhase.VariationalGenerator"/>
    /// first, then <see cref="PortaSpeechTrainingPhase.PostNet"/> (the reference switches after 160k updates). Synthesis
    /// runs the post-net only in the post-net phase, since an untrained post-net emits noise.
    /// </summary>
    public PortaSpeechTrainingPhase CurrentPhase { get; set; } = PortaSpeechTrainingPhase.VariationalGenerator;

    /// <summary>Generator updates so far; the KL term is optimized from <see cref="PortaSpeechOptions.KlStartUpdates"/>
    /// on. Set it when resuming training.</summary>
    public int GeneratorUpdates { get; set; }

    /// <summary>Generates a mel spectrogram from text (the paper pairs it with HiFi-GAN).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Durations;

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
        int h = o.HiddenDim;
        _embedding = new EmbeddingLayer<T>(o.VocabSize, h);
        _phonemePrenet = new ResidualConvReluNormLayer<T>(h, o.PrenetKernelSize, o.PrenetLayers, 0.0);
        for (int i = 0; i < o.NumEncoderLayers; i++)
            _phonemeBlocks.Add(new RelativePositionTransformerBlock<T>(h, o.NumHeads, o.FilterChannels, o.EncoderKernelSize,
                o.DropoutRate, o.RelativeWindow));
        _wordPrenet = new ResidualConvReluNormLayer<T>(h, o.PrenetKernelSize, o.PrenetLayers, 0.0);
        for (int i = 0; i < o.NumWordEncoderLayers; i++)
            _wordBlocks.Add(new RelativePositionTransformerBlock<T>(h, o.NumHeads, o.FilterChannels, o.EncoderKernelSize,
                o.DropoutRate, o.RelativeWindow));
        _durationPredictor = new VariancePredictorLayer<T>(h, h, 1, o.DurationPredictorKernelSize, o.DurationPredictorDropout);
        _queryPosition = new BiasFreeLinearLayer<T>(1, h);
        _keyValuePosition = new BiasFreeLinearLayer<T>(1, h);
        _query = new BiasFreeLinearLayer<T>(h, h);
        _key = new BiasFreeLinearLayer<T>(h, h);
        _value = new BiasFreeLinearLayer<T>(h, h);
        _attentionOutput = new BiasFreeLinearLayer<T>(h, h);
        _generator = new PortaSpeechVariationalGenerator<T>(Engine, o.MelChannels, h, o.GeneratorChannels, o.ProsodyDim,
            o.GeneratorKernelSize, o.GeneratorEncoderLayers, o.GeneratorDecoderLayers, o.GeneratorStride, o.PriorFlowSteps,
            o.PriorFlowLayers, o.PriorFlowChannels, o.PriorFlowKernelSize);
        _postNet = new PortaSpeechPostNet<T>(Engine, o.MelChannels, o.MelChannels + h, o.PostNetChannels, o.PostNetKernelSize,
            o.NumFlowLayers, o.PostNetLayers, 4, o.PostNetShareGroupSize);

        var encoder = new List<ILayer<T>> { _embedding, _phonemePrenet };
        encoder.AddRange(_phonemeBlocks);
        AddEncoderDecoderLayers(encoder, Array.Empty<ILayer<T>>());
        ComponentLayers.Add(_wordPrenet);
        ComponentLayers.AddRange(_wordBlocks);
        ComponentLayers.Add(_durationPredictor);
        ComponentLayers.AddRange(new LayerBase<T>[] { _queryPosition, _keyValuePosition, _query, _key, _value, _attentionOutput });
        ComponentLayers.AddRange(_generator.Layers);
        ComponentLayers.AddRange(_postNet.Layers);
    }

    private bool HasPaperLayers => _embedding is not null;

    // ---------------------------------------------------------------- linguistic encoder

    /// <summary>Tokens per word: the caller's, or runs of non-space tokens with each space token its own word.</summary>
    private int[] ResolveWords(Tensor<T> tokens, int[]? wordLengths)
    {
        int count = tokens.Length;
        if (wordLengths is not null)
        {
            if (wordLengths.Any(l => l <= 0) || wordLengths.Sum() != count)
                throw new ArgumentException($"Word lengths must be positive and sum to the {count} tokens.", nameof(wordLengths));
            return wordLengths;
        }
        var domain = GetInputDomain(new[] { count });
        int range = Math.Max(1, domain.MaxExclusive - domain.MinInclusive);
        double space = domain.IsResolved ? domain.MinInclusive + ' ' % range : ' ';
        var words = new List<int>();
        int run = 0;
        for (int i = 0; i < count; i++)
        {
            if (Math.Abs(NumOps.ToDouble(tokens[i]) - space) < 0.5)
            {
                if (run > 0) words.Add(run);
                words.Add(1);
                run = 0;
            }
            else run++;
        }
        if (run > 0) words.Add(run);
        return words.ToArray();
    }

    // [words, tokens] with 1 where the token belongs to the word.
    private Tensor<T> Membership(int[] words, int tokens)
    {
        var m = new Tensor<T>(new[] { words.Length, tokens });
        int p = 0;
        for (int w = 0; w < words.Length; w++)
            for (int i = 0; i < words[w]; i++, p++) m[w, p] = NumOps.One;
        return m;
    }

    private (Tensor<T> Phonemes, Tensor<T> Words) Encode(Tensor<T> tokens, int[] words)
    {
        var x = Engine.TensorMultiplyScalar(_embedding!.Forward(tokens), NumOps.FromDouble(Math.Sqrt(_options.HiddenDim)));
        x = _phonemePrenet!.Forward(x);
        foreach (var block in _phonemeBlocks) x = block.Forward(x);

        // Word-level pooling (Fig. 4d): the mean of each word's phoneme states.
        var pooling = Membership(words, tokens.Length);
        for (int w = 0; w < words.Length; w++)
            for (int p = 0; p < tokens.Length; p++)
                if (NumOps.ToDouble(pooling[w, p]) > 0) pooling[w, p] = NumOps.FromDouble(1.0 / words[w]);
        var y = _wordPrenet!.Forward(Engine.TensorMatMul(pooling, x));
        foreach (var block in _wordBlocks) y = block.Forward(y);
        return (x, y);
    }

    /// <summary>Predicted word durations in frames: Softplus phoneme durations summed per word, with the encoder's
    /// gradient scaled by <see cref="PortaSpeechOptions.DurationPredictorGradientScale"/>.</summary>
    private Tensor<T> PredictWordDurations(Tensor<T> phonemes, int[] words)
    {
        var detached = new Tensor<T>(phonemes._shape, phonemes.ToVector());
        var input = Engine.TensorAdd(detached, Engine.TensorMultiplyScalar(Engine.TensorSubtract(phonemes, detached),
            NumOps.FromDouble(_options.DurationPredictorGradientScale)));
        var perPhoneme = Engine.Softplus(Engine.Reshape(_durationPredictor!.Forward(input), new[] { phonemes.Shape[0], 1 }));
        return Engine.Reshape(Engine.TensorMatMul(Membership(words, phonemes.Shape[0]), perPhoneme), new[] { words.Length });
    }

    /// <summary>
    /// Mixture alignment (§3.1, App. A.1): word states expanded over their frames attend, with 2 heads, only to the
    /// phonemes of their own word; queries carry <c>(j / T_w) E_q</c>, keys and values <c>(i / L_w) E_kv</c>. The result
    /// is the attention output plus the query.
    /// </summary>
    private Tensor<T> MixtureAlign(Tensor<T> phonemes, Tensor<T> words, int[] wordLengths, int[] wordFrames)
    {
        int h = _options.HiddenDim, frames = wordFrames.Sum(), tokens = phonemes.Shape[0], heads = _options.AttentionHeads;
        var frameFraction = new Tensor<T>(new[] { frames, 1 });
        var tokenFraction = new Tensor<T>(new[] { tokens, 1 });
        var frameWord = new int[frames];
        var tokenWord = new int[tokens];
        for (int w = 0, f = 0, p = 0; w < wordLengths.Length; w++)
        {
            for (int j = 0; j < wordFrames[w]; j++, f++) { frameFraction[f, 0] = NumOps.FromDouble((double)j / wordFrames[w]); frameWord[f] = w; }
            for (int i = 0; i < wordLengths[w]; i++, p++) { tokenFraction[p, 0] = NumOps.FromDouble((double)i / wordLengths[w]); tokenWord[p] = w; }
        }
        var mask = new Tensor<T>(new[] { frames, tokens });
        for (int f = 0; f < frames; f++)
            for (int p = 0; p < tokens; p++)
                if (frameWord[f] != tokenWord[p]) mask[f, p] = NumOps.FromDouble(-1e9);

        var query = Engine.TensorAdd(LengthRegulator.Expand(words, wordFrames), _queryPosition!.Forward(frameFraction));
        var keyValue = Engine.TensorAdd(phonemes, _keyValuePosition!.Forward(tokenFraction));
        var q = _query!.Forward(query);
        var k = _key!.Forward(keyValue);
        var v = _value!.Forward(keyValue);
        int dk = h / heads;
        var outputs = new Tensor<T>[heads];
        for (int i = 0; i < heads; i++)
        {
            var qh = Engine.TensorSlice(q, new[] { 0, i * dk }, new[] { frames, dk });
            var kh = Engine.TensorSlice(k, new[] { 0, i * dk }, new[] { tokens, dk });
            var vh = Engine.TensorSlice(v, new[] { 0, i * dk }, new[] { tokens, dk });
            var logits = Engine.TensorAdd(Engine.TensorMultiplyScalar(Engine.TensorMatMul(qh, Engine.TensorTranspose(kh)),
                NumOps.FromDouble(1.0 / Math.Sqrt(dk))), mask);
            outputs[i] = Engine.TensorMatMul(Engine.TensorSoftmax(logits, axis: 1), vh);
        }
        var attended = _attentionOutput!.Forward(heads == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 1));
        return Engine.TensorAdd(attended, query);
    }

    /// <summary>Drops trailing frames so the total is a multiple of <see cref="PortaSpeechOptions.FramesMultiple"/>
    /// (reference <c>clip_mel2token_to_multiple</c>); at least one multiple is kept.</summary>
    private int[] ClipFrames(int[] wordFrames)
    {
        int multiple = Math.Max(1, _options.FramesMultiple);
        var clipped = (int[])wordFrames.Clone();
        int total = clipped.Sum(), target = Math.Max(multiple, total / multiple * multiple);
        if (target > total) clipped[^1] += target - total;
        for (int w = clipped.Length - 1; w >= 0 && clipped.Sum() > target; w--)
            clipped[w] -= Math.Min(clipped[w], clipped.Sum() - target);
        return clipped;
    }

    private Tensor<T> Gaussian(int[] shape, Random random, double scale)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = NumOps.FromDouble(scale * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return t;
    }

    // ---------------------------------------------------------------- inference

    /// <inheritdoc />
    /// <remarks>Inference (§3.4): encode, round the predicted word durations (× <see cref="PortaSpeechOptions.DurationScale"/>),
    /// align, sample the latent from the flow prior and decode the coarse mel; in the post-net phase the post-net then
    /// generates the fine mel from noise at temperature 0.8.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        if (!HasPaperLayers)
        {
            var x = input;
            foreach (var layer in Layers) x = layer.Forward(x);
            return x;
        }
        if (input.Rank == 1)
            return SynthesizeOne(input);
        if (input.Rank != 2)
            throw new ArgumentException($"Expected tokens [tokens] or [batch, tokens], got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = input.Shape[0], count = input.Shape[1];
        var outputs = new List<Tensor<T>>(batch);
        for (int b = 0; b < batch; b++)
        {
            var row = new Tensor<T>(new[] { count });
            for (int i = 0; i < count; i++) row[i] = input[b, i];
            outputs.Add(SynthesizeOne(row));
        }
        int longest = outputs.Max(o => o.Shape[0]);
        var result = new Tensor<T>(new[] { batch, longest, _options.MelChannels });
        for (int b = 0; b < batch; b++)
            for (int f = 0; f < outputs[b].Shape[0]; f++)
                for (int m = 0; m < _options.MelChannels; m++) result[b, f, m] = outputs[b][f, m];
        return result;
    }

    private Tensor<T> SynthesizeOne(Tensor<T> tokens)
    {
        using var _ = new NoGradScope<T>();
        var words = ResolveWords(tokens, null);
        var (phonemes, wordStates) = Encode(tokens, words);
        var predicted = PredictWordDurations(phonemes, words);
        var wordFrames = new int[words.Length];
        for (int w = 0; w < words.Length; w++)
            wordFrames[w] = (int)Math.Round(NumOps.ToDouble(predicted[w]) * _options.DurationScale, MidpointRounding.ToEven);
        wordFrames = ClipFrames(wordFrames);
        int frames = wordFrames.Sum();

        var linguistic = PortaSpeechOps.ChannelsFirst(Engine, MixtureAlign(phonemes, wordStates, words, wordFrames));
        var squeezed = _generator!.SqueezeCondition(linguistic);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var latent = _generator.SamplePrior(squeezed, Gaussian(new[] { 1, _options.ProsodyDim, squeezed.Shape[2] }, random, 1.0));
        var mel = _generator.Decode(latent, linguistic);
        if (CurrentPhase == PortaSpeechTrainingPhase.PostNet)
        {
            var condition = Engine.TensorConcatenate(new[] { mel, linguistic }, 1);
            mel = _postNet!.Generate(Gaussian(new[] { 1, _options.MelChannels, frames }, random, _options.PostNetTemperature), condition);
        }
        return PortaSpeechOps.Rows(Engine, mel);
    }

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        ThrowIfTokenMelTrainingUnsupported();
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var (mel, objective) = BuildObjective(sample, _trainingRandom, countUpdate: true);
        return TrainWithCustomObjective(sample.Tokens, mel, objective, PhaseOptimizer());
    }

    /// <inheritdoc />
    /// <remarks>The VAE noise is fixed by the sampling seed so the same parameters always score the same.</remarks>
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var (mel, objective) = BuildObjective(sample, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed), countUpdate: false);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective(sample.Tokens, mel)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? PhaseOptimizer()
    {
        if (_suppliedOptimizer is not null)
            return _suppliedOptimizer;
        if (!_phaseOptimizers.TryGetValue(CurrentPhase, out var optimizer))
        {
            optimizer = CurrentPhase == PortaSpeechTrainingPhase.VariationalGenerator
                ? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this) ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this)
                // The post-net's own optimizer (reference post_flow_optimizer: AdamW at 1e-3, betas 0.9/0.98, no decay).
                : PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
                    new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                    {
                        InitialLearningRate = _options.PostNetLearningRate,
                        Beta1 = 0.9,
                        Beta2 = 0.98,
                        WeightDecay = 0.0,
                    }));
            _phaseOptimizers[CurrentPhase] = optimizer;
        }
        return optimizer;
    }

    /// <inheritdoc />
    /// <remarks>The generator phase updates everything but the post-net; the post-net phase only the post-net.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var postNet = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_postNet!.Layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return CurrentPhase == PortaSpeechTrainingPhase.PostNet
            ? parameters.Where(postNet.Contains).ToList()
            : parameters.Where(p => !postNet.Contains(p)).ToList();
    }

    /// <summary>
    /// §3.4: the generator phase minimizes <c>L_dur + L_VG + L_KL</c> — the MSE of log(1 + d) word durations, the MAE
    /// of the coarse mel, and the Monte-Carlo KL (held at its value, without gradient, for the first
    /// <see cref="PortaSpeechOptions.KlStartUpdates"/> updates); the post-net phase minimizes <c>L_PN</c>, the post-net's
    /// negative log-likelihood of the mel given the (detached) coarse mel and linguistic features.
    /// </summary>
    private (Tensor<T> Mel, Func<Tensor<T>, Tensor<T>, Tensor<T>> Objective) BuildObjective(TtsTrainingSample<T> sample,
        Random random, bool countUpdate)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        var durations = sample.Durations ?? throw new ArgumentException(
            $"{nameof(PortaSpeech<T>)} trains on phoneme durations; set {nameof(sample.Durations)}.", nameof(sample));
        if (durations.Length != sample.Tokens.Length)
            throw new ArgumentException($"Expected one duration per token ({sample.Tokens.Length}), got {durations.Length}.", nameof(sample));
        var targets = DeriveAcousticTargets(sample);
        if (durations.Sum() != targets.MelFrames)
            throw new ArgumentException($"The durations sum to {durations.Sum()} frames but the mel spectrogram has {targets.MelFrames}.", nameof(sample));
        var words = ResolveWords(sample.Tokens, sample.WordLengths);
        var wordFrames = new int[words.Length];
        for (int w = 0, p = 0; w < words.Length; w++)
            for (int i = 0; i < words[w]; i++, p++) wordFrames[w] += durations[p];
        int full = wordFrames.Sum();
        wordFrames = ClipFrames(wordFrames);
        int frames = wordFrames.Sum();
        if (frames > full)
            throw new ArgumentException($"PortaSpeech needs at least {frames} frames (its generator stride); got {full}.", nameof(sample));
        var mel = Engine.TensorSlice(targets.Mel, new[] { 0, 0 }, new[] { frames, _options.MelChannels });
        var target = new Tensor<T>(new[] { words.Length });
        for (int w = 0; w < words.Length; w++) target[w] = NumOps.FromDouble(Math.Log(1 + wordFrames[w]));
        var noise = Gaussian(new[] { 1, _options.ProsodyDim, frames / _options.GeneratorStride }, random, 1.0);
        var phase = CurrentPhase;
        bool klActive = GeneratorUpdates >= _options.KlStartUpdates;
        if (countUpdate && phase == PortaSpeechTrainingPhase.VariationalGenerator) GeneratorUpdates++;

        Tensor<T> Objective(Tensor<T> tokens, Tensor<T> melTarget)
        {
            var melChannelsFirst = PortaSpeechOps.ChannelsFirst(Engine, melTarget);
            if (phase == PortaSpeechTrainingPhase.PostNet)
            {
                Tensor<T> condition;
                using (new NoGradScope<T>())
                {
                    var (ph, wd) = Encode(tokens, words);
                    var lingustic = PortaSpeechOps.ChannelsFirst(Engine, MixtureAlign(ph, wd, words, wordFrames));
                    var (latent, _) = _generator!.Encode(melChannelsFirst, _generator.SqueezeCondition(lingustic), noise);
                    var coarse = _generator.Decode(latent, lingustic);
                    var joined = Engine.TensorConcatenate(new[] { coarse, lingustic }, 1);
                    condition = new Tensor<T>(joined._shape, joined.ToVector());
                }
                return _postNet!.NegativeLogLikelihood(melChannelsFirst, condition);
            }

            var (phonemes, wordStates) = Encode(tokens, words);
            var logDuration = Engine.TensorLog(Engine.TensorAddScalar(PredictWordDurations(phonemes, words), NumOps.One));
            var durationError = Engine.TensorSubtract(logDuration, target);
            var durationLoss = Engine.ReduceMean(Engine.TensorMultiply(durationError, durationError), new[] { 0 }, keepDims: false);

            var linguistic = PortaSpeechOps.ChannelsFirst(Engine, MixtureAlign(phonemes, wordStates, words, wordFrames));
            var (z, kl) = _generator!.Encode(melChannelsFirst, _generator.SqueezeCondition(linguistic), noise);
            var reconstruction = _generator.Decode(z, linguistic);
            var l1 = Engine.ReduceMean(Engine.TensorAbs(Engine.TensorSubtract(reconstruction, melChannelsFirst)), new[] { 0, 1, 2 }, keepDims: false);
            var klTerm = klActive ? kl : new Tensor<T>(kl._shape, kl.ToVector());
            return Engine.TensorAdd(Engine.TensorAdd(durationLoss, l1), klTerm);
        }

        return (mel, Objective);
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
            Name = _useNativeMode ? "PortaSpeech-Native" : "PortaSpeech-ONNX",
            Description = "PortaSpeech: Portable and High-Quality Generative Text-to-Speech (Ren et al., 2021)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumFlowLayers,
        };
        m.AdditionalInfo["Architecture"] = "PortaSpeech";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(PortaSpeech<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
