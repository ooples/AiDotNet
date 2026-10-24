using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// CLAP (Contrastive Language-Audio Pre-training) as an audio classifier: zero-shot against text prompts, or
/// fine-tuned with a linear head.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLAP (Wu et al., ICASSP 2023) learns joint audio-text representations through contrastive learning,
/// similar to CLIP for images. This class is the classification task over that network: it runs the audio and
/// text towers of <see cref="CLAPModel{T}"/> (built by the same factories from the same
/// <see cref="CLAPOptions.Network"/> options), and adds
/// <list type="bullet">
/// <item><b>zero-shot classification</b>: each prompt's text embedding against the audio embedding, both
/// L2-normalized, scaled by 1/τ and turned into probabilities with a softmax over the prompts;</item>
/// <item><b>fine-tuning</b>: LAION-CLAP's linear probe (<c>LinearProbe(mlp=False)</c>), a linear layer on the
/// audio projection, trained with binary cross-entropy on logits, read through a sigmoid.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CLAP is special because it understands both audio and text. Instead of having
/// fixed labels like "dog bark" or "siren", you can describe any sound in plain English and CLAP will
/// find it in audio. This is called "zero-shot" classification.
///
/// <b>Usage:</b>
/// <code>
/// var clap = new CLAP&lt;float&gt;(new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.MultiLabelClassification, inputSize: 480000, outputSize: 527));
/// clap.SetTextPrompts(new[] { "a dog barking", "rain falling", "a car engine" }, tokenizer);
/// var result = clap.Detect(audioTensor);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation" (Wu et al., ICASSP 2023)</item>
/// <item>Repository: https://github.com/LAION-AI/CLAP</item>
/// </list>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.EmbeddingModel)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Large-Scale Contrastive Language-Audio Pre-Training with Feature Fusion and Keyword-to-Caption Augmentation", "https://doi.org/10.1109/ICASSP49357.2023.10095969", Year = 2023, Authors = "Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta1 = 0.99, Beta2 = 0.9,
                MinLearningRate = 0,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Wu et al. 2023, Sec. 4: the Adam optimizer with beta1 0.99 and beta2 0.9, a "
                        + "warm-up and cosine learning rate decay at a basic learning rate of 1e-4. The "
                        + "betas are declared in the order the paper gives them, which is the reverse of "
                        + "the usual 0.9 and 0.99.")]
[PreprocessesInput("PreprocessAudio converts the waveform to CLAPModel's log-mel image before the audio "
    + "tower runs, so Layers[0] never sees the raw samples.")]
[StackInputLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true)]
[TaskVariantOf(typeof(AiDotNet.Audio.Fingerprinting.CLAPModel<>), "zero-shot and fine-tuned audio classification")]
public partial class CLAP<T> : AudioClassifierBase<T>, IAudioEventDetector<T>
{
    #region Fields

    private readonly CLAPOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly bool _useNativeMode;
    private bool _disposed;
    // CLAPModel's waveform -> log-mel pipeline (clip fitting, fusion views, filterbank), built on first use.
    private AiDotNet.Audio.Fingerprinting.ClapAudioFrontEnd<T>? _frontEnd;
    // Random crops of long clips while fine-tuning; inference reseeds per call, as CLAPModel does.
    private Random? _trainingRandom;
    private string[] _textPrompts = Array.Empty<string>();
    private int[][] _promptTokens = Array.Empty<int[]>();
    private string[] _headLabels;
    private OnnxModel<T>? _textEncoder;
    public static readonly string[] AudioSetLabels = BEATs<T>.AudioSetLabels;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a CLAP classifier in ONNX inference mode from a pre-trained audio encoder model file.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="modelPath">Path to the pre-trained ONNX audio encoder, which reads CLAP's log-mel image.</param>
    /// <param name="options">Optional configuration. Defaults are used if null.</param>
    public CLAP(NeuralNetworkArchitecture<T> architecture, string modelPath, CLAPOptions? options = null)
        : base(architecture, new BinaryCrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new CLAPOptions();
        Options = _options;
        _options.Network.Validate();
        _useNativeMode = false;
        base.SampleRate = _options.Network.SampleRate;
        base.NumMels = _options.Network.NumMelBands;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        if (_options.TextEncoderModelPath is { } tp && !string.IsNullOrEmpty(tp))
        {
            if (!File.Exists(tp))
                throw new FileNotFoundException($"Text encoder ONNX model not found: {tp}", tp);
            _textEncoder = new OnnxModel<T>(tp, _options.OnnxOptions);
        }
        _headLabels = _options.CustomLabels ?? AudioSetLabels;
        ClassLabels = _headLabels;
        InitializeLayers();
    }

    /// <summary>
    /// Creates a trainable CLAP classifier.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="options">Optional configuration. Defaults are used if null.</param>
    /// <param name="optimizer">Optional gradient-based optimizer; the paper's Adam is used if null.</param>
    public CLAP(NeuralNetworkArchitecture<T> architecture, CLAPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new BinaryCrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new CLAPOptions();
        Options = _options;
        _options.Network.Validate();
        _useNativeMode = true;
        // The rate this model publishes on its own options. Built bare, the optimizer
        // would use its own default instead and LearningRate would be configuration that
        // nothing reads — the defect that diverged MusicFlamingo's training.
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
        new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        { InitialLearningRate = _options.LearningRate });
        base.SampleRate = _options.Network.SampleRate;
        base.NumMels = _options.Network.NumMelBands;
        _headLabels = _options.CustomLabels ?? AudioSetLabels;
        ClassLabels = _headLabels;
        InitializeLayers();
    }

    /// <summary>
    /// Downloads and creates a CLAP classifier asynchronously from a model repository.
    /// </summary>
    /// <param name="options">Optional configuration. Defaults are used if null.</param>
    /// <param name="progress">Optional progress reporter (0.0 to 1.0).</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>A fully initialized CLAP classifier ready for inference.</returns>
    internal static async Task<CLAP<T>> CreateAsync(
        CLAPOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new CLAPOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("clap", "clap_audio_encoder.onnx",
                progress: progress, cancellationToken);
            options.ModelPath = mp;
        }

        var arch = new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiLabelClassification,
            inputSize: (int)(options.WindowSize * options.Network.SampleRate),
            outputSize: (options.CustomLabels ?? AudioSetLabels).Length);
        return new CLAP<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEventDetector Properties

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedEvents => ClassLabels;

    /// <inheritdoc />
    public IReadOnlyList<string> EventLabels => ClassLabels;

    /// <inheritdoc />
    public double TimeResolution => _options.WindowSize * (1 - _options.WindowOverlap);

    #endregion

    #region IAudioEventDetector Methods

    /// <inheritdoc />
    public AudioEventResult<T> Detect(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return Detect(audio, NumOps.FromDouble(_options.Threshold));
    }

    /// <inheritdoc />
    public AudioEventResult<T> Detect(Tensor<T> audio, T threshold)
    {
        ThrowIfDisposed();
        double tv = NumOps.ToDouble(threshold);
        double dur = audio.Length / (double)_options.Network.SampleRate;
        var wins = SplitIntoWindows(audio);
        var all = new List<AudioEvent<T>>();

        for (int w = 0; w < wins.Count; w++)
        {
            double st = w * TimeResolution;
            var sc = ClassifyWindow(wins[w]);

            for (int i = 0; i < sc.Length && i < ClassLabels.Count; i++)
            {
                if (NumOps.ToDouble(sc[i]) >= tv)
                {
                    all.Add(new AudioEvent<T>
                    {
                        EventType = ClassLabels[i],
                        Confidence = sc[i],
                        StartTime = st,
                        EndTime = Math.Min(st + _options.WindowSize, dur),
                        PeakTime = st + _options.WindowSize / 2
                    });
                }
            }
        }

        var merged = MergeEvents(all);
        return new AudioEventResult<T>
        {
            Events = merged,
            TotalDuration = dur,
            DetectedEventTypes = merged.Select(e => e.EventType).Distinct().ToList(),
            EventStats = ComputeEventStatistics(merged)
        };
    }

    /// <inheritdoc />
    public Task<AudioEventResult<T>> DetectAsync(Tensor<T> audio, CancellationToken ct = default)
        => Task.Run(() => Detect(audio), ct);

    /// <inheritdoc />
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> types)
        => DetectSpecific(audio, types, NumOps.FromDouble(_options.Threshold));

    /// <inheritdoc />
    public AudioEventResult<T> DetectSpecific(Tensor<T> audio, IReadOnlyList<string> types, T threshold)
    {
        var r = Detect(audio, threshold);
        var s = new HashSet<string>(types, StringComparer.OrdinalIgnoreCase);
        var f = r.Events.Where(e => s.Contains(e.EventType)).ToList();
        return new AudioEventResult<T>
        {
            Events = f,
            TotalDuration = r.TotalDuration,
            DetectedEventTypes = f.Select(e => e.EventType).Distinct().ToList(),
            EventStats = ComputeEventStatistics(f)
        };
    }

    /// <inheritdoc />
    public Tensor<T> GetEventProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var wins = SplitIntoWindows(audio);
        var p = new Tensor<T>([wins.Count, ClassLabels.Count]);
        for (int w = 0; w < wins.Count; w++)
        {
            var sc = ClassifyWindow(wins[w]);
            for (int i = 0; i < ClassLabels.Count && i < sc.Length; i++)
                p[w, i] = sc[i];
        }
        return p;
    }

    /// <inheritdoc />
    public IStreamingEventDetectionSession<T> StartStreamingSession()
        => StartStreamingSession(_options.Network.SampleRate, NumOps.FromDouble(_options.Threshold));

    /// <inheritdoc />
    public IStreamingEventDetectionSession<T> StartStreamingSession(int sr, T thr)
    {
        if (sr <= 0)
            throw new ArgumentOutOfRangeException(nameof(sr), "Sample rate must be positive.");
        if (sr != _options.Network.SampleRate)
            throw new ArgumentException($"Sample rate {sr} does not match model's configured sample rate {_options.Network.SampleRate}. Resample audio before streaming.", nameof(sr));
        return new CLAPStreamingSession(this, sr, thr);
    }

    #endregion

    #region CLAP-Specific Methods

    /// <summary>
    /// Sets the text prompts for zero-shot classification.
    /// </summary>
    /// <param name="prompts">Natural language descriptions of sounds to detect; empty returns to the
    /// fine-tuning head.</param>
    /// <param name="tokenize">The text tower's tokenizer (RoBERTa's BPE for the published checkpoints), giving
    /// at most <see cref="CLAPModelOptions.MaxTextLength"/> ids below <see cref="CLAPModelOptions.VocabSize"/>.</param>
    /// <remarks>
    /// <para>Each window's audio embedding is compared with each prompt's text embedding; the class labels
    /// become the prompts.</para>
    /// <para><b>For Beginners:</b> Call this to tell CLAP what sounds to look for:
    /// <code>
    /// clap.SetTextPrompts(new[] { "a dog barking", "rain falling", "a car horn honking" }, tokenizer);
    /// </code></para>
    /// </remarks>
    public void SetTextPrompts(string[] prompts, Func<string, int[]> tokenize)
    {
        ThrowIfDisposed();
        if (prompts is null) throw new ArgumentNullException(nameof(prompts));
        if (tokenize is null) throw new ArgumentNullException(nameof(tokenize));
        if (prompts.Length > 0 && !_useNativeMode && _textEncoder is null)
            throw new InvalidOperationException(
                "Zero-shot prompts in ONNX mode need a text encoder; set CLAPOptions.TextEncoderModelPath.");
        var tokens = new int[prompts.Length][];
        for (int i = 0; i < prompts.Length; i++)
        {
            var ids = tokenize(prompts[i]) ?? throw new ArgumentException($"The tokenizer returned null for \"{prompts[i]}\".", nameof(tokenize));
            if (ids.Length == 0 || ids.Length > _options.Network.MaxTextLength)
                throw new ArgumentException(
                    $"\"{prompts[i]}\" tokenizes to {ids.Length} ids; the text tower reads 1 to {_options.Network.MaxTextLength}.", nameof(tokenize));
            foreach (int id in ids)
                if (id < 0 || id >= _options.Network.VocabSize)
                    throw new ArgumentException($"Token id {id} is outside the vocabulary of {_options.Network.VocabSize}.", nameof(tokenize));
            tokens[i] = ids;
        }
        _textPrompts = (string[])prompts.Clone();
        _promptTokens = tokens;
        ClassLabels = _textPrompts.Length > 0 ? _textPrompts : _headLabels;
    }

    /// <summary>
    /// Extracts the audio embedding in the joint audio-text space, L2-normalized.
    /// </summary>
    /// <param name="audio">Raw audio waveform tensor.</param>
    /// <returns>Audio embedding in the joint space.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> An embedding is a compact numerical representation of audio.
    /// You can compare embeddings from different audio clips to measure similarity.</para>
    /// </remarks>
    public Tensor<T> ExtractAudioEmbedding(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return InEvaluationMode(() => CLAPModel<T>.L2Normalize(Engine, AudioProjection(audio)));
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc />
    /// <remarks>
    /// <c>Layers</c> holds CLAP's audio tower followed by the linear head, so the fine-tuning forward is a plain
    /// fold; the text tower lives in <c>TextEncoderLayers</c>, which the base saves and counts with the rest.
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
        {
            Layers.AddRange(CLAPModel<T>.AudioTower(_options.Network));
            Layers.Add(new DenseLayer<T>(outputSize: _headLabels.Length,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<T>()));
        }
        TextEncoderLayers.AddRange(CLAPModel<T>.TextTower(_options.Network));
    }

    /// <inheritdoc />
    /// <remarks>The fine-tuning head's sigmoid probabilities, per label.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null)
            return PostprocessOutput(OnnxEncoder.Run(PreprocessAudio(input)));
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            var (features, longer) = PrepareAudio(input);
            var c = CLAPModel<T>.RunAudioTower(Layers, Layers.Count, features, longer);
            return PostprocessOutput(Unbatch(input, c));
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => Unbatch(input, base.ForwardForTraining(PreprocessAudio(input)));

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? base.GetOrCreateBaseOptimizer();

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    /// <remarks>CLAPModel's log-mel image, <c>[batch, 1, frames, mels]</c>.</remarks>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => PrepareAudio(rawAudio).Features;

    // CLAPModel's audio input: the features [batch, views, frames, mels] and LAION-CLAP's fusion mask.
    private (Tensor<T> Features, Tensor<T> Longer) PrepareAudio(Tensor<T> rawAudio)
    {
        Guard.NotNull(rawAudio);
        _frontEnd ??= new AiDotNet.Audio.Fingerprinting.ClapAudioFrontEnd<T>(_options.Network);
        var random = IsTrainingMode
            ? _trainingRandom ??= RandomHelper.CreateSeededRandom(_options.Network.Seed ?? 0)
            : RandomHelper.CreateSeededRandom(_options.Network.Seed ?? 0);
        using (new NoGradScope<T>())
            return _frontEnd.Prepare(Engine, rawAudio, random);
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => Engine.Sigmoid(o);

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "CLAP-Native" : "CLAP-ONNX",
            Description = "CLAP Contrastive Language-Audio Pre-training (Wu et al., ICASSP 2023)",
            FeatureCount = ClassLabels.Count,
            Complexity = _options.Network.AudioStageDepths.Sum()
        };
        m.AdditionalInfo["Architecture"] = "CLAP";
        m.AdditionalInfo["ProjectionDim"] = _options.Network.ProjectionDim.ToString();
        m.AdditionalInfo["AudioEmbedDim"] = _options.Network.AudioEmbedDim.ToString();
        m.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        m.AdditionalInfo["ZeroShotMode"] = (_textPrompts.Length > 0).ToString();
        return m;
    }

    #endregion

    #region Private Helpers

    /// <summary>
    /// Runs <paramref name="inference"/> with the towers in evaluation mode and restores the previous mode, as PredictCore
    /// does. Networks and layers start in training mode, so an embedding taken without this ran the text tower's dropout
    /// and the same caption encoded differently on every call.
    /// </summary>
    private TResult InEvaluationMode<TResult>(Func<TResult> inference)
    {
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            return inference();
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    // A single waveform [samples] gives [labels]; a batch [batch, samples] keeps its batch axis.
    private Tensor<T> Unbatch(Tensor<T> input, Tensor<T> output)
        => input.Rank == 1 && output.Rank == 2 && output.Shape[0] == 1
            ? Engine.Reshape(output, new[] { output.Shape[1] })
            : output;

    // The audio tower's projection into the joint space, before normalization (what the linear head reads).
    private Tensor<T> AudioProjection(Tensor<T> audio)
    {
        var (features, longer) = PrepareAudio(audio);
        if (IsOnnxMode && OnnxEncoder is not null)
            return OnnxEncoder.Run(features);
        if (!_useNativeMode)
            throw new InvalidOperationException(
                "No model available for audio embedding. Provide an ONNX model path or use native training mode.");
        using var _ = new NoGradScope<T>();
        return CLAPModel<T>.RunAudioTower(Layers, Layers.Count - 1, features, longer);
    }

    private Tensor<T> TextEmbedding(int[] ids)
    {
        var tokens = new Tensor<T>([1, ids.Length]);
        for (int i = 0; i < ids.Length; i++) tokens[0, i] = NumOps.FromDouble(ids[i]);
        if (_textEncoder is not null)
            return CLAPModel<T>.L2Normalize(Engine, _textEncoder.Run(tokens));
        using var _ = new NoGradScope<T>();
        var c = tokens;
        foreach (var l in TextEncoderLayers) c = l.Forward(c);
        return CLAPModel<T>.L2Normalize(Engine, c);
    }

    private T[] ClassifyWindow(Tensor<T> window)
    {
        if (_textPrompts.Length > 0)
            return ZeroShotScores(window);
        var probabilities = Predict(window);
        var scores = new T[ClassLabels.Count];
        for (int i = 0; i < Math.Min(probabilities.Length, scores.Length); i++)
            scores[i] = probabilities[i];
        return scores;
    }

    // Zero-shot (Wu et al. 2023, §4.2; LAION-CLAP's evaluation): the softmax over the prompts of the cosine
    // similarities scaled by 1/τ.
    private T[] ZeroShotScores(Tensor<T> window) => InEvaluationMode(() => ZeroShotScoresCore(window));

    private T[] ZeroShotScoresCore(Tensor<T> window)
    {
        var audio = ExtractAudioEmbedding(window).ToVector();
        var logits = new double[_textPrompts.Length];
        double scale = 1.0 / _options.Network.InitialTemperature, max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            var text = TextEmbedding(_promptTokens[i]).ToVector();
            double dot = 0;
            for (int d = 0; d < Math.Min(audio.Length, text.Length); d++)
                dot += NumOps.ToDouble(audio[d]) * NumOps.ToDouble(text[d]);
            logits[i] = scale * dot;
            max = Math.Max(max, logits[i]);
        }
        double sum = 0;
        for (int i = 0; i < logits.Length; i++) sum += logits[i] = Math.Exp(logits[i] - max);
        var scores = new T[logits.Length];
        for (int i = 0; i < logits.Length; i++) scores[i] = NumOps.FromDouble(logits[i] / sum);
        return scores;
    }

    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio)
    {
        var w = new List<Tensor<T>>();
        int ws = (int)(_options.WindowSize * _options.Network.SampleRate);
        int hs = (int)(ws * (1 - _options.WindowOverlap));
        if (hs <= 0) hs = 1;
        int ls = 0;

        for (int s = 0; s + ws <= audio.Length; s += hs)
        {
            var t = new Tensor<T>([ws]);
            for (int i = 0; i < ws; i++) t[i] = audio[s + i];
            w.Add(t);
            ls = s + hs;
        }

        int rs = w.Count > 0 ? ls : 0;
        int rem = audio.Length - rs;
        if (rem > ws / 10)
        {
            var t = new Tensor<T>([ws]);
            for (int i = 0; i < rem && i < ws; i++) t[i] = audio[rs + i];
            w.Add(t);
        }
        else if (w.Count == 0 && audio.Length > 0)
        {
            var t = new Tensor<T>([ws]);
            for (int i = 0; i < audio.Length; i++) t[i] = audio[i];
            w.Add(t);
        }

        return w;
    }

    private List<AudioEvent<T>> MergeEvents(List<AudioEvent<T>> events)
    {
        if (events.Count == 0) return events;
        var m = new List<AudioEvent<T>>();
        foreach (var g in events.GroupBy(e => e.EventType))
        {
            var sorted = g.OrderBy(e => e.StartTime).ToList();
            var cur = sorted[0];
            for (int i = 1; i < sorted.Count; i++)
            {
                var next = sorted[i];
                if (next.StartTime <= cur.EndTime + 0.1)
                {
                    double cc = NumOps.ToDouble(cur.Confidence);
                    double nc = NumOps.ToDouble(next.Confidence);
                    cur = new AudioEvent<T>
                    {
                        EventType = cur.EventType,
                        StartTime = cur.StartTime,
                        EndTime = Math.Max(cur.EndTime, next.EndTime),
                        Confidence = cc > nc ? cur.Confidence : next.Confidence,
                        PeakTime = cc > nc ? cur.PeakTime : next.PeakTime
                    };
                }
                else
                {
                    m.Add(cur);
                    cur = next;
                }
            }
            m.Add(cur);
        }
        return m.OrderBy(e => e.StartTime).ToList();
    }

    private Dictionary<string, EventStatistics<T>> ComputeEventStatistics(IReadOnlyList<AudioEvent<T>> events)
    {
        var s = new Dictionary<string, EventStatistics<T>>();
        foreach (var g in events.GroupBy(e => e.EventType))
        {
            var l = g.ToList();
            s[g.Key] = new EventStatistics<T>
            {
                Count = l.Count,
                TotalDuration = l.Sum(e => e.Duration),
                AverageConfidence = NumOps.FromDouble(l.Average(e => NumOps.ToDouble(e.Confidence))),
                MaxConfidence = NumOps.FromDouble(l.Max(e => NumOps.ToDouble(e.Confidence)))
            };
        }
        return s;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CLAP<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            OnnxEncoder?.Dispose();
            _textEncoder?.Dispose();
        }
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion

    #region Streaming Session

    private sealed class CLAPStreamingSession : IStreamingEventDetectionSession<T>
    {
        private readonly CLAP<T> _det;
        private readonly T _thr;
        private readonly List<T> _buf;
        private readonly List<AudioEvent<T>> _new;
        private readonly Dictionary<string, T> _state;
        private readonly int _ws;
        private double _pt;
        private volatile bool _disp;
        private readonly object _lk = new();

        public event EventHandler<AudioEvent<T>>? EventDetected;

        public CLAPStreamingSession(CLAP<T> det, int sr, T thr)
        {
            _det = det;
            _thr = thr;
            _buf = [];
            _new = [];
            _state = new Dictionary<string, T>();
            _ws = (int)(det._options.WindowSize * sr);
            foreach (var l in det.ClassLabels)
                _state[l] = det.NumOps.Zero;
        }

        public void FeedAudio(Tensor<T> chunk)
        {
            if (_disp) throw new ObjectDisposedException(nameof(CLAPStreamingSession));
            List<AudioEvent<T>>? raise = null;
            lock (_lk)
            {
                if (_disp) throw new ObjectDisposedException(nameof(CLAPStreamingSession));
                for (int i = 0; i < chunk.Length; i++) _buf.Add(chunk[i]);

                while (_buf.Count >= _ws)
                {
                    var w = new Tensor<T>([_ws]);
                    for (int i = 0; i < _ws; i++) w[i] = _buf[i];
                    var scores = _det.ClassifyWindow(w);
                    double tv = _det.NumOps.ToDouble(_thr);

                    for (int i = 0; i < scores.Length && i < _det.ClassLabels.Count; i++)
                    {
                        _state[_det.ClassLabels[i]] = scores[i];
                        if (_det.NumOps.ToDouble(scores[i]) >= tv)
                        {
                            var e = new AudioEvent<T>
                            {
                                EventType = _det.ClassLabels[i],
                                Confidence = scores[i],
                                StartTime = _pt,
                                EndTime = _pt + _det._options.WindowSize,
                                PeakTime = _pt + _det._options.WindowSize / 2
                            };
                            _new.Add(e);
                            raise ??= [];
                            raise.Add(e);
                        }
                    }

                    int hs = (int)(_ws * (1 - _det._options.WindowOverlap));
                    if (hs <= 0) hs = 1;
                    _buf.RemoveRange(0, hs);
                    _pt += hs / (double)_det._options.Network.SampleRate;
                }
            }

            if (raise is not null)
                foreach (var e in raise)
                    EventDetected?.Invoke(this, e);
        }

        public IReadOnlyList<AudioEvent<T>> GetNewEvents()
        {
            lock (_lk)
            {
                var e = _new.ToList();
                _new.Clear();
                return e;
            }
        }

        public IReadOnlyDictionary<string, T> GetCurrentState()
        {
            lock (_lk) { return new Dictionary<string, T>(_state); }
        }

        public void Dispose()
        {
            if (_disp) return;
            lock (_lk)
            {
                if (_disp) return;
                _disp = true;
                _buf.Clear();
                _new.Clear();
                _state.Clear();
            }
        }
    }

    #endregion
}
