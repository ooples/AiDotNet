using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.SpeechRecognition.Foundation;

/// <summary>
/// SPIRAL: a convolution-Transformer speech encoder pre-trained to make its representation of perturbed speech match a
/// moving-average teacher's representation of the clean speech, with a convolutional CTC classifier.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "SPIRAL: Self-supervised Perturbation-Invariant Representation Learning for Speech Pre-Training" (Huang et al., ICLR 2022)</item></list></para>
/// <para><b>For Beginners:</b> SPIRAL learns from unlabeled speech with two copies of one network. The teacher hears the
/// clean recording; the student hears it with parts blanked out, and learns to produce what the teacher produced. The
/// teacher is a slowly moving average of the student, so the two improve together. Afterwards the student's encoder is
/// fine-tuned on transcribed speech.</para>
/// <para>
/// The network is the BASE model of Table 1: 128-bin log-mel input, two blocks of down-sampling convolutions and a
/// pre-norm Transformer with convolutional relative position encoding (40 ms, then 80 ms frames), a linear projection head
/// and the student's convolutional predictor (section 3.2). <see cref="Pretrain"/> runs the pre-training of section 3.1:
/// adaptive SpecAugment on the student's input (section 3.3), random padding of the teacher's input against positional
/// collapse, the in-utterance contrastive loss of Equation 2, and the teacher's moving-average update on a cosine
/// schedule (Equation 1, section 4.2). Ordinary training fine-tunes the encoder with the convolutional classifier and CTC
/// (section 3.5).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SPIRAL recognizer
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 1024);
/// var model = new SPIRAL&lt;double&gt;(architecture);
///
/// // Or load an exported ONNX model
/// var onnxModel = new SPIRAL&lt;double&gt;(architecture, "spiral.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SPIRAL: Self-supervised Perturbation-Invariant Representation Learning for Speech Pre-Training", "https://arxiv.org/abs/2201.10207", Year = 2022, Authors = "Huang et al.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 3e-3, ReferenceBatchSize = 384,
                WarmupFraction = 0.08, Schedule = LearningRateSchedulerType.CosineAnnealing, MinLearningRate = 0,
                Phase = TrainingPhase.PreTraining,
                Source = "Huang et al. 2022, Sec. 4.2: pre-training optimizes with Adam, warming the learning rate up "
                        + "for the first 8% of updates to a peak of 3e-3, then decaying it to 0 with a cosine schedule. "
                        + "BASE trains at 24 per GPU on 16 GPUs, 384 in total.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 3e-5, ReferenceBatchSize = 112,
                Schedule = LearningRateSchedulerType.TriStage, WarmupFraction = 0.10, HoldFraction = 0.40, MinLearningRate = 0,
                Phase = TrainingPhase.FineTuning,
                Source = "Huang et al. 2022, Sec. 4.2: fine-tuning uses Adam with a tri-state schedule, warmed up for the "
                        + "first 10% of updates to 3e-5, held for the next 40%, then decayed linearly to zero. BASE "
                        + "fine-tunes at 14 per GPU on 8 GPUs, 112 in total.")]
public partial class SPIRAL<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    /// <inheritdoc />
    /// <remarks>The CTC classifier's width: the projection head and predictor are used only by <see cref="Pretrain"/>.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly SPIRALOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _pretrainingOptimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private SpiralNetwork<T>? _network;
    // The student's layers, in creation order, and the teacher: the same network outside the model's layers, built at the
    // first pre-training step and updated only as the student's moving average.
    private List<LayerBase<T>> _studentLayers = new();
    private SpiralNetwork<T>? _teacher;
    private readonly List<LayerBase<T>> _teacherLayers = new();
    private int _pretrainingStep;
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);
    internal SpiralNetwork<T>? Network => _network;
    internal SpiralNetwork<T>? Teacher => _teacher;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public SPIRAL(NeuralNetworkArchitecture<T> architecture, string modelPath, SPIRALOptions? options = null) : base(architecture) { _options = options ?? new SPIRALOptions(); Options = _options; _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = 0; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "en" }; InitializeLayers(); }
    public SPIRAL(NeuralNetworkArchitecture<T> architecture, SPIRALOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new SPIRALOptions(); Options = _options; _useNativeMode = true; _suppliedOptimizer = optimizer; _optimizer = optimizer
        // Ordinary training fine-tunes on transcripts; Pretrain builds the pre-training recipe's optimizer.
        ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, phase: AiDotNet.Attributes.TrainingPhase.FineTuning)
        ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.NumMels = 0; SupportedLanguages = new[] { "en" }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio: the encoder, projection and CTC layer give per-frame distributions, decoded greedily.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> logits;

        logits = Logits(features, training: false);

        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(logits);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = ResolveReturnTimestamps(includeTimestamps) ? ExtractSegments(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var logits = Logits(PreprocessAudio(audio), training: false); var (tokens, _) = CTCGreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(lang == detected ? primaryProb : otherProb); return result; }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("SPIRAL does not support streaming.");

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The paper network: two convolution + Transformer blocks over log-mel features, the projection head, the
        // predictor and the convolutional CTC classifier (Huang et al. 2022). It used to be a generic Transformer template
        // with no teacher, perturbation or contrastive loss (#2321).
        var layers = new List<LayerBase<T>>();
        _network = new SpiralNetwork<T>(Engine, layers, _options.ToConfiguration());
        _network.InitializePositionalConvolutions(Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 42));
        _studentLayers = layers;
        _teacher = null;
        _teacherLayers.Clear();
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        Layers.AddRange(layers);
    }

    // SPIRAL reads the raw waveform, truncated to the longest window a transcription covers.
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_network is null) return base.PreprocessAudio(rawAudio);
        int length = Math.Min(rawAudio.Length, SampleRate * _options.MaxAudioLengthSeconds);
        var waveform = new Tensor<T>(new[] { length });
        rawAudio.Data.Span.Slice(0, length).CopyTo(waveform.Data.Span);
        return waveform;
    }

    private double[] Samples(Tensor<T> waveform, int offset, int count)
    {
        var samples = new double[count];
        var span = waveform.Data.Span;
        for (int i = 0; i < count; i++) samples[i] = NumOps.ToDouble(span[offset + i]);
        return samples;
    }

    // CTC logits [frames, vocab] for a waveform [S], or [batch, frames, vocab] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input, bool training)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        if (input.Rank == 1) return Ctc(Samples(input, 0, input.Length), training);
        int rows = input.Shape[0], samples = input.Length / rows;
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
            outputs[r] = Ctc(Samples(input, r * samples, samples), training);
        return Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input, training: false); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input, training: true);

    private Tensor<T> Ctc(double[] waveform, bool training)
    {
        var network = _network ?? throw new InvalidOperationException("The native SPIRAL network is not initialized.");
        return network.CtcLogits(network.Encode(network.ToTensor(network.Features(waveform)), training, _random), training, _random);
    }

    /// <summary>
    /// The SPIRAL pre-training loss of waveforms <c>[S]</c> or <c>[batch, S]</c> (Huang et al. 2022, section 3.1), averaged
    /// over the batch: the student's predictor output on the SpecAugment-perturbed input against the teacher's projection of
    /// the clean input, padded at random at both ends, under the in-utterance contrastive loss of Equation 2.
    /// </summary>
    /// <param name="audio">The waveforms.</param>
    /// <param name="random">The perturbation, padding, dropout and distractor draws; the model's own generator when null.</param>
    public T PretrainingLoss(Tensor<T> audio, Random? random = null)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        return PretrainingLossTensor(audio, training: false, random ?? _random)[0];
    }

    private Tensor<T> PretrainingLossTensor(Tensor<T> audio, bool training, Random random)
    {
        var network = _network ?? throw new InvalidOperationException("Pre-training runs on the native SPIRAL network.");
        int rows = audio.Rank == 1 ? 1 : audio.Shape[0], samples = audio.Length / rows;
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var features = network.Features(Samples(audio, r * samples, samples));
            var teacherOutput = TeacherOutput(features, training, random);
            var student = network.Predictor(network.Project(network.Encode(network.ToTensor(Perturb(features, random)), training, random)), training);
            var loss = network.ContrastiveLoss(student, teacherOutput, _options.Negatives, _options.Temperature, random);
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(audio)), NumOps.FromDouble(1.0 / rows));
    }

    // The teacher's projection of the clean features with a random number of padding units (multiples of the encoder's
    // total stride, so outputs stay aligned) at each end; the padded positions are dropped (section 3.1).
    private Tensor<T> TeacherOutput(double[,] features, bool training, Random random)
    {
        var network = _network ?? throw new InvalidOperationException("Pre-training runs on the native SPIRAL network.");
        var teacher = EnsureTeacher(features);
        int frames = features.GetLength(0), bins = features.GetLength(1), stride = network.TotalStride;
        int most = Math.Max(0, Math.Min(_options.MaxTeacherPaddingUnits, frames / 2 / stride));
        int left = random.Next(most + 1), right = random.Next(most + 1);
        var padded = new double[frames + (left + right) * stride, bins];
        for (int t = 0; t < frames; t++)
            for (int b = 0; b < bins; b++) padded[left * stride + t, b] = features[t, b];
        using var _ = new NoGradScope<T>();
        var projected = teacher.Project(teacher.Encode(teacher.ToTensor(padded), training, random));
        int length = network.EncodedLength(frames);
        var kept = Engine.TensorSlice(projected, new[] { left, 0 }, new[] { length, projected.Shape[1] });
        // A constant copy: the teacher receives no gradient (it moves only by Equation 1).
        return new Tensor<T>(kept._shape, new Vector<T>(kept.ToArray()));
    }

    // Adaptive SpecAugment (section 3.3): a proportion of the time steps start masks of TimeMaskLength frames filled with
    // N(0, 1) noise, and a proportion of the bins start masks of FrequencyMaskLength bins filled with zeros.
    private double[,] Perturb(double[,] features, Random random)
    {
        int frames = features.GetLength(0), bins = features.GetLength(1);
        var perturbed = (double[,])features.Clone();
        foreach (int start in Starts(frames, _options.TimeMaskProbability, random))
            for (int t = start; t < Math.Min(frames, start + _options.TimeMaskLength); t++)
                for (int b = 0; b < bins; b++)
                {
                    double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                    perturbed[t, b] = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
                }
        foreach (int start in Starts(bins, _options.FrequencyMaskProbability, random))
            for (int b = start; b < Math.Min(bins, start + _options.FrequencyMaskLength); b++)
                for (int t = 0; t < frames; t++) perturbed[t, b] = 0.0;
        return perturbed;
    }

    // ⌊p·n + U(0, 1)⌋ distinct start indices drawn uniformly from n: a proportion p of the steps, rounded at random.
    private static IEnumerable<int> Starts(int n, double proportion, Random random)
    {
        int count = Math.Min(n, (int)Math.Floor(proportion * n + random.NextDouble()));
        var order = Enumerable.Range(0, n).ToArray();
        for (int i = 0; i < count; i++)
        {
            int j = i + random.Next(n - i);
            (order[i], order[j]) = (order[j], order[i]);
            yield return order[i];
        }
    }

    // The teacher starts as a copy of the student. Its lazily sized layers are resolved by one forward first.
    private SpiralNetwork<T> EnsureTeacher(double[,] features)
    {
        if (_teacher is not null) return _teacher;
        var network = _network ?? throw new InvalidOperationException("Pre-training runs on the native SPIRAL network.");
        var teacher = new SpiralNetwork<T>(Engine, _teacherLayers, _options.ToConfiguration());
        using (new NoGradScope<T>())
        {
            var x = network.ToTensor(features);
            teacher.Project(teacher.Encode(x, false, _random));
            network.Project(network.Encode(x, false, _random));
        }
        _teacher = teacher;
        UpdateTeacher(0.0);
        return teacher;
    }

    // Equation 1: θ' ← α θ' + (1 − α) θ, layer by layer (the predictor and classifier, which the teacher never runs, are
    // skipped while their sizes are unresolved).
    private void UpdateTeacher(double alpha)
    {
        for (int i = 0; i < _teacherLayers.Count; i++)
        {
            var student = _studentLayers[i].GetParameters();
            var teacher = _teacherLayers[i].GetParameters();
            if (student.Length == 0 || teacher.Length != student.Length) continue;
            var blended = new T[student.Length];
            for (int k = 0; k < blended.Length; k++)
                blended[k] = NumOps.FromDouble(alpha * NumOps.ToDouble(teacher[k]) + (1 - alpha) * NumOps.ToDouble(student[k]));
            _teacherLayers[i].SetParameters(new Vector<T>(blended));
        }
    }

    /// <summary>
    /// One SPIRAL pre-training step (section 3.1) on waveforms <c>[S]</c> or <c>[batch, S]</c>: a gradient step on the
    /// student, then the teacher's moving-average update with α on the cosine schedule from TeacherMomentumStart to
    /// TeacherMomentumEnd over PretrainingSteps (section 4.2).
    /// </summary>
    /// <param name="audio">The unlabeled waveforms.</param>
    /// <returns>The step's loss.</returns>
    public T Pretrain(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (_network is null) throw new NotSupportedException("Pre-training runs on the native SPIRAL network.");
        // A caller's optimizer serves both phases; otherwise the paper's pre-training recipe (section 4.2).
        _pretrainingOptimizer ??= _suppliedOptimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, phase: AiDotNet.Attributes.TrainingPhase.PreTraining)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        var loss = TrainWithCustomLoss(audio, _ => PretrainingLossTensor(audio, training: true, _random), _pretrainingOptimizer);
        double progress = Math.Min(1.0, (double)_pretrainingStep++ / Math.Max(1, _options.PretrainingSteps));
        double alpha = _options.TeacherMomentumEnd
            - (_options.TeacherMomentumEnd - _options.TeacherMomentumStart) * (Math.Cos(Math.PI * progress) + 1) / 2;
        UpdateTeacher(alpha);
        return loss;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "SPIRAL-Native" : "SPIRAL-ONNX", Description = "SPIRAL: perturbation-invariant teacher-student pre-training with a convolutional CTC classifier (Huang et al., 2022)", FeatureCount = 1, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };



    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames && tokens.Count < _options.MaxTextLength; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } }; }
    private string ClassifyLanguageFromTokens(List<int> tokens) { if (tokens.Count == 0) return _options.Language; int cjkCount = 0, latinCount = 0; foreach (var t in tokens) { if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++; else if (t >= 0x41 && t <= 0x7A) latinCount++; } if (cjkCount > latinCount && SupportedLanguages.Contains("zh")) return "zh"; return _options.Language; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SPIRAL<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
