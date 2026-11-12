using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>
/// Medical-conversation ASR: Chiu et al. 2018's end-to-end recognizers for doctor-patient dialogue, offering both arms of
/// the paper's comparison, Listen-Attend-Spell and CTC.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Chiu, Tripathi, Chou, Co, Jaitly, Jaunzeikare, Kannan, Nguyen, Sak, Sankar, Tansuwan, Wan, Wu and Zhang, "Speech
/// recognition for medical conversations" (Interspeech 2018, arXiv:1711.07274). The paper is a COMPARISON: "We explored
/// both CTC and LAS systems for building speech recognition models."
/// </para>
/// <para>
/// Both arms read 80 filterbank channels up to 3800 Hz from 25 ms frames every 10 ms, stacked three at a time with a
/// stride of three (240 values every 30 ms, section 4).
/// </para>
/// <list type="bullet">
/// <item><description><b>LAS</b> (the default, section 4.2 and 5.2): six bidirectional LSTM layers of 768 units per
/// direction, a multi-headed attention, and three unidirectional 768-unit LSTM layers predicting graphemes, trained with
/// scheduled sampling ramped linearly to 30%. Its conclusion sets the default: "The LAS was more resilient to noisy data
/// and CTC required more data clean up."</description></item>
/// <item><description><b>CTC</b> (section 4.1 and 5.1): a stack of LSTM layers under a softmax over 8K context-dependent
/// phones and the blank: five bidirectional layers of 350 units per direction (<see cref="MedicalASROptions.ForCtc"/>),
/// or five unidirectional 700-unit layers with an output delay of ten steps.</description></item>
/// </list>
/// <para>
/// <b>For Beginners:</b> Two ways to turn sound into text. One (CTC) labels each slice of audio and is fast but needs
/// clean recordings. The other (LAS) reads the whole recording first, then writes the sentence letter by letter while
/// looking back at the audio, which copes better when people talk over each other or trail off, as they do in real
/// consultations. This model gives you both, and defaults to the one the paper found more robust.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SequenceToSequence,
///     inputSize: 16000, outputSize: 128);
///
/// // Defaults to LAS, the paper's more noise-resilient arm.
/// var las = new MedicalASR&lt;double&gt;(architecture);
///
/// // The paper's bidirectional CTC model.
/// var ctc = new MedicalASR&lt;double&gt;(architecture, MedicalASROptions.ForCtc());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Speech recognition for medical conversations",
    "https://arxiv.org/abs/1711.07274",
    Year = 2018,
    Authors = "Chung-Cheng Chiu, Anshuman Tripathi, Katherine Chou, Chris Co, Navdeep Jaitly, " +
              "Diana Jaunzeikare, Anjuli Kannan, Patrick Nguyen, Hasim Sak, Ananth Sankar, " +
              "Justin Tansuwan, Nathan Wan, Yonghui Wu, Xuedong Zhang")]
public partial class MedicalASR<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly MedicalASROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private MedicalAsrNetwork<T>? _network;
    private Random _random;
    private int _trainingSteps;
    internal MedicalAsrNetwork<T>? Network => _network;

    /// <summary>Gets which arm of the paper's comparison this instance is built as.</summary>
    public MedicalAsrDecoderType DecoderType => _options.DecoderType;

    /// <summary>Gets how many 10 ms frames one encoder step covers (the frame stacking stride).</summary>
    public int TimeReductionFactor => Math.Max(1, _options.StackedFrames);

    /// <summary>Gets the probability with which the next training step feeds the LAS decoder its own previous prediction:
    /// a linear ramp from 0 to <see cref="MedicalASROptions.ScheduledSamplingProbability"/>, then constant.</summary>
    public double CurrentSamplingProbability =>
        _options.ScheduledSamplingRampSteps <= 0 ? _options.ScheduledSamplingProbability
            : _options.ScheduledSamplingProbability * Math.Min(1.0, _trainingSteps / (double)_options.ScheduledSamplingRampSteps);

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedLanguages { get; }

    /// <inheritdoc />
    public bool SupportsStreaming => false;

    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    /// <summary>Creates a MedicalASR model that runs a pretrained ONNX export.</summary>
    public MedicalASR(NeuralNetworkArchitecture<T> architecture, string modelPath, MedicalASROptions? options = null) : base(architecture)
    {
        _options = options ?? new MedicalASROptions();
        Options = _options;
        _useNativeMode = false;
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        base.SampleRate = _options.SampleRate;
        base.NumMels = 0;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language };
        InitializeLayersCore();
    }

    /// <summary>Creates a trainable MedicalASR model (LAS by default).</summary>
    public MedicalASR(NeuralNetworkArchitecture<T> architecture, MedicalASROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new MedicalASROptions();
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        base.SampleRate = _options.SampleRate;
        base.NumMels = 0;
        SupportedLanguages = new[] { _options.Language };
        InitializeLayersCore();
    }

    /// <inheritdoc />
    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _network = null;
        Layers.Clear();
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The paper's networks (section 4). They used to be Conformer templates, with LAS a pair of dense layers over an
        // averaged "pyramid" trained by mean squared error.
        var layers = new List<NeuralNetworks.Layers.LayerBase<T>>();
        _network = new MedicalAsrNetwork<T>(Engine, layers, _options.ToConfiguration());
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        _trainingSteps = 0;
        Layers.AddRange(layers);
    }

    // The recognizer reads the raw waveform, truncated to the longest window a transcription covers.
    /// <inheritdoc />
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_network is null) return rawAudio;
        int length = Math.Min(rawAudio.Length, SampleRate * _options.MaxAudioLengthSeconds);
        var waveform = new Tensor<T>(new[] { length });
        rawAudio.Data.Span.Slice(0, length).CopyTo(waveform.Data.Span);
        return waveform;
    }

    private MedicalAsrNetwork<T> RequireNetwork() =>
        _network ?? throw new NotSupportedException("This operation runs on the native paper network.");

    /// <inheritdoc />
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for MedicalASR.");
        List<int> tokens;
        if (_network is not null && _options.DecoderType == MedicalAsrDecoderType.ListenAttendSpell)
        {
            using var _ = new NoGradScope<T>();
            tokens = _network.Greedy(_network.Encode(_network.Features(PreprocessAudio(audio))), _options.MaxTextLength).Tokens;
        }
        else
        {
            var logits = PredictCore(audio);
            tokens = Audio.Foundations.ConformerAsrNetwork<T>.GreedyCtc(Engine.Reshape(logits, new[] { logits.Length / logits.Shape[^1], logits.Shape[^1] }));
        }
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    // LAS graphemes are code points; CTC units are written by their inventory names (or indices), space-separated.
    private string Text(List<int> tokens)
    {
        if (_options.DecoderType == MedicalAsrDecoderType.Ctc || _network is null)
            return string.Join(" ", tokens.Select(t => t < _options.PhoneInventory.Length ? _options.PhoneInventory[t] : t.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens) if (t > 0 && t <= 0x10FFFF && (t < 0xD800 || t > 0xDFFF)) sb.Append(char.ConvertFromUtf32(t));
        return sb.ToString().Trim();
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null,
        CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <inheritdoc />
    public string DetectLanguage(Tensor<T> audio) { ThrowIfDisposed(); return _options.Language; }

    /// <summary>The paper's recognizers are monolingual (English medical conversations): the configured language with
    /// probability 1.</summary>
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return new Dictionary<string, T> { [_options.Language] = NumOps.One };
    }

    /// <inheritdoc />
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) =>
        throw new NotSupportedException("MedicalASR does not support streaming.");

    // Per waveform row: CTC logits [steps, V], or LAS greedy-decoding logits [MaxTextLength, V].
    private Tensor<T> Logits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        var rows = Rows(input);
        var outputs = new Tensor<T>[rows.Length];
        for (int r = 0; r < rows.Length; r++)
        {
            var encoded = _network.Encode(_network.Features(rows[r]));
            outputs[r] = _options.DecoderType == MedicalAsrDecoderType.Ctc ? _network.CtcLogits(encoded) : _network.Greedy(encoded, _options.MaxTextLength).Logits;
        }
        return input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    private Tensor<T>[] Rows(Tensor<T> input)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var flat = Engine.Reshape(input, new[] { rows, samples });
        var result = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
            result[r] = PreprocessAudio(Engine.Reshape(Engine.TensorSlice(flat, new[] { r, 0 }, new[] { 1, samples }), new[] { samples }));
        return result;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input); }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input);

    // The transcript a training target names: unit ids [length], or one distribution per row [n, V] read as the most likely
    // unit per row (for CTC, repeats merged and blanks removed; for LAS, up to the first boundary).
    private int[] Labels(Tensor<T> target)
    {
        int v = _options.VocabSize, width = target.Rank >= 2 ? target.Shape[^1] : 0;
        if (width == v)
        {
            var frames = Engine.Reshape(target, new[] { target.Length / width, width });
            if (_options.DecoderType == MedicalAsrDecoderType.Ctc) return Audio.Foundations.ConformerAsrNetwork<T>.GreedyCtc(frames).ToArray();
            var labels = new List<int>();
            for (int i = 0; i < frames.Shape[0]; i++)
            {
                int best = 0;
                for (int k = 1; k < v; k++) if (NumOps.GreaterThan(frames[i, k], frames[i, best])) best = k;
                if (best == MedicalAsrNetwork<T>.Boundary) break;
                labels.Add(best);
            }
            return labels.ToArray();
        }
        var ids = new int[target.Length];
        var span = target.Data.Span;
        for (int i = 0; i < ids.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value <= 0 || value >= v) throw new ArgumentException($"Unit {value} is the blank or boundary, or outside the {v}-unit inventory.", nameof(target));
            ids[i] = (int)value;
        }
        return ids;
    }

    private (Tensor<T> Waveform, int[] Labels)[] Batch(Tensor<T> input, Tensor<T> target)
    {
        var rows = Rows(input);
        int perRow = target.Length / rows.Length, width = target.Rank >= 2 ? target.Shape[^1] : 0;
        bool distributions = width == _options.VocabSize;
        var batch = new (Tensor<T>, int[])[rows.Length];
        for (int r = 0; r < rows.Length; r++)
        {
            var row = rows.Length == 1 ? target : Engine.Reshape(Engine.TensorSlice(Engine.Reshape(target, new[] { rows.Length, perRow }), new[] { r, 0 }, new[] { 1, perRow }),
                distributions ? new[] { perRow / width, width } : new[] { perRow });
            batch[r] = (rows[r], Labels(row));
        }
        return batch;
    }

    /// <summary>One step of the paper's training: CTC, or LAS cross-entropy with scheduled sampling.</summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        var batch = Batch(input, target);
        double sampling = CurrentSamplingProbability;
        TrainWithCustomLoss(input, _ =>
        {
            Tensor<T>? total = null;
            foreach (var (waveform, labels) in batch)
            {
                var loss = network.Loss(network.Features(waveform), labels, sampling, _random);
                total = total is null ? loss : Engine.TensorAdd(total, loss);
            }
            return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / batch.Length));
        }, _optimizer);
        _trainingSteps++;
        return true;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var network = RequireNetwork();
        using var _ = new NoGradScope<T>();
        var batch = Batch(input, target);
        double total = 0;
        foreach (var (waveform, labels) in batch) total += NumOps.ToDouble(network.Loss(network.Features(waveform), labels)[0]);
        return NumOps.FromDouble(total / batch.Length);
    }

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = _useNativeMode ? "MedicalASR-Native" : "MedicalASR-ONNX",
        Description = _options.DecoderType == MedicalAsrDecoderType.Ctc
            ? "LSTM CTC recognizer for medical conversations (Chiu et al. 2018)"
            : "Bidirectional LSTM Listen-Attend-Spell recognizer for medical conversations (Chiu et al. 2018)",
        FeatureCount = _options.NumMels,
        Complexity = _options.NumEncoderLayers,
        AdditionalInfo = BaseAudioMetadataInfo()
    };

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MedicalASR<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }
}
