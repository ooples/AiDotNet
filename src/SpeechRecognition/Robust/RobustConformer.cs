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

namespace AiDotNet.SpeechRecognition.Robust;

/// <summary>
/// Noise-robust wav2vec 2.0 (Wang et al. 2021): contrastive pre-training continued on noisy speech with an auxiliary
/// module that reconstructs the clean speech, then CTC fine-tuning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The model reads raw 16 kHz audio. A stack of convolutions turns it into one feature vector
/// every 20 ms, and a Transformer gives each vector context from the whole utterance. To make those vectors survive
/// background noise, pre-training feeds noisy audio and asks a small extra network to rebuild the clean audio from them;
/// that extra network is dropped afterwards. For recognition a linear layer reads each vector as character
/// probabilities, and collapsing repeats and blanks gives the transcript.</para>
/// <para>
/// Section 2: the wav2vec 2.0 encoder (Baevski et al. 2020), whose contrastive loss L_c and diversity loss L_d are kept,
/// plus a reconstruction module on the contextual representations C: a two-layer BLSTM with layer normalization after
/// each layer and a CNN decoder symmetric to the feature encoder, trained by the mean absolute error L_r to the clean
/// waveform (Equation 3). Continual pre-training (<see cref="ContinualPretrain"/>) minimizes L = L_c + 0.1 L_d + 0.1 L_r
/// (Equation 4) over noisy input; wav2vec 2.0's feature penalty is kept as in its reference objective. Fine-tuning
/// (<c>Train</c>) adds a linear layer for character predictions and minimizes the CTC loss (section 3.3); the
/// reconstruction module is not used in fine-tuning or inference. The defaults are section 3.2's BASE model. The BLSTM
/// width is not stated; the default makes its output the encoder width. The class keeps its earlier name, from when it
/// was a generic Conformer template.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 128);
/// var model = new RobustConformer&lt;float&gt;(architecture);
/// model.ContinualPretrain(noisyWaveforms, cleanWaveforms);
/// model.Train(waveform, transcriptCodePoints);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Improving Noise Robustness of Contrastive Speech Representation Learning with Speech Reconstruction", "https://arxiv.org/abs/2110.15430", Year = 2021, Authors = "Heming Wang, Yao Qian, Xiaofei Wang, Yiming Wang, Chengyi Wang, Shujie Liu, Takuya Yoshioka, Jinyu Li, DeLiang Wang")]
[ArchitectureFromPaper("https://arxiv.org/abs/2006.11477",
    "The encoder is wav2vec 2.0 BASE (Wang et al. 2021, section 3.2, continuing from wav2vec_small.pt); the paper adds a "
    + "reconstruction module for pre-training, so it runs on the wav2vec 2.0 network.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4,
                Phase = TrainingPhase.FineTuning,
                Source = "Wang et al. 2021, Sec. 3.3: models are trained with an Adam optimizer at a "
                        + "learning rate of 2e-4, optimized by a CTC loss for 20k steps. The model keeps "
                        + "its own optimizer and is verified against this record rather than built from "
                        + "it: 2e-4 over 20k steps is a paper-scale recipe and leaves the loss flat over "
                        + "the handful of steps a conformance run performs.")]
public partial class RobustConformer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The CTC vocabulary, blank (0) included.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly RobustConformerOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private NoiseRobustWav2Vec2Network<T>? _network;

    /// <inheritdoc />
    /// <remarks>True when the encoder's LayerDrop is positive: a skipped layer is not stepped that step.</remarks>
    private protected override bool TrainingForwardSkipsLayersPerStep => _network is not null && _network.Stack.Network.Configuration.LayerDrop > 0;
    private Random _random;
    [AiDotNet.Attributes.Scratch]
    private bool _inContinualPretrain;
    internal NoiseRobustWav2Vec2Network<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => false;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    /// <summary>The last continual pre-training step's reconstruction loss L_r.</summary>
    internal double LastReconstructionLoss { get; private set; }

    public RobustConformer(NeuralNetworkArchitecture<T> architecture, string modelPath, RobustConformerOptions? options = null) : base(architecture)
    {
        _options = options ?? new RobustConformerOptions(); Options = _options; _useNativeMode = false;
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public RobustConformer(NeuralNetworkArchitecture<T> architecture, RobustConformerOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new RobustConformerOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this));
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        base.SampleRate = _options.SampleRate; base.NumMels = 0;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <summary>Transcribes audio: the encoder and the CTC layer with greedy decoding.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for RobustConformer.");
        var waveform = PreprocessAudio(audio);
        Tensor<T> logits;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>()) logits = Logits(waveform, training: false);
        var frames = Engine.Reshape(logits, new[] { logits.Length / logits.Shape[^1], logits.Shape[^1] });
        var tokens = Audio.Foundations.ConformerAsrNetwork<T>.GreedyCtc(frames);
        if (tokens.Count > _options.MaxTextLength) tokens.RemoveRange(_options.MaxTextLength, tokens.Count - _options.MaxTextLength);
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens) if (t > 0 && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t));
        return new TranscriptionResult<T>
        {
            Text = sb.ToString().Trim(), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("RobustConformer does not support streaming.");

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The paper network (Figure 1). It used to be a generic Conformer template without the paper's encoder or
        // reconstruction module.
        _network = new NoiseRobustWav2Vec2Network<T>(Engine, _options.ToConfiguration(), _options.ToPretrainingConfiguration(), _options.VocabSize,
            _options.ReconstructionHidden, _options.ReconstructionWeight, _options.Seed ?? 42);
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        Layers.AddRange(_network.Layers);
    }

    private protected override int? NativeWaveformMaxAudioLengthSeconds => _network is null ? null : _options.MaxAudioLengthSeconds;

    // CTC logits [frames, vocabulary] for a waveform [S], or [batch, frames, vocabulary] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input, bool training)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        return _network.Stack.Run(input, training, _random, freezeFeatureEncoder: training && _options.FreezeFeatureEncoder, _options.FinalDropout);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input, training: false); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input, training: true);

    protected override AiDotNet.Models.Parameters.ParameterSlotRole ResolveLayerParameterRole(int layerIndex, AiDotNet.Models.Parameters.ParameterSlotRole declared)
        => declared == AiDotNet.Models.Parameters.ParameterSlotRole.Trainable && !_inContinualPretrain && _options.FreezeFeatureEncoder
            && _network is not null && _network.Stack.IsFeatureEncoderLayer(Layers[layerIndex])
            ? AiDotNet.Models.Parameters.ParameterSlotRole.Frozen
            : declared;

    /// <summary>
    /// One continual pre-training step (section 2.3) on noisy waveforms <c>[B, S]</c> (or <c>[S]</c>) and their clean
    /// references of the same shape: L = L_c + λ1 L_d + λ2 L_r (Equation 4).
    /// </summary>
    public void ContinualPretrain(Tensor<T> noisy, Tensor<T> clean)
    {
        ThrowIfDisposed();
        var network = _network ?? throw new InvalidOperationException("Pre-training runs on the native network.");
        if (noisy.Length != clean.Length) throw new ArgumentException("The clean references must match the noisy waveforms.", nameof(clean));
        var x = Rows(noisy);
        var y = Engine.Reshape(clean, x.Shape.ToArray());
        var net = network.Stack.Network;
        var c = network.Pretraining.Configuration;
        var stepMask = c.MaskProbability > 0
            ? net.DrawMask(x.Shape[0], net.Frames(x.Shape[1]), c.MaskProbability, _random)
            : new bool[x.Shape[0], net.Frames(x.Shape[1])];
        var stepNegatives = network.Pretraining.SampleNegatives(stepMask, _random);
        using var outputs = AiDotNet.Tensors.Engines.Autodiff.RetainedOutputCapture<T>.Begin();
        _inContinualPretrain = true;
        try
        {
            TrainWithCustomObjective(x, y, (stepX, stepY) =>
            {
                var (total, reconstruction) = network.ContinualPretrainingLoss(stepX, stepY, training: true, _random, stepMask, stepNegatives);
                if (AiDotNet.Tensors.Engines.Autodiff.GradientRecording.IsRecording<T>())
                    AiDotNet.Tensors.Engines.Autodiff.GradientRecording.RetainOutput(
                        "robust-conformer-reconstruction",
                        Engine.TensorAddScalar(Engine.StopGradient(reconstruction), NumOps.Zero));
                return total;
            }, _optimizer);
        }
        finally
        {
            _inContinualPretrain = false;
        }
        outputs.Complete();
        LastReconstructionLoss = NumOps.ToDouble(outputs.Get("robust-conformer-reconstruction")[0]);
        network.Pretraining.AfterUpdate();
    }

    private Tensor<T> Rows(Tensor<T> waveform) => waveform.Rank == 1 ? Engine.Reshape(waveform, new[] { 1, waveform.Length })
        : waveform.Rank == 2 ? waveform : Engine.Reshape(waveform, new[] { waveform.Shape[0], waveform.Length / waveform.Shape[0] });

    private int[] Labels(Tensor<T> target) =>
        ParseCtcOrTokenTargetLabels(target, _options.VocabSize, Audio.Foundations.ConformerAsrNetwork<T>.GreedyCtc);

    private Tensor<T> CtcLoss(Tensor<T> logits, int[] labels)
    {
        int frames = logits.Shape[0];
        return Engine.TensorCTCLoss(Engine.Reshape(Engine.TensorLogSoftmax(logits, axis: -1), new[] { frames, 1, _options.VocabSize }),
            new Tensor<int>(labels, new[] { labels.Length }), new[] { frames }, new[] { labels.Length }, 0);
    }

    /// <inheritdoc />
    /// <remarks>CTC fine-tuning (section 3.3): the CTC loss of the target characters.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target) =>
        TrainNativeWaveformBatch(Rows(input), target, _network, _optimizer, (_, row) => Labels(row),
            (_, wave, rowLabels) => CtcLoss(Logits(wave, training: true), rowLabels));

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The CTC loss <see cref="TryApplyLearningRule"/> minimizes, in inference mode, averaged over the batch.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) =>
        EvaluateNativeWaveformBatchObjective(Rows(input), target, _network, (_, row) => Labels(row),
            (_, wave, rowLabels) => CtcLoss(Logits(wave, training: false), rowLabels));

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "RobustConformer-Native" : "RobustConformer-ONNX", Description = "Noise-robust wav2vec 2.0 with speech reconstruction (Wang et al. 2021) + CTC", FeatureCount = 1, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RobustConformer<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
