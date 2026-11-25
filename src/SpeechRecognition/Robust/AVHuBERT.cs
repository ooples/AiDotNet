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

namespace AiDotNet.SpeechRecognition.Robust;

/// <summary>
/// AV-HuBERT: a Transformer over fused audio and lip-video features, pre-trained by masked multimodal cluster prediction,
/// with a CTC projection for recognition from audio, lips or both.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction" (Shi et al., Meta, ICLR 2022)</item></list></para>
/// <para><b>For Beginners:</b> AV-HuBERT watches the speaker's mouth and listens at the same time. During pre-training it
/// hides stretches of the audio and the video and learns to predict cluster labels of what was hidden; sometimes it gets
/// only the audio or only the video, so afterwards it can recognize speech from either, which makes it robust to noise
/// and able to lip-read.</para>
/// <para>
/// The network is the BASE model (section 4, appendix B.3): a linear audio encoder over 104-dimensional stacked log
/// filterbanks, a modified ResNet-18 (5x7x7 3-D convolution front, PReLU) over 88x88 grayscale mouth crops, both at 25 Hz,
/// fused by concatenation with modality dropout (Equation 2), and the pre-norm wav2vec 2.0 Transformer (12 layers, 768
/// wide). <see cref="Pretrain"/> runs the masked prediction of section 3 against cluster targets the caller supplies (the
/// paper's k-means over MFCC, then over the previous iteration's features); recognition puts a CTC projection on the
/// Transformer (section 3.3). <see cref="Transcribe"/> uses the audio alone, <see cref="TranscribeLips"/> the video alone
/// and <see cref="TranscribeAudioVisual"/> both.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var audio = Tensor&lt;double&gt;.CreateRandom(1, 16000);   // 1 s at 16 kHz
/// var lips = Tensor&lt;double&gt;.CreateRandom(25, 88, 88);  // the matching 25 mouth-region frames
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 46);
/// var model = new AVHuBERT&lt;double&gt;(architecture);
/// var result = model.TranscribeAudioVisual(audio, lips);   // lips: [frames, 88, 88] at 25 frames a second
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction", "https://arxiv.org/abs/2201.02184", Year = 2022, Authors = "Shi et al.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.002, WarmupFraction = 0.08,
                MinLearningRate = 0, Schedule = LearningRateSchedulerType.LinearWarmup,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear, Phase = TrainingPhase.PreTraining,
                Source = "Shi et al. 2022, appendix B.4: pre-training uses Adam, warming the learning rate up over the "
                        + "first 8% of updates to a peak of 0.002 and then decaying it linearly.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.001, SearchedValues = [0.10, 0.30, 0.50],
                Provenance = RecipeProvenance.Searched, Phase = TrainingPhase.FineTuning,
                Source = "Shi et al. 2022, appendix B.4: fine-tuning uses Adam with the learning rate warmed up for the "
                        + "first P% of updates to a peak of 0.001 and decayed linearly, P tuned among {10, 30, 50}; the "
                        + "candidates are the warm-up fractions recorded.")]
public partial class AVHuBERT<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    /// <inheritdoc />
    /// <remarks>The CTC projection's width: the cluster head is used only by <see cref="Pretrain"/>.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly AVHuBERTOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _pretrainingOptimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private AvHubertNetwork<T>? _network;
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);
    internal AvHubertNetwork<T>? Network => _network;
    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => false;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    public AVHuBERT(NeuralNetworkArchitecture<T> architecture, string modelPath, AVHuBERTOptions? options = null) : base(architecture) { _options = options ?? new AVHuBERTOptions(); Options = _options; _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = 0; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "en" }; InitializeLayersCore(); }
    public AVHuBERT(NeuralNetworkArchitecture<T> architecture, AVHuBERTOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new AVHuBERTOptions(); Options = _options; _useNativeMode = true; _suppliedOptimizer = optimizer; _optimizer = optimizer
        // Ordinary training fine-tunes on transcripts; Pretrain builds the pre-training recipe's optimizer.
        ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, phase: AiDotNet.Attributes.TrainingPhase.FineTuning)
        ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.NumMels = 0; SupportedLanguages = new[] { "en" }; InitializeLayersCore(); }

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

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    /// <inheritdoc/>
    public string DetectLanguage(Tensor<T> audio) { var logits = Logits(PreprocessAudio(audio), training: false); var (tokens, _) = CTCGreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    /// <inheritdoc/>
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; T primaryValue = NumOps.FromDouble(primaryProb); T otherValue = NumOps.FromDouble(otherProb); foreach (var lang in SupportedLanguages) result[lang] = lang == detected ? primaryValue : otherValue; return result; }
    /// <inheritdoc />
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("AVHuBERT does not support streaming.");

    // The constructors call the private body directly: a virtual call from a base constructor would reach an override
    // before the derived constructor has run.
    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The paper network: the linear audio encoder, the ResNet-18 lip encoder, concatenation fusion, the pre-norm
        // Transformer, the cluster head and the CTC projection (Shi et al. 2022). It used to be a generic audio-only
        // Transformer template with no visual stream (#2321).
        var layers = new List<LayerBase<T>>();
        _network = new AvHubertNetwork<T>(Engine, layers, _options.ToConfiguration());
        _network.Encoder.InitializePositionalConvolution(Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 42));
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        Layers.AddRange(layers);
    }

    private protected override int? NativeWaveformMaxAudioLengthSeconds => _network is null ? null : _options.MaxAudioLengthSeconds;

    // CTC logits [frames, vocab] for a waveform [S], or [batch, frames, vocab] for [batch, S]: the audio stream alone.
    private Tensor<T> Logits(Tensor<T> input, bool training) =>
        ForwardNativeWaveformBatch(input, _network, (_, wave) => Recognize(wave, null, training));

    // CTC logits from the audio, the lips [frames, H, W], or both; a missing stream is zeros (Equation 2).
    private Tensor<T> Recognize(Tensor<T>? waveform, Tensor<T>? lips, bool training)
    {
        var network = _network ?? throw new InvalidOperationException("The native AV-HuBERT network is not initialized.");
        var audio = waveform is null ? null : network.EncodeAudio(network.AudioFeatureTensor(waveform), null);
        var video = lips is null ? null : network.EncodeVideo(RequireLips(lips), null, _random, training);
        return network.CtcLogits(network.Encode(network.Fuse(audio, video, training, _random), training, _random));
    }


    private Tensor<T> RequireLips(Tensor<T> lips)
    {
        if (lips.Rank != 3) throw new ArgumentException("Lips must be [frames, height, width] grayscale mouth crops.", nameof(lips));
        if (lips.Shape[1] != _options.LipSize || lips.Shape[2] != _options.LipSize)
            throw new ArgumentException($"Lips must be {_options.LipSize}x{_options.LipSize} mouth crops (LipSize).", nameof(lips));
        return lips;
    }

    private TranscriptionResult<T> Decode(Tensor<T> logits, string? language, double duration, bool? includeTimestamps)
    {
        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(logits);
        var text = TokensToText(tokens);
        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = ResolveReturnTimestamps(includeTimestamps) ? ExtractSegments(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    /// <summary>Transcribes from the audio and the lip video together.</summary>
    /// <param name="audio">The 16 kHz waveform.</param>
    /// <param name="lips">Grayscale mouth crops <c>[frames, height, width]</c> at 25 frames a second.</param>
    /// <param name="language">The language to report.</param>
    /// <param name="includeTimestamps">Whether to return segments.</param>
    public TranscriptionResult<T> TranscribeAudioVisual(Tensor<T> audio, Tensor<T> lips, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (_network is null) throw new NotSupportedException("Audio-visual recognition runs on the native AV-HuBERT network.");
        var waveform = PreprocessAudio(audio);
        return Decode(Recognize(waveform, lips, training: false), language, (double)waveform.Length / SampleRate, includeTimestamps);
    }

    /// <summary>Transcribes from the lip video alone (lip reading).</summary>
    /// <param name="lips">Grayscale mouth crops <c>[frames, height, width]</c> at 25 frames a second.</param>
    /// <param name="language">The language to report.</param>
    /// <param name="includeTimestamps">Whether to return segments.</param>
    public TranscriptionResult<T> TranscribeLips(Tensor<T> lips, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (_network is null) throw new NotSupportedException("Lip reading runs on the native AV-HuBERT network.");
        return Decode(Recognize(null, lips, training: false), language, RequireLips(lips).Shape[0] / 25.0, includeTimestamps);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input, training: false); }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input, training: true);

    /// <summary>
    /// The masked multimodal cluster prediction loss of one utterance (Shi et al. 2022, section 3): spans of the audio and
    /// the video are masked at the input, one modality is dropped for the whole sequence with probability ModalityDropout,
    /// and the cross-entropy of the cosine cluster logits against <paramref name="clusterTargets"/> is averaged over the
    /// masked frames, plus FeaturePenaltyWeight times the mean squared fused features.
    /// </summary>
    /// <param name="audio">The 16 kHz waveform.</param>
    /// <param name="lips">Grayscale mouth crops <c>[frames, height, width]</c> at 25 frames a second.</param>
    /// <param name="clusterTargets">One cluster index per 25 Hz frame.</param>
    /// <param name="random">The mask and modality-dropout draws; the model's own generator when null.</param>
    public T PretrainingLoss(Tensor<T> audio, Tensor<T> lips, int[] clusterTargets, Random? random = null)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        return PretrainingLossTensor(audio, lips, clusterTargets, training: false, random ?? _random)[0];
    }

    private Tensor<T> PretrainingLossTensor(Tensor<T> audio, Tensor<T> lips, int[] clusterTargets, bool training, Random random)
    {
        var network = _network ?? throw new InvalidOperationException("Pre-training runs on the native AV-HuBERT network.");
        if (clusterTargets is null) throw new ArgumentNullException(nameof(clusterTargets));
        var features = network.AudioFeatureTensor(audio);
        return PretrainingLossFromFeaturesTensor(features, lips, clusterTargets, training, random);
    }

    private Tensor<T> PretrainingLossFromFeaturesTensor(Tensor<T> features, Tensor<T> lips, int[] clusterTargets, bool training, Random random)
    {
        var network = _network ?? throw new InvalidOperationException("Pre-training runs on the native AV-HuBERT network.");
        if (clusterTargets is null) throw new ArgumentNullException(nameof(clusterTargets));
        int frames = Math.Min(Math.Min(features.Shape[0], RequireLips(lips).Shape[0]), clusterTargets.Length);
        if (frames < 1) throw new ArgumentException("The audio, the lips and the targets share no frames.");
        int numClusters = _options.NumClusters;
        for (int t = 0; t < frames; t++)
        {
            if (clusterTargets[t] < 0 || clusterTargets[t] >= numClusters)
                throw new ArgumentOutOfRangeException(nameof(clusterTargets), $"Target {clusterTargets[t]} is outside the {numClusters} clusters.");
        }

        int width = network.AudioWidth;
        var cut = Engine.TensorSlice(features, new[] { 0, 0 }, new[] { frames, features.Shape[1] });
        var video = Engine.TensorSlice(lips, new[] { 0, 0, 0 }, new[] { frames, lips.Shape[1], lips.Shape[2] });

        int packetCols = width + 1 + numClusters + 2;
        var stepPacket = AiDotNet.Training.CompiledStepRandom<T>.Draw(new[] { frames, packetCols }, g =>
        {
            var packetDouble = new double[frames * packetCols];
            var audioMask = network.DrawMask(frames, _options.AudioMaskProbability, _options.AudioMaskLength, random);
            var videoMask = network.DrawMask(frames, _options.VideoMaskProbability, _options.VideoMaskLength, random);
            var copy = network.SampleVideoCopyIndices(frames, videoMask, random);

            double keepAVal = 1.0;
            double keepVVal = 1.0;
            if (training && _options.ModalityDropout > 0 && random.NextDouble() < _options.ModalityDropout)
            {
                if (random.NextDouble() < _options.AudioDropout) keepAVal = 0.0;
                else keepVVal = 0.0;
            }

            int masked = 0;
            for (int t = 0; t < frames; t++)
            {
                if (audioMask[t] || videoMask[t]) masked++;
            }

            double negInvMasked = masked > 0 ? -1.0 / masked : 0.0;
            for (int t = 0; t < frames; t++)
            {
                int row = t * packetCols;
                double audioKeepVal = audioMask[t] ? 0.0 : 1.0;
                for (int i = 0; i < width; i++) packetDouble[row + i] = audioKeepVal;
                packetDouble[row + width] = copy[t];
                if (audioMask[t] || videoMask[t])
                {
                    packetDouble[row + width + 1 + clusterTargets[t]] = negInvMasked;
                }
                packetDouble[row + width + 1 + numClusters] = keepAVal;
                packetDouble[row + width + 2 + numClusters] = keepVVal;
            }

            new Tensor<double>(packetDouble, new[] { frames, packetCols }).Cast<T>().AsSpan().CopyTo(g.AsWritableSpan());
        });

        var audioKeep = Engine.TensorSlice(stepPacket, new[] { 0, 0 }, new[] { frames, width });
        var videoCopyIndices = Engine.Reshape(Engine.TensorSlice(stepPacket, new[] { 0, width }, new[] { frames, 1 }), new[] { frames });
        var scaledPick = Engine.TensorSlice(stepPacket, new[] { 0, width + 1 }, new[] { frames, numClusters });

        Tensor<T> a = network.EncodeAudioWithKeepMask(cut, audioKeep);
        Tensor<T> v = network.EncodeVideoWithCopyIndices(video, videoCopyIndices, training);
        if (training && _options.ModalityDropout > 0)
        {
            var keepA = Engine.TensorBroadcastTo(Engine.TensorSlice(stepPacket, new[] { 0, width + 1 + numClusters }, new[] { frames, 1 }), a._shape);
            var keepV = Engine.TensorBroadcastTo(Engine.TensorSlice(stepPacket, new[] { 0, width + 2 + numClusters }, new[] { frames, 1 }), v._shape);
            a = Engine.TensorMultiply(a, keepA);
            v = Engine.TensorMultiply(v, keepV);
        }
        // Mean squared fused features over [frames, 2D]; a dropped stream contributes zeros.
        Tensor<T> Squares(Tensor<T> x) => Engine.ReduceSum(Engine.TensorMultiply(x, x), new[] { 0, 1 }, keepDims: false);
        var penalty = Engine.TensorMultiplyScalar(Engine.TensorAdd(Squares(a), Squares(v)), NumOps.FromDouble(1.0 / (frames * 2.0 * _options.EncoderDim)));
        var logits = network.ClusterLogits(network.Encode(network.Fuse(a, v, training, random), training, random));
        var prediction = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorLogSoftmax(logits, axis: -1), scaledPick), new[] { 0, 1 }, keepDims: false);
        return Engine.TensorAdd(Engine.Reshape(prediction, new[] { 1 }),
            Engine.TensorMultiplyScalar(Engine.Reshape(penalty, new[] { 1 }), NumOps.FromDouble(_options.FeaturePenaltyWeight)));
    }

    /// <summary>One masked multimodal cluster prediction step (section 3) on one utterance.</summary>
    /// <param name="audio">The 16 kHz waveform.</param>
    /// <param name="lips">Grayscale mouth crops <c>[frames, height, width]</c> at 25 frames a second.</param>
    /// <param name="clusterTargets">One cluster index per 25 Hz frame.</param>
    /// <returns>The step's loss.</returns>
    public T Pretrain(Tensor<T> audio, Tensor<T> lips, int[] clusterTargets)
    {
        ThrowIfDisposed();
        if (_network is null) throw new NotSupportedException("Pre-training runs on the native AV-HuBERT network.");
        if (clusterTargets is null) throw new ArgumentNullException(nameof(clusterTargets));
        // A caller's optimizer serves both phases; otherwise the paper's pre-training recipe (appendix B.4).
        _pretrainingOptimizer ??= _suppliedOptimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, phase: AiDotNet.Attributes.TrainingPhase.PreTraining)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        var features = _network.AudioFeatureTensor(audio);
        return TrainWithCustomObjective(features, lips,
            (stepFeatures, stepLips) => PretrainingLossFromFeaturesTensor(stepFeatures, stepLips, clusterTargets, training: true, _random),
            _pretrainingOptimizer);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "AVHuBERT-Native" : "AVHuBERT-ONNX", Description = "AV-HuBERT: audio-visual masked multimodal cluster prediction with a CTC projection (Meta, 2022)", FeatureCount = 1, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };



    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) => CtcGreedyDecoding.Decode(logits, _options.MaxTextLength);
    private static string TokensToText(List<int> tokens) => DecodeUnicodeTokens(tokens);
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) => BuildSingleUtteranceSegment(text, duration, confidence);
    private string ClassifyLanguageFromTokens(List<int> tokens) => ClassifyCjkOrDefaultLanguage(tokens, SupportedLanguages, _options.Language);
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AVHuBERT<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
