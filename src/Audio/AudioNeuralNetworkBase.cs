using AiDotNet.Diffusion.Audio;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds to ours when this import shadows them from a nearer
// scope. Without it the attribute resolves to the wrong type and ADNSHAPE003 reports this contract as
// having no input layout.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio;

/// <summary>
/// Base class for audio-focused neural networks that can operate in both ONNX inference and native training modes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class extends <see cref="NeuralNetworkBase{T}"/> to provide audio-specific functionality
/// while maintaining full integration with the AiDotNet neural network infrastructure.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio neural networks process sound data (like speech or music).
/// This base class provides:
///
/// - Support for pre-trained ONNX models (fast inference with existing models)
/// - Full training capability from scratch (like other neural networks)
/// - Audio preprocessing utilities (mel spectrograms, etc.)
/// - Sample rate handling
///
/// You can use this class in two ways:
/// 1. Load a pre-trained ONNX model for quick inference
/// 2. Build and train a new model from scratch
/// </para>
/// </remarks>
// MEASURED across the family, not assumed: every audio model probed returns RANK 2, [Batch, Features].
// Batch tracks the input; the feature width is model-specific and comes from OutputFeatureWidth below.
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public abstract partial class AudioNeuralNetworkBase<T> : NeuralNetworkBase<T>, IShapeContract
{
    /// <summary>
    /// The width of this model's output feature axis, or 0 when it has not been stated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// VIRTUAL returning 0, not abstract, and that is deliberate on two counts. Adding an abstract
    /// member to a public base is a breaking change for anything outside this repository that derives
    /// from it; and 0 lets the 206 models in this family be migrated INCREMENTALLY, with the remaining
    /// count readable from the conformance sweep at any point - the same ladder ADNSHAPE006 climbed
    /// from 85 of ~270 layers to zero.
    /// </para>
    /// <para>
    /// WHY THIS CANNOT LIVE ON THE BASE, unlike segmentation's _numClasses. Measured across the family,
    /// the width is a different quantity per task and is stored under a different name in every options
    /// class - AudioLM's SemanticVocabSize (1024), BasicPitch's NumHarmonicBins (264) - and is
    /// sometimes DERIVED rather than stored: BandSplitRNNEnhancer returns 257, which is its
    /// FFTSize (512) / 2 + 1. No single field and no name-matching rule can produce all three, which is
    /// why each model states its own one-line expression instead.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> the last number in an audio model's output says how many values it
    /// predicts per step - a vocabulary size for speech recognition, a bin count for pitch detection, a
    /// frequency count for enhancement. Each model knows its own, so each one reports it here.
    /// </para>
    /// </remarks>
    protected virtual int OutputFeatureWidth => 0;

    /// <summary>
    /// The output axes for an audio model: [Batch, Features].
    /// </summary>
    /// <remarks>
    /// Declines - returns null - until the model states its <see cref="OutputFeatureWidth"/>. Declining
    /// is the honest answer where nothing has been measured, and it is what keeps this contract from
    /// claiming a width it cannot know.
    /// </remarks>
    public virtual IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        int width = OutputFeatureWidth;
        if (inputRank != 2 || width <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(width)),
        };
    }

    /// <summary>
    /// Gets the sample rate expected by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values: 16000 Hz (speech), 22050 Hz (music), 44100 Hz (high quality).
    /// Input audio should be resampled to match this rate.
    /// </para>
    /// </remarks>
    public int SampleRate { get; protected set; } = 16000;

    /// <summary>
    /// Gets the number of mel spectrogram channels used by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mel spectrograms divide the frequency range into perceptual bands.
    /// Common values: 64, 80, or 128 mel bins.
    /// </para>
    /// </remarks>
    public int NumMels { get; protected set; } = 80;

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the model uses pre-trained ONNX weights for inference.
    /// When false, the model uses native layers and can be trained.
    /// </para>
    /// </remarks>
    public bool IsOnnxMode => OnnxEncoder is not null || OnnxDecoder is not null || OnnxModel is not null;

    /// <summary>
    /// Builds the common audio-model metadata entries (sample rate, mel-bin count, and
    /// inference mode) for <see cref="ModelMetadata{T}.AdditionalInfo"/>. Audio models call
    /// this from <c>GetModelMetadata</c> so their reported metadata contains concrete
    /// configuration rather than an empty <c>AdditionalInfo</c> dictionary.
    /// </summary>
    /// <returns>A populated key/value dictionary of audio configuration.</returns>
    protected Dictionary<string, object> BaseAudioMetadataInfo() => new()
    {
        ["SampleRate"] = SampleRate,
        ["NumMels"] = NumMels,
        ["Mode"] = IsOnnxMode ? "ONNX" : "Native"
    };

    /// <summary>
    /// Gets or sets the ONNX encoder model (for encoder-decoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxEncoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX decoder model (for encoder-decoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxDecoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX model (for single-model architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxModel { get; set; }

    /// <summary>
    /// Text-encoder layer stack for dual-encoder audio-text models (CLAP and
    /// similar contrastive audio-language models). Lives outside the inherited
    /// <see cref="NeuralNetworkBase{T}.Layers"/> list so the standard
    /// <see cref="NeuralNetworkBase{T}.Predict"/> / <c>TrainWithTape</c> paths
    /// operate on the audio-only stack; subclasses walk this collection
    /// explicitly inside their <c>EncodeText</c> implementations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mirrors <c>VisionLanguageModelBase.TextEncoderLayers</c> on the vision
    /// side — keeping the slot on this base means any future audio-text dual
    /// encoder (audio-side of an ImageBind-style multimodal model, CLAP
    /// derivatives, etc.) inherits a uniform two-encoder structure without
    /// re-declaring the field locally.
    /// </para>
    /// <para>
    /// Audio-only models (PANNsModel, ConformerFP, …) simply leave this empty;
    /// the slot has no effect on single-encoder code paths.
    /// </para>
    /// </remarks>
    protected readonly List<ILayer<T>> TextEncoderLayers = new List<ILayer<T>>();

    /// <summary>
    /// Surfaces <see cref="TextEncoderLayers"/> to the parameter walk for every audio model that
    /// owns a text tower.
    /// </summary>
    /// <remarks>
    /// TextEncoderLayers live outside <see cref="NeuralNetworkBase{T}.Layers"/>, so without this the
    /// base folds only the audio stream and the text tower reaches no ParameterCount and no
    /// checkpoint. Ten models compensated by hand-writing an UpdateParameters that walked both
    /// lists -- which fixed the write path and left the count and the vector still describing the
    /// audio stream alone. Yielding here fixes all three at once, in one place.
    /// </remarks>
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
    {
        foreach (var l in base.GetExtraTrainableLayers())
            yield return l;
        foreach (var layer in TextEncoderLayers)
        {
            if (layer is LayerBase<T> lb)
                yield return lb;
        }
    }

    private MelSpectrogram<T>? _melSpec;

    /// <summary>
    /// Gets the mel spectrogram extractor for preprocessing. Never null.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Defaults to a transform built for <see cref="SampleRate"/> the first time it is read, so a
    /// model that never assigns one still gets a real front-end instead of nothing.
    /// </para>
    /// <para>
    /// <b>Why this is not nullable.</b> It used to be, and the two behaviours that grew around that
    /// were both wrong. Most models call <c>MelSpec.Forward(rawAudio)</c> bare, which relies on some
    /// constructor having assigned it and gives a null dereference when none did. A few instead
    /// wrote <c>MelSpec is not null ? MelSpec.Forward(raw) : raw</c> — silently forwarding the RAW
    /// WAVEFORM, a rank-1 tensor with no time or frequency axis, into a transformer encoder. That
    /// second form is the more damaging: it does not fail, it just feeds meaningless features
    /// forward until an attention layer rejects the rank far from the cause.
    /// </para>
    /// </remarks>
    protected MelSpectrogram<T> MelSpec
    {
        get => _melSpec ??= new MelSpectrogram<T>(sampleRate: SampleRate);
        set => _melSpec = value ?? throw new ArgumentNullException(
            nameof(value),
            "MelSpec cannot be null. Audio models require a front-end; assign a configured " +
            "MelSpectrogram or leave it unset to get the SampleRate default.");
    }

    /// <summary>
    /// Initializes a new instance of the AudioNeuralNetworkBase class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, a default MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected AudioNeuralNetworkBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
        Options = new AudioNeuralNetworkOptions();
    }

    /// <summary>
    /// Creates the Adam/Transformer-schedule recipe used by Conformer-family ASR papers.
    /// </summary>
    /// <param name="modelDimension">Hidden width used by the inverse-square-root schedule.</param>
    /// <param name="warmupSteps">Number of linear warmup steps.</param>
    /// <param name="scheduleFactor">Multiplicative factor applied by the Noam schedule.</param>
    /// <param name="l2WeightDecay">Coupled L2 penalty used by <c>torch.optim.Adam(weight_decay=...)</c>.</param>
    /// <param name="maxGradientNorm">Global gradient-norm limit, or zero to disable clipping.</param>
    /// <returns>A paper-configured Adam optimizer owned by this network.</returns>
    /// <remarks>
    /// The recipe deliberately disables AiDotNet's adaptive-beta extension. Research configurations
    /// that say "Adam" prescribe fixed beta values; allowing those values to drift would implement a
    /// different optimizer. Weight decay is coupled L2 regularization here, matching PyTorch Adam,
    /// rather than decoupled AdamW decay.
    /// </remarks>
    protected IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateTransformerScheduleAdamOptimizer(
        int modelDimension,
        int warmupSteps,
        double scheduleFactor,
        double l2WeightDecay = 0.0,
        double maxGradientNorm = 0.0)
    {
        if (double.IsNaN(l2WeightDecay) || double.IsInfinity(l2WeightDecay) || l2WeightDecay < 0.0)
            throw new ArgumentOutOfRangeException(nameof(l2WeightDecay), "L2 weight decay must be finite and non-negative.");
        if (double.IsNaN(maxGradientNorm) || double.IsInfinity(maxGradientNorm) || maxGradientNorm < 0.0)
            throw new ArgumentOutOfRangeException(nameof(maxGradientNorm), "Maximum gradient norm must be finite and non-negative.");

        var scheduler = new NoamSchedule(modelDimension, warmupSteps, scheduleFactor);
        return CreateFixedAdamOptimizer(
            initialLearningRate: scheduler.CurrentLearningRate,
            beta1: 0.9,
            beta2: 0.98,
            epsilon: 1e-9,
            l2RegularizationStrength: l2WeightDecay,
            learningRateScheduler: scheduler,
            schedulerStepMode: SchedulerStepMode.StepPerBatch,
            maxGradientNorm: maxGradientNorm);
    }

    /// <summary>
    /// Creates fixed-beta AdamW with a per-batch one-cycle learning-rate schedule.
    /// </summary>
    /// <param name="maxLearningRate">Peak learning rate reached during the cycle.</param>
    /// <param name="totalSteps">Total number of optimizer steps in the cycle.</param>
    /// <param name="pctStart">Fraction of the cycle spent increasing the learning rate.</param>
    /// <param name="weightDecay">Decoupled AdamW weight decay.</param>
    /// <param name="beta1">First-moment coefficient.</param>
    /// <param name="beta2">Second-moment coefficient.</param>
    /// <param name="epsilon">Numerical-stability constant.</param>
    /// <param name="maxGradientNorm">Global gradient-norm limit, or zero to disable clipping.</param>
    /// <returns>An AdamW optimizer owned by this network.</returns>
    protected IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateOneCycleAdamWOptimizer(
        double maxLearningRate,
        int totalSteps,
        double pctStart,
        double weightDecay,
        double beta1,
        double beta2,
        double epsilon,
        double maxGradientNorm)
    {
        if (double.IsNaN(maxLearningRate) || double.IsInfinity(maxLearningRate) || maxLearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(maxLearningRate), "Maximum learning rate must be finite and positive.");
        if (totalSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalSteps), "Total steps must be positive.");
        if (double.IsNaN(pctStart) || double.IsInfinity(pctStart) || pctStart < 0.0 || pctStart >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(pctStart), "Warmup fraction must be finite and in [0, 1).");
        if (double.IsNaN(weightDecay) || double.IsInfinity(weightDecay) || weightDecay < 0.0)
            throw new ArgumentOutOfRangeException(nameof(weightDecay), "Weight decay must be finite and non-negative.");
        if (double.IsNaN(beta1) || double.IsInfinity(beta1) || beta1 < 0.0 || beta1 >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(beta1), "Beta1 must be finite and in [0, 1).");
        if (double.IsNaN(beta2) || double.IsInfinity(beta2) || beta2 < 0.0 || beta2 >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(beta2), "Beta2 must be finite and in [0, 1).");
        if (double.IsNaN(epsilon) || double.IsInfinity(epsilon) || epsilon <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be finite and positive.");
        if (double.IsNaN(maxGradientNorm) || double.IsInfinity(maxGradientNorm) || maxGradientNorm < 0.0)
            throw new ArgumentOutOfRangeException(nameof(maxGradientNorm), "Maximum gradient norm must be finite and non-negative.");

        var scheduler = new OneCycleLRScheduler(maxLearningRate, totalSteps, pctStart);
        return new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = scheduler.CurrentLearningRate,
                Beta1 = beta1,
                Beta2 = beta2,
                Epsilon = epsilon,
                WeightDecay = weightDecay,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                UseAMSGrad = false,
                EnableGradientClipping = maxGradientNorm > 0.0,
                MaxGradientNorm = maxGradientNorm > 0.0 ? maxGradientNorm : 1.0,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
    }
    /// <summary>
    /// Creates plain stochastic gradient descent with per-batch exponential learning-rate decay.
    /// </summary>
    /// <param name="learningRate">Initial SGD learning rate.</param>
    /// <param name="decayFactor">Multiplicative decay applied after each optimizer step.</param>
    /// <returns>An SGD optimizer owned by this network.</returns>
    /// <remarks>
    /// This is the single-process analogue of research recipes that use distributed asynchronous
    /// SGD. It preserves the stated update rule and schedule without substituting momentum,
    /// adaptive moments, or the unrelated averaged-SGD algorithm.
    /// </remarks>
    protected IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateExponentialSgdOptimizer(
        double learningRate,
        double decayFactor)
    {
        if (double.IsNaN(learningRate) || double.IsInfinity(learningRate) || learningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be finite and positive.");
        if (double.IsNaN(decayFactor) || double.IsInfinity(decayFactor) || decayFactor <= 0.0 || decayFactor > 1.0)
            throw new ArgumentOutOfRangeException(nameof(decayFactor), "Decay factor must be finite and in (0, 1].");

        var scheduler = new ExponentialLRScheduler(learningRate, decayFactor);
        return new StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new StochasticGradientDescentOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = learningRate,
                UseAdaptiveLearningRate = false,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
    }
    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In ONNX mode, training is not supported - the model is inference-only.
    /// In native mode, training is fully supported.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => !IsOnnxMode;

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    /// <param name="rawAudio">Raw audio waveform tensor [samples] or [batch, samples].</param>
    /// <returns>Preprocessed audio features suitable for model input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Raw audio is just a series of numbers representing sound pressure.
    /// Neural networks often work better with transformed representations like mel spectrograms.
    /// This method converts raw audio into the format the model expects.
    /// </para>
    /// </remarks>
    /// <remarks>
    /// <para>
    /// The default is the log-mel transform from <see cref="MelSpec"/>, which is what the large
    /// majority of audio models want. It was abstract, and 92 models each restated exactly this one
    /// line; those overrides are gone. Models with genuinely different preprocessing still override.
    /// </para>
    /// <para>
    /// <b>The rank check is the point.</b> Audio features must carry a time axis. A rank-1 result is
    /// a waveform or a single pooled frame, and feeding either to an attention stack gives a
    /// sequence of length one — attention over one token is a no-op, so a model built to localise
    /// events in time silently stops being able to. That used to surface far from here, as
    /// <c>MultiHeadAttentionLayer requires rank&gt;=2 input; got rank 1</c> deep inside an encoder.
    /// Failing at the boundary names the model instead.
    /// </para>
    /// </remarks>
    protected virtual Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        var features = MelSpec.Forward(rawAudio);
        RequireTimeAxis(features);
        return features;
    }

    /// <summary>
    /// Throws when preprocessed audio features have no time axis.
    /// </summary>
    /// <param name="features">The features returned by <see cref="PreprocessAudio"/>.</param>
    /// <exception cref="InvalidOperationException">The features are rank-1 or lower.</exception>
    /// <remarks>
    /// Available to overrides so a model with custom preprocessing can assert the same contract.
    /// </remarks>
    protected void RequireTimeAxis(Tensor<T> features)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (features.Shape.Length >= 2) return;

        throw new InvalidOperationException(
            $"{GetType().Name}.PreprocessAudio returned rank {features.Shape.Length}; audio features " +
            "must have at least a [frames, features] shape. A rank-1 result is a raw waveform or a " +
            "single pooled frame, which gives any downstream attention a sequence of length one.");
    }

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    /// <param name="modelOutput">Raw output from the model.</param>
    /// <returns>Postprocessed output in the expected format.</returns>
    protected abstract Tensor<T> PostprocessOutput(Tensor<T> modelOutput);

    /// <summary>
    /// Runs inference using ONNX model(s).
    /// </summary>
    /// <param name="input">Preprocessed input tensor.</param>
    /// <returns>Model output tensor.</returns>
    /// <remarks>
    /// <para>
    /// Override this method to implement ONNX-specific inference logic
    /// for models with complex encoder-decoder or multi-model architectures.
    /// </para>
    /// </remarks>
    protected virtual Tensor<T> RunOnnxInference(Tensor<T> input)
    {
        if (OnnxModel is not null)
        {
            return OnnxModel.Run(input);
        }

        if (OnnxEncoder is not null)
        {
            var encoded = OnnxEncoder.Run(input);
            if (OnnxDecoder is not null)
            {
                return OnnxDecoder.Run(encoded);
            }
            return encoded;
        }

        throw new InvalidOperationException("No ONNX model is loaded.");
    }

    /// <summary>
    /// Performs a forward pass through the native neural network layers.
    /// </summary>
    /// <param name="input">Preprocessed input tensor.</param>
    /// <returns>Model output tensor.</returns>
    protected virtual Tensor<T> Forward(Tensor<T> input)
    {
        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    /// <summary>
    /// Gets the default loss function for this model.
    /// </summary>
    public override ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Disposes of resources used by this model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            // TextEncoderLayers live outside NeuralNetworkBase<T>.Layers,
            // so base.Dispose won't reach them. Dispose any disposable
            // entries explicitly (Conv/Dense/etc. layers wrap pooled
            // tensor scratch buffers that would otherwise be retained
            // until process exit on long-running dual-encoder runs).
            foreach (var layer in TextEncoderLayers)
            {
                if (layer is IDisposable disposable)
                {
                    disposable.Dispose();
                }
            }
            TextEncoderLayers.Clear();

            OnnxEncoder?.Dispose();
            OnnxDecoder?.Dispose();
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }

    /// <summary>
    /// Creates a mel spectrogram extractor with the model's settings.
    /// </summary>
    /// <param name="sampleRate">Sample rate of input audio.</param>
    /// <param name="nMels">Number of mel bands.</param>
    /// <param name="nFft">FFT window size.</param>
    /// <param name="hopLength">Hop length between frames.</param>
    /// <returns>A configured mel spectrogram extractor.</returns>
    protected MelSpectrogram<T> CreateMelSpectrogram(
        int sampleRate = 16000,
        int nMels = 80,
        int nFft = 1024,
        int hopLength = 256)
    {
        return new MelSpectrogram<T>(
            sampleRate: sampleRate,
            nMels: nMels,
            nFft: nFft,
            hopLength: hopLength);
    }

    /// <summary>
    /// Trains one step with CIF's alignment supervision applied for the duration of the step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Any model whose stack contains a <see cref="CifAlignmentLayer{T}"/> must call this rather
    /// than <c>TrainWithTape</c> directly. Dong &amp; Xu 2020 (arXiv:1905.11235) supervise CIF's
    /// weight predictor through two mechanisms, both keyed to the label length S~: the scaling
    /// strategy (S3.3), which multiplies every firing weight by S~ / sum(alpha) so the integrated
    /// token count is teacher-forced to the target, and the quantity loss (S3.4),
    /// |sum(alpha) - S~|.
    /// </para>
    /// <para>
    /// The layer implements both and enables them by default, but gates them on
    /// <c>TargetTokenCount</c>, which only the training caller can know. Before this existed no
    /// model in the library set it, so on every CIF model both mechanisms were inert and the
    /// weight predictor trained with nothing supervising how many tokens it should emit. With
    /// sum(alpha) unconstrained the firing pattern changes discontinuously between steps, and
    /// CIFEncoder went non-finite within a step or two of training.
    /// </para>
    /// <para>
    /// The target is cleared again afterwards so inference runs on the raw alphas and decides its
    /// own output length, exactly as the paper specifies. The scan is a no-op for models that
    /// build no CIF stage -- the Paraformer family gates its CIF layer on
    /// <c>UseCifAlignment</c> -- so this is safe to call unconditionally.
    /// </para>
    /// </remarks>
    /// <param name="input">The training input.</param>
    /// <param name="expected">The target tensor; its token axis supplies S~.</param>
    /// <param name="optimizer">The model's configured optimizer.</param>
    protected void TrainWithCifSupervision(
        Tensor<T> input,
        Tensor<T> expected,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
    {
        SetCifTargetTokenCount(expected);
        try
        {
            TrainWithTape(input, expected, optimizer);
        }
        finally
        {
            SetCifTargetTokenCount(null);
        }
    }

    /// <summary>
    /// Points every CIF stage in this model at the current batch's target token count, or clears
    /// it when <paramref name="expected"/> is null.
    /// </summary>
    /// <remarks>
    /// Labels are [batch, tokens, vocab] or [tokens, vocab] unbatched, so the token axis is the
    /// one before the vocabulary axis.
    /// </remarks>
    private void SetCifTargetTokenCount(Tensor<T>? expected)
    {
        int? tokenCount = null;
        if (expected is not null)
        {
            int count = expected.Rank >= 2
                ? expected.Shape[expected.Rank - 2]
                : expected.Shape[0];
            if (count > 0) tokenCount = count;
        }

        foreach (var layer in Layers)
        {
            if (layer is CifAlignmentLayer<T> cif)
            {
                cif.TargetTokenCount = tokenCount;
            }
        }
    }
}
