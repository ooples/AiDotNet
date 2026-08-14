using AiDotNet.Diffusion.Audio;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds to ours when this import shadows them from a nearer
// scope. Without it the attribute resolves to the wrong type and ADNSHAPE003 reports this contract as
// having no input layout.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;

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

    /// <summary>
    /// Gets the mel spectrogram extractor for preprocessing.
    /// </summary>
    protected MelSpectrogram<T>? MelSpec { get; set; }

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
    protected abstract Tensor<T> PreprocessAudio(Tensor<T> rawAudio);

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
