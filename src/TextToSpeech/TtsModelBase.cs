using System.Collections.Generic;
// AiDotNet.Attributes is REQUIRED for [TensorLayout] to bind to the right type: two other Tensors
// namespaces declare a TensorLayout, and without this using the attribute silently resolves to one
// of those and the contract is never seen.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// Base class for text-to-speech neural networks that can operate in both ONNX inference and native training modes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class extends <see cref="NeuralNetworkBase{T}"/> to provide TTS-specific functionality
/// while maintaining full integration with the AiDotNet neural network infrastructure.
/// </para>
/// <para>
/// <b>For Beginners:</b> Text-to-speech models convert written text into spoken audio. This base class provides:
///
/// - Support for pre-trained ONNX models (fast inference with existing models)
/// - Full training capability from scratch (like other neural networks)
/// - Audio preprocessing utilities (mel-spectrogram computation, normalization)
/// - Text encoding utilities (phoneme/token conversion)
///
/// You can use this class in two ways:
/// 1. Load a pre-trained ONNX model for quick inference
/// 2. Build and train a new model from scratch
/// </para>
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "The encoded text or conditioning the layer stack consumes.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output,
    Note = "One frame per input position: the input's second axis is carried through as TIME and the "
         + "width is appended. Models whose Predict ends somewhere else state their own width through "
         + "OutputFeatureWidth, or decline by leaving it at 0.")]
public abstract class TtsModelBase<T> : NeuralNetworkBase<T>, IShapeContract
{
    /// <summary>
    /// The width of this model's <c>Predict</c> output, or 0 for "not stated".
    /// </summary>
    /// <remarks>
    /// <para>
    /// The shape here is the LAYER STACK'S, not the duration-predicted synthesis path that
    /// <c>TextToMel</c> drives: every <c>PredictCore</c> in this family is a plain fold over
    /// <c>Layers</c> - Tacotron2, FastSpeech2 and GlowTTS are all literally
    /// <c>foreach (var l in Layers) c = l.Forward(c)</c>.
    /// </para>
    /// <para>
    /// DEFAULTS TO 0 - "not stated" - rather than to <see cref="MelChannels"/>, and that was measured
    /// rather than chosen. Defaulting to MelChannels gave 18 agreed and 80 DISAGREED: the family
    /// splits into acoustic models that really do end at a mel width, and codec models that end at a
    /// token vocabulary (192, 626, 4096, 8192, 12288, 65536). A default right for 18 and wrong for 80
    /// is worse than no default, because the 80 then carry a confident false claim instead of an
    /// honest silence - and the sweep would report the family as broken rather than as unfinished.
    /// </para>
    /// <para>
    /// VIRTUAL AND DEFAULTED, not abstract: adding an abstract member to a public base breaks every
    /// external subclass, and a 0 lets a model that ends somewhere else opt out honestly instead of
    /// carrying a wrong width. The vocoders do exactly that - <c>VocoderBase</c> overrides
    /// <c>OutputAxesFor</c> outright, because a waveform is not a mel frame.
    /// </para>
    /// </remarks>
    protected virtual int OutputFeatureWidth => 0;

    /// <summary>
    /// The TTS family's output law: <c>[Batch, Time, OutputFeatureWidth]</c>, where Time is the
    /// input's second axis carried through.
    /// </summary>
    /// <remarks>
    /// MEASURED, and the first version of this was wrong in exactly the way a rank assumption usually
    /// is. It declared <c>[Batch, Width]</c> - rank 2 in, rank 2 out - and the sweep returned 86
    /// DISAGREEMENTS, every one of the form "in [1,64] contract says [1,80] but Predict returned
    /// [1,64,80]". The width was right; the RANK was not. These models emit one frame per input
    /// position, so the input axis survives as TIME and the width is appended to it. Nothing about
    /// "rank 2 in" implies "rank 2 out", and assuming so is what made the audio family's six
    /// rank-mismatched models look like width errors too.
    /// </remarks>
    public virtual IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        int width = OutputFeatureWidth;
        if (inputRank != 2 || width <= 0) return null;
        return
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Features)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(width)),
        ];
    }

    /// <summary>
    /// Gets the audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; protected set; } = 22050;

    /// <summary>
    /// Gets the number of mel-spectrogram frequency channels.
    /// </summary>
    public int MelChannels { get; protected set; } = 80;

    /// <summary>
    /// Gets the hop size in audio samples for mel-spectrogram computation.
    /// </summary>
    public int HopSize { get; protected set; } = 256;

    /// <summary>
    /// Gets the model's hidden dimension.
    /// </summary>
    public int HiddenDim { get; protected set; } = 256;

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    public bool IsOnnxMode =>
        OnnxEncoder is not null || OnnxDecoder is not null || OnnxModel is not null;

    /// <summary>
    /// Gets or sets the ONNX encoder model (for two-stage architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxEncoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX decoder model (for two-stage architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxDecoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX model (for single-model architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxModel { get; set; }

    /// <summary>
    /// Initializes a new instance of the TtsModelBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, a default MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected TtsModelBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0
    )
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm) { }

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => !IsOnnxMode;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _vocoderBaseOptimizer;

    /// <summary>
    /// Vocoder generators (those implementing <see cref="AiDotNet.TextToSpeech.Interfaces.IVocoder{T}"/>)
    /// train with AMSGrad rather than plain Adam. Their MRF / dilated-conv loss
    /// surfaces are bumpy enough that plain Adam's effective step can grow as the
    /// second-moment estimate shrinks near convergence, letting long training
    /// drift back up off the minimum. AMSGrad (Reddi et al. 2018; equivalent to
    /// <c>torch.optim.Adam(amsgrad=True)</c>) keeps a non-decreasing second-moment
    /// denominator, which bounds that drift. It is a strict convergence-stability
    /// improvement and does not affect inference. Non-vocoder TTS models (acoustic
    /// / end-to-end) keep the default base optimizer.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
    {
        if (
            this is AiDotNet.TextToSpeech.Interfaces.IVocoder<T>
            || this is AiDotNet.TextToSpeech.Interfaces.IEndToEndTts<T>
        )
        {
            return _vocoderBaseOptimizer ??= new AiDotNet.Optimizers.AdamOptimizer<
                T,
                Tensor<T>,
                Tensor<T>
            >(this, new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { UseAMSGrad = true });
        }
        return base.GetOrCreateBaseOptimizer();
    }

    /// <summary>
    /// Preprocesses raw text into a token tensor for model input.
    /// </summary>
    /// <param name="text">Raw text input.</param>
    /// <returns>Token tensor suitable for model input.</returns>
    protected abstract Tensor<T> PreprocessText(string text);

    /// <summary>
    /// Postprocesses model output into the final audio format.
    /// </summary>
    /// <param name="modelOutput">Raw output from the model.</param>
    /// <returns>Postprocessed audio tensor.</returns>
    protected abstract Tensor<T> PostprocessAudio(Tensor<T> modelOutput);

    /// <summary>
    /// Normalizes a mel-spectrogram tensor.
    /// </summary>
    /// <param name="mel">Mel-spectrogram tensor.</param>
    /// <param name="minLevel">Minimum amplitude level in dB (default: -100).</param>
    /// <param name="refLevel">Reference amplitude level in dB (default: 20).</param>
    /// <returns>Normalized mel-spectrogram tensor.</returns>
    protected Tensor<T> NormalizeMel(
        Tensor<T> mel,
        double minLevel = -100.0,
        double refLevel = 20.0
    )
    {
        var result = new Tensor<T>(mel._shape);
        double range = refLevel - minLevel;
        if (Math.Abs(range) < 1e-10)
            range = 1.0;

        for (int i = 0; i < mel.Length; i++)
        {
            double val = NumOps.ToDouble(mel[i]);
            double normalized = (val - minLevel) / range;
            normalized = Math.Max(0.0, Math.Min(1.0, normalized));
            result[i] = NumOps.FromDouble(normalized);
        }

        return result;
    }

    /// <summary>
    /// Applies GELU activation function element-wise.
    /// </summary>
    /// <param name="x">Input value.</param>
    /// <returns>GELU-activated value.</returns>
    protected static double Gelu(double x)
    {
        return x * 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
    }

    /// <summary>
    /// Applies softmax to convert logits to probabilities.
    /// </summary>
    /// <param name="logits">Raw scores.</param>
    /// <returns>Probabilities that sum to 1.</returns>
    protected Tensor<T> Softmax(Tensor<T> logits)
    {
        double maxVal = double.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            double v = NumOps.ToDouble(logits[i]);
            if (v > maxVal)
                maxVal = v;
        }

        var result = new Tensor<T>(logits._shape);
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double v = Math.Exp(NumOps.ToDouble(logits[i]) - maxVal);
            result[i] = NumOps.FromDouble(v);
            sum += v;
        }

        if (sum > 1e-8)
        {
            for (int i = 0; i < result.Length; i++)
                result[i] = NumOps.FromDouble(NumOps.ToDouble(result[i]) / sum);
        }

        return result;
    }

    /// <summary>
    /// L2-normalizes a tensor.
    /// </summary>
    /// <param name="tensor">Tensor to normalize.</param>
    /// <returns>Unit-normalized tensor.</returns>
    protected Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        double norm = 0;
        for (int i = 0; i < tensor.Length; i++)
        {
            double v = NumOps.ToDouble(tensor[i]);
            norm += v * v;
        }

        norm = Math.Sqrt(norm);
        if (norm < 1e-8)
            return tensor;

        var result = new Tensor<T>(tensor._shape);
        for (int i = 0; i < tensor.Length; i++)
            result[i] = NumOps.FromDouble(NumOps.ToDouble(tensor[i]) / norm);

        return result;
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
            OnnxEncoder?.Dispose();
            OnnxDecoder?.Dispose();
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
