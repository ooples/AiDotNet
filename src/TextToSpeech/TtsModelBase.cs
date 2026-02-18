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
public abstract class TtsModelBase<T> : NeuralNetworkBase<T>
{
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
    public bool IsOnnxMode => OnnxEncoder is not null || OnnxDecoder is not null || OnnxModel is not null;

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
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => !IsOnnxMode;

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
    protected Tensor<T> NormalizeMel(Tensor<T> mel, double minLevel = -100.0, double refLevel = 20.0)
    {
        var result = new Tensor<T>(mel.Shape);
        double range = refLevel - minLevel;

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
            if (v > maxVal) maxVal = v;
        }

        var result = new Tensor<T>(logits.Shape);
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
        if (norm < 1e-8) return tensor;

        var result = new Tensor<T>(tensor.Shape);
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
