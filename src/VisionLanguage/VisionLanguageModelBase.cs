using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;

namespace AiDotNet.VisionLanguage;

/// <summary>
/// Base class for vision-language neural networks that can operate in both ONNX inference and native training modes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class extends <see cref="NeuralNetworkBase{T}"/> to provide vision-language-specific functionality
/// while maintaining full integration with the AiDotNet neural network infrastructure.
/// </para>
/// <para>
/// <b>For Beginners:</b> Vision-language models process both images and text together. This base class provides:
///
/// - Support for pre-trained ONNX models (fast inference with existing models)
/// - Full training capability from scratch (like other neural networks)
/// - Image preprocessing utilities (normalization, resizing)
/// - Dual-encoder architecture support (separate image and text encoders)
///
/// You can use this class in two ways:
/// 1. Load a pre-trained ONNX model for quick inference
/// 2. Build and train a new model from scratch
/// </para>
/// </remarks>
public abstract class VisionLanguageModelBase<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the expected input image size (height = width in pixels).
    /// </summary>
    public int ImageSize { get; protected set; } = 224;

    /// <summary>
    /// Gets the number of image channels expected (typically 3 for RGB).
    /// </summary>
    public int ImageChannels { get; protected set; } = 3;

    /// <summary>
    /// Gets the embedding dimensionality for this model.
    /// </summary>
    public int EmbeddingDim { get; protected set; } = 512;

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    public bool IsOnnxMode => OnnxImageEncoder is not null || OnnxTextEncoder is not null || OnnxModel is not null;

    /// <summary>
    /// Gets or sets the ONNX image encoder model (for dual-encoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxImageEncoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX text encoder model (for dual-encoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxTextEncoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX model (for single-model architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxModel { get; set; }

    /// <summary>
    /// Initializes a new instance of the VisionLanguageModelBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, a default MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected VisionLanguageModelBase(
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
    /// Preprocesses a raw image tensor for model input.
    /// </summary>
    /// <param name="image">Raw image tensor in [channels, height, width] or [height, width, channels] format.</param>
    /// <returns>Preprocessed image tensor suitable for model input.</returns>
    protected abstract Tensor<T> PreprocessImage(Tensor<T> image);

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    /// <param name="modelOutput">Raw output from the model.</param>
    /// <returns>Postprocessed output in the expected format.</returns>
    protected abstract Tensor<T> PostprocessOutput(Tensor<T> modelOutput);

    /// <summary>
    /// Normalizes an image tensor using ImageNet mean and standard deviation.
    /// </summary>
    /// <param name="image">Image tensor with values in [0, 1].</param>
    /// <param name="mean">Per-channel mean values (default: ImageNet [0.485, 0.456, 0.406]).</param>
    /// <param name="std">Per-channel standard deviation values (default: ImageNet [0.229, 0.224, 0.225]).</param>
    /// <returns>Normalized image tensor.</returns>
    protected Tensor<T> NormalizeImage(Tensor<T> image, double[]? mean = null, double[]? std = null)
    {
        mean ??= [0.48145466, 0.4578275, 0.40821073]; // CLIP/OpenAI default
        std ??= [0.26862954, 0.26130258, 0.27577711];

        var result = new Tensor<T>(image.Shape);
        int channels = ImageChannels;
        int spatialSize = image.Length / channels;

        for (int c = 0; c < channels && c < mean.Length; c++)
        {
            double m = mean[c];
            double s = std[c];
            for (int i = 0; i < spatialSize; i++)
            {
                int idx = c * spatialSize + i;
                if (idx < image.Length)
                {
                    double val = NumOps.ToDouble(image[idx]);
                    result[idx] = NumOps.FromDouble((val - m) / s);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Computes cosine similarity between two embedding tensors.
    /// </summary>
    /// <param name="a">First embedding.</param>
    /// <param name="b">Second embedding.</param>
    /// <returns>Cosine similarity score in [-1, 1].</returns>
    protected T CosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        double dotProduct = 0;
        double normA = 0;
        double normB = 0;
        int dim = Math.Min(a.Length, b.Length);

        for (int i = 0; i < dim; i++)
        {
            double av = NumOps.ToDouble(a[i]);
            double bv = NumOps.ToDouble(b[i]);
            dotProduct += av * bv;
            normA += av * av;
            normB += bv * bv;
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        double similarity = denom > 1e-8 ? dotProduct / denom : 0;
        return NumOps.FromDouble(similarity);
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
    /// L2-normalizes an embedding tensor.
    /// </summary>
    /// <param name="embedding">Embedding to normalize.</param>
    /// <returns>Unit-normalized embedding.</returns>
    protected Tensor<T> L2Normalize(Tensor<T> embedding)
    {
        double norm = 0;
        for (int i = 0; i < embedding.Length; i++)
        {
            double v = NumOps.ToDouble(embedding[i]);
            norm += v * v;
        }

        norm = Math.Sqrt(norm);
        if (norm < 1e-8) return embedding;

        var result = new Tensor<T>(embedding.Shape);
        for (int i = 0; i < embedding.Length; i++)
            result[i] = NumOps.FromDouble(NumOps.ToDouble(embedding[i]) / norm);

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
            OnnxImageEncoder?.Dispose();
            OnnxTextEncoder?.Dispose();
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
