using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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

        var result = new Tensor<T>(image._shape);
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
        return NumOps.FromDouble(VectorHelper.CosineSimilarity(a.ToVector(), b.ToVector()));
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

        var result = new Tensor<T>(embedding._shape);
        for (int i = 0; i < embedding.Length; i++)
            result[i] = NumOps.FromDouble(NumOps.ToDouble(embedding[i]) / norm);

        return result;
    }

    /// <summary>
    /// Validates image/token settings used to derive patch sizes for native VLM layer factories.
    /// </summary>
    protected static void ValidateVisualPatchOptions(int imageSize, int maxVisualTokens)
    {
        if (imageSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageSize), imageSize, "ImageSize must be greater than 0.");
        if (maxVisualTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxVisualTokens), maxVisualTokens, "MaxVisualTokens must be greater than 0.");
    }

    /// <summary>
    /// Computes a patch size from an image size and target visual-token budget.
    /// </summary>
    protected static int ComputeVisualPatchSize(int imageSize, int maxVisualTokens, bool roundUp = false)
    {
        ValidateVisualPatchOptions(imageSize, maxVisualTokens);
        double targetTokensPerSide = Math.Sqrt(maxVisualTokens);
        double patchSize = imageSize / targetTokensPerSide;
        return Math.Max(1, roundUp ? (int)Math.Ceiling(patchSize) : (int)patchSize);
    }

    /// <summary>
    /// Returns the layer count contributed by a standard transformer block.
    /// </summary>
    protected static int TransformerBlockLayerCount(double dropoutRate) => dropoutRate > 0 ? 6 : 5;

    /// <summary>
    /// Returns the layer count contributed by a cross-attention resampler block.
    /// </summary>
    protected static int ResamplerBlockLayerCount(double dropoutRate) => dropoutRate > 0 ? 8 : 7;

    /// <summary>
    /// Computes an encoder/decoder split boundary from repeated block counts.
    /// </summary>
    protected static int ComputeVisionLanguageBoundary(
        int leadingLayerCount,
        int visionLayerCount,
        int visionBlockLayerCount,
        int auxiliaryBlockCount = 0,
        int auxiliaryBlockLayerCount = 0,
        int trailingLayerCount = 0)
    {
        if (leadingLayerCount < 0) throw new ArgumentOutOfRangeException(nameof(leadingLayerCount));
        if (visionLayerCount < 0) throw new ArgumentOutOfRangeException(nameof(visionLayerCount));
        if (visionBlockLayerCount < 0) throw new ArgumentOutOfRangeException(nameof(visionBlockLayerCount));
        if (auxiliaryBlockCount < 0) throw new ArgumentOutOfRangeException(nameof(auxiliaryBlockCount));
        if (auxiliaryBlockLayerCount < 0) throw new ArgumentOutOfRangeException(nameof(auxiliaryBlockLayerCount));
        if (trailingLayerCount < 0) throw new ArgumentOutOfRangeException(nameof(trailingLayerCount));

        return checked(
            leadingLayerCount +
            visionLayerCount * visionBlockLayerCount +
            auxiliaryBlockCount * auxiliaryBlockLayerCount +
            trailingLayerCount);
    }

    /// <summary>
    /// Verifies the computed encoder/decoder split is inside the current layer list.
    /// </summary>
    protected void ValidateEncoderDecoderBoundary(int encoderLayerEnd)
    {
        if (encoderLayerEnd <= 0 || encoderLayerEnd > Layers.Count)
            throw new InvalidOperationException($"Invalid encoder boundary {encoderLayerEnd} for {Layers.Count} layers.");
    }

    /// <summary>
    /// Gets the default loss function for this model.
    /// </summary>
    public override ILossFunction<T> DefaultLossFunction => LossFunction;

    // --- Dual-stream layer split (CLIP / SigLIP / BLIP / etc.) ---
    //
    // CLIP-style models concatenate vision-encoder + text-encoder layers into
    // a single LayerHelper.CreateDefaultOpenCLIPLayers() output. Walking that
    // sequentially is incorrect: visionEmbeddingDim ≠ textEmbeddingDim in the
    // OpenCLIP defaults, so vision-projection output mismatches text-encoder
    // input on the first text-encoder MultiHeadAttention. This base provides
    // the storage + accessors so subclasses' Predict / Train walk the vision
    // stack only, and EncodeText walks the text stack independently — mirrors
    // PyTorch / HuggingFace CLIPModel where vision_model and text_model are
    // separate nn.Modules instead of one concatenated list.

    /// <summary>
    /// Text-encoder layers, walked by <c>EncodeText</c> in subclasses.
    /// Lives outside <see cref="NeuralNetworkBase{T}.Layers"/> so the inherited
    /// forward / TrainWithTape paths operate on the vision-only stack.
    /// Surface to streaming-pool / weight-registry by overriding
    /// <see cref="NeuralNetworkBase{T}.GetExtraTrainableLayers"/> in the
    /// concrete subclass (e.g. via <see cref="EnumerateTextEncoderTrainableLayers"/>).
    /// </summary>
    protected readonly List<ILayer<T>> TextEncoderLayers = new List<ILayer<T>>();

    /// <summary>
    /// Splits an OpenCLIP-shaped layer factory output (vision pre-norm +
    /// N×vision-block + vision-projection + text pre-norm + N×text-block +
    /// text-projection) into the model's <see cref="NeuralNetworkBase{T}.Layers"/>
    /// list (vision portion) and <see cref="TextEncoderLayers"/> (text portion).
    /// </summary>
    /// <param name="allLayers">The combined factory output to split.</param>
    /// <param name="visionLayerCount">Count of vision-portion layers; for the
    /// standard OpenCLIP / SigLIP factory this is
    /// <c>2 + numVisionLayers × blockSize</c> where <c>blockSize = 5</c> at
    /// dropoutRate = 0 and <c>6</c> with dropout (the +2 is the pre-norm and
    /// the projection head).</param>
    protected void SplitDualStreamLayers(IEnumerable<ILayer<T>> allLayers, int visionLayerCount)
    {
        if (allLayers is null) throw new System.ArgumentNullException(nameof(allLayers));
        if (visionLayerCount < 0)
            throw new System.ArgumentOutOfRangeException(nameof(visionLayerCount), "visionLayerCount must be ≥ 0.");

        // Materialize once so we can validate before mutating Layers /
        // TextEncoderLayers. An oversized visionLayerCount that the
        // previous loop silently accepted (everything piles into Layers,
        // TextEncoderLayers stays empty) would surface as the same
        // class of "EncodeText silently degrades" bug the per-encoder
        // overrides try to prevent — fail fast here so a layer-factory
        // drift becomes an obvious construction error instead of a
        // runtime no-op.
        var layerList = allLayers as IList<ILayer<T>> ?? new List<ILayer<T>>(allLayers);
        if (visionLayerCount > layerList.Count)
            throw new System.ArgumentOutOfRangeException(
                nameof(visionLayerCount),
                $"visionLayerCount ({visionLayerCount}) exceeds the supplied layer sequence " +
                $"({layerList.Count}). The supplied factory's layer count and the dual-stream " +
                "split point have drifted apart — check the OpenCLIP-style block-size / layer-count " +
                "math against the architecture's NumVisionLayers / DropoutRate.");

        for (int idx = 0; idx < layerList.Count; idx++)
        {
            if (idx < visionLayerCount) Layers.Add(layerList[idx]);
            else TextEncoderLayers.Add(layerList[idx]);
        }
    }

    /// <summary>
    /// Helper for subclasses overriding <see cref="NeuralNetworkBase{T}.GetExtraTrainableLayers"/>
    /// to surface their <see cref="TextEncoderLayers"/> to the base weight-registry walker.
    /// </summary>
    protected IEnumerable<LayerBase<T>?> EnumerateTextEncoderTrainableLayers()
    {
        foreach (var layer in TextEncoderLayers)
            if (layer is LayerBase<T> lb) yield return lb;
    }

    // --- Auxiliary-stream support (BLIP-2 / Q-Former / fusion VL models) ---
    //
    // Triple-stream architectures (vision encoder → Q-Former → decoder, or
    // vision + text + cross-modal fusion bridge) need MORE than the dual-stream
    // split. Subclasses register additional streams via
    // <see cref="RegisterAuxiliaryEncoderStream"/>; the base's auxiliary
    // enumerator walks them all so the inherited GetExtraTrainableLayers /
    // streaming-pool / weight-registry hooks stay correct without per-subclass
    // boilerplate.

    private readonly List<List<ILayer<T>>> _auxiliaryEncoderStreams = new List<List<ILayer<T>>>();

    /// <summary>
    /// Registers an auxiliary encoder stream that lives outside
    /// <see cref="NeuralNetworkBase{T}.Layers"/>. The base does NOT walk these
    /// in <c>Predict</c> / <c>TrainWithTape</c>; subclasses use them in their
    /// own forward methods (e.g. <c>GenerateFromImage</c> for Q-Former models,
    /// <c>FuseVisionAndText</c> for BridgeTower-style fusion). Registered
    /// streams are surfaced through
    /// <see cref="EnumerateAuxiliaryStreamTrainableLayers"/> so subclasses can
    /// override <see cref="NeuralNetworkBase{T}.GetExtraTrainableLayers"/>
    /// to yield from auxiliary + text streams in one call.
    /// </summary>
    /// <param name="stream">A non-null stream to register. Adds a reference,
    /// not a copy — subsequent mutations to <paramref name="stream"/> are
    /// visible.</param>
    protected void RegisterAuxiliaryEncoderStream(List<ILayer<T>> stream)
    {
        if (stream is null) throw new System.ArgumentNullException(nameof(stream));
        _auxiliaryEncoderStreams.Add(stream);
    }

    /// <summary>
    /// Iterates the registered auxiliary streams (Q-Former, decoder, fusion
    /// bridge, etc.) yielding each layer's <see cref="LayerBase{T}"/> view.
    /// Use alongside <see cref="EnumerateTextEncoderTrainableLayers"/> when
    /// the subclass owns BOTH a text encoder and additional streams.
    /// </summary>
    protected IEnumerable<LayerBase<T>?> EnumerateAuxiliaryStreamTrainableLayers()
    {
        foreach (var stream in _auxiliaryEncoderStreams)
            foreach (var layer in stream)
                if (layer is LayerBase<T> lb) yield return lb;
    }

    /// <summary>
    /// Combined helper that yields trainable-layer references for both the
    /// dual-stream <see cref="TextEncoderLayers"/> and any registered
    /// auxiliary streams. Most subclasses override
    /// <see cref="NeuralNetworkBase{T}.GetExtraTrainableLayers"/> to return
    /// this directly.
    /// </summary>
    protected IEnumerable<LayerBase<T>?> EnumerateAllAuxiliaryTrainableLayers()
    {
        foreach (var l in EnumerateTextEncoderTrainableLayers()) yield return l;
        foreach (var l in EnumerateAuxiliaryStreamTrainableLayers()) yield return l;
    }

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
