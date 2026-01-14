using AiDotNet.Document.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Postprocessing;
using AiDotNet.Preprocessing;

namespace AiDotNet.Document;

/// <summary>
/// Base class for document-focused neural networks that can operate in both ONNX inference and native training modes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class extends <see cref="NeuralNetworkBase{T}"/> to provide document-specific functionality
/// while maintaining full integration with the AiDotNet neural network infrastructure.
/// </para>
/// <para>
/// <b>For Beginners:</b> Document neural networks process images of documents (scanned pages, PDFs, photos).
/// This base class provides:
///
/// - Support for pre-trained ONNX models (fast inference with existing models)
/// - Full training capability from scratch (like other neural networks)
/// - Document preprocessing utilities (normalization, resizing, etc.)
/// - Layout-aware feature extraction
/// - Integration with text encoding for layout-aware models
///
/// You can use this class in two ways:
/// 1. Load a pre-trained ONNX model for quick inference
/// 2. Build and train a new model from scratch
/// </para>
/// </remarks>
public abstract class DocumentNeuralNetworkBase<T> : NeuralNetworkBase<T>
{
    #region Document-Specific Properties

    /// <summary>
    /// Gets the expected input image size for this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values: 224 (ViT base), 384, 448, 512, 768, 1024.
    /// Document images should be resized to match this size.
    /// </para>
    /// </remarks>
    public int ImageSize { get; protected set; } = 224;

    /// <summary>
    /// Gets the maximum text sequence length for layout-aware models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For models that process text tokens (like LayoutLM), this is the maximum
    /// number of tokens that can be processed. Typical values: 512, 1024, 2048.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength { get; protected set; } = 512;

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
    /// Gets the supported document types for this model.
    /// </summary>
    public abstract DocumentType SupportedDocumentTypes { get; }

    /// <summary>
    /// Gets whether this model requires OCR preprocessing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Layout-aware models (LayoutLM, etc.) require OCR to provide text and bounding boxes.
    /// OCR-free models (Donut, Pix2Struct) process raw pixels directly.
    /// </para>
    /// </remarks>
    public abstract bool RequiresOCR { get; }

    #endregion

    #region ONNX Mode Fields

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

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of the DocumentNeuralNetworkBase class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, CrossEntropyLoss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected DocumentNeuralNetworkBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), maxGradNorm)
    {
    }

    #endregion

    #region Core Methods

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
    /// Preprocesses a raw document image for model input.
    /// </summary>
    /// <param name="rawImage">Raw document image tensor [channels, height, width] or [batch, channels, height, width].</param>
    /// <returns>Preprocessed image suitable for model input.</returns>
    /// <remarks>
    /// <para>
    /// <b>Priority Order:</b>
    /// 1. If user configured a pipeline via PredictionModelBuilder.ConfigurePreprocessing() → use it
    /// 2. Otherwise → use industry-standard defaults for this specific model type
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Raw images need to be transformed before the model can process them.
    /// You can either let the model use its industry-standard defaults (recommended for most cases),
    /// or configure custom preprocessing:
    /// <code>
    /// var result = new PredictionModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
    ///     .ConfigurePreprocessing(pipeline => pipeline
    ///         .Add(new ImageResizer&lt;double&gt;(224, 224))
    ///         .Add(new ImageNormalizer&lt;double&gt;()))
    ///     .Build(X, y);
    /// </code>
    /// </para>
    /// </remarks>
    protected Tensor<T> PreprocessDocument(Tensor<T> rawImage)
    {
        // Priority 1: User-configured pipeline via PredictionModelBuilder
        if (PreprocessingRegistry<T, Tensor<T>>.IsConfigured)
        {
            return PreprocessingRegistry<T, Tensor<T>>.Transform(rawImage);
        }

        // Priority 2: Model-specific industry-standard defaults
        return ApplyDefaultPreprocessing(rawImage);
    }

    /// <summary>
    /// Applies industry-standard preprocessing defaults for this specific model type.
    /// </summary>
    /// <param name="rawImage">Raw document image tensor.</param>
    /// <returns>Preprocessed image using model-specific defaults.</returns>
    /// <remarks>
    /// <para>
    /// Each model should implement this with its paper-recommended preprocessing.
    /// For example:
    /// - TrOCR: Resize to 384x384, normalize with mean=0.5, std=0.5
    /// - LayoutLMv3: Resize to 224x224, ImageNet normalization
    /// - Donut: Resize to 2560x1920, normalize to [-1,1]
    /// </para>
    /// </remarks>
    protected abstract Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage);

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    /// <param name="modelOutput">Raw output from the model.</param>
    /// <returns>Postprocessed output in the expected format.</returns>
    /// <remarks>
    /// <para>
    /// <b>Priority Order:</b>
    /// 1. If user configured a pipeline via PredictionModelBuilder.ConfigurePostprocessing() → use it
    /// 2. Otherwise → use industry-standard defaults for this specific model type
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Model outputs often need to be transformed into a usable format.
    /// You can either let the model use its industry-standard defaults (recommended for most cases),
    /// or configure custom postprocessing:
    /// <code>
    /// var result = new PredictionModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
    ///     .ConfigurePostprocessing(pipeline => pipeline
    ///         .Add(new SoftmaxTransformer&lt;double&gt;())
    ///         .Add(new LabelDecoder&lt;double&gt;(labels)))
    ///     .Build(X, y);
    /// </code>
    /// </para>
    /// </remarks>
    protected Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Priority 1: User-configured pipeline via PredictionModelBuilder
        if (PostprocessingRegistry<T, Tensor<T>>.IsConfigured)
        {
            return PostprocessingRegistry<T, Tensor<T>>.Transform(modelOutput);
        }

        // Priority 2: Model-specific industry-standard defaults
        return ApplyDefaultPostprocessing(modelOutput);
    }

    /// <summary>
    /// Applies industry-standard postprocessing defaults for this specific model type.
    /// </summary>
    /// <param name="modelOutput">Raw model output tensor.</param>
    /// <returns>Postprocessed output using model-specific defaults.</returns>
    /// <remarks>
    /// <para>
    /// Each model should implement this with its paper-recommended postprocessing.
    /// For example:
    /// - Classification models: Softmax + argmax
    /// - Detection models: NMS + confidence thresholding
    /// - OCR models: CTC decoding or attention decoding
    /// </para>
    /// </remarks>
    protected abstract Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput);

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
    /// <para>
    /// This method expects either <see cref="OnnxModel"/> or
    /// <see cref="OnnxEncoder"/>/<see cref="OnnxDecoder"/> to be configured,
    /// but not both. When only an encoder is set, the encoded output is returned.
    /// </para>
    /// </remarks>
    protected virtual Tensor<T> RunOnnxInference(Tensor<T> input)
    {
        if (OnnxModel is not null && (OnnxEncoder is not null || OnnxDecoder is not null))
        {
            throw new InvalidOperationException(
                "OnnxModel cannot be combined with OnnxEncoder/OnnxDecoder. Configure only one ONNX pipeline.");
        }

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

        if (OnnxDecoder is not null)
        {
            throw new InvalidOperationException(
                "OnnxDecoder is set but OnnxEncoder is null. Encoder-decoder models require both components.");
        }

        throw new InvalidOperationException(
            "No ONNX model is loaded. Set either OnnxModel or OnnxEncoder to enable ONNX inference.");
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
    /// Validates that an input image tensor has the correct shape.
    /// </summary>
    /// <param name="image">The tensor to validate.</param>
    /// <exception cref="ArgumentNullException">If image is null.</exception>
    /// <exception cref="ArgumentException">If the tensor shape is invalid.</exception>
    protected void ValidateImageShape(Tensor<T> image)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));

        if (image.Rank < 3 || image.Rank > 4)
            throw new ArgumentException(
                $"Document image must be 3D [C,H,W] or 4D [B,C,H,W], got {image.Rank}D tensor.",
                nameof(image));
    }

    /// <summary>
    /// Adds a batch dimension to a 3D tensor if needed.
    /// </summary>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A 4D tensor with batch dimension.</returns>
    protected Tensor<T> EnsureBatchDimension(Tensor<T> tensor)
    {
        if (tensor.Rank == 4)
            return tensor;

        if (tensor.Rank == 3)
        {
            // [C, H, W] -> [1, C, H, W]
            int c = tensor.Shape[0];
            int h = tensor.Shape[1];
            int w = tensor.Shape[2];
            var result = new Tensor<T>([1, c, h, w]);
            Array.Copy(tensor.Data.ToArray(), result.Data.ToArray(), tensor.Data.Length);
            return result;
        }

        throw new ArgumentException($"Expected 3D or 4D tensor, got {tensor.Rank}D");
    }

    /// <summary>
    /// Gets the default loss function for this model.
    /// </summary>
    public override ILossFunction<T> DefaultLossFunction => LossFunction;

    #endregion

    #region Disposal

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

    #endregion
}
