using System.Collections.Generic;
// AiDotNet.Attributes is REQUIRED for [TensorLayout] to bind to the right type: two other Tensors
// namespaces declare a TensorLayout, and without this using the attribute silently resolves to one
// of those and the contract is never seen.
using AiDotNet.Attributes;
using AiDotNet.Document.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Postprocessing;

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
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input,
    Note = "A page or line image.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Classes,
    Direction = TensorLayoutDirection.Output,
    Note = "One class distribution per decode step: MaxSequenceLength steps, OutputClassCount "
         + "classes. Both are model constants, so neither spatial axis reaches the output.")]
[TensorPort("input", TensorPortDirection.Input, LayerInputDomainKind.Continuous,
    Role = TensorPortRole.Features,
    DomainResolver = nameof(ResolveDocumentInputDomain))]
public abstract partial class DocumentNeuralNetworkBase<T> : NeuralNetworkBase<T>, IShapeContract
{
    /// <summary>
    /// Resolves the public document-input domain from the first semantic consumer in the model graph.
    /// </summary>
    /// <remarks>
    /// Document models are deliberately heterogeneous: page-image models accept continuous pixels,
    /// token-first models accept bounded integer IDs, and layout-aware models accept packed continuous
    /// rows which split into validated token and coordinate streams internally.  The base neural-network
    /// graph already owns that distinction, so the generated public contract delegates to it instead of
    /// forcing every document model to repeat a hand-written override.
    /// </remarks>
    protected virtual LayerInputDomain ResolveDocumentInputDomain(int[]? inputShape) =>
        inputShape is { Length: >= 3 }
            ? LayerInputDomain.Continuous
            : base.GetInputDomain(inputShape);

    /// <summary>
    /// The number of classes this model emits per decode step, or 0 for "not stated".
    /// </summary>
    /// <remarks>
    /// Per-model rather than on the base, because it is the CHARSET size and every recognizer carries
    /// its own - and it is the charset PLUS ONE, for the CTC blank. ABINet and CRNN both end at
    /// <c>[MaxSequenceLength, _charset.Length + 1]</c>, which is where this law was read from.
    /// </remarks>
    protected virtual int OutputClassCount => 0;

    /// <summary>
    /// The document family's law: <c>[Batch, MaxSequenceLength, OutputClassCount]</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// MEASURED by probing, and the striking part is what does NOT appear. ABINet answered
    /// <c>[1,3,8,8] -&gt; [1,256,96]</c> and CRNN <c>[1,3,8,8] -&gt; [1,32,96]</c>, and moving EITHER
    /// spatial axis - <c>[1,3,12,8]</c>, <c>[1,3,8,12]</c> - left the output unchanged. A recognizer
    /// decodes a fixed number of steps regardless of how large the page it was handed is, so neither
    /// Height nor Width reaches the output at all. Only the batch axis is carried.
    /// </para>
    /// <para>
    /// Both constants are model configuration rather than literals: the step count is
    /// <see cref="MaxSequenceLength"/> on this base, and the class count is the model's own charset.
    /// Recording the probed 256 and 96 would have been right for one construction and wrong for any
    /// other - the error that made three vision-language contracts wrong.
    /// </para>
    /// </remarks>
    [ShapeContractRequiresPropertyOverride(nameof(OutputClassCount),
        "The generic document contract is concrete only when a model supplies its output class count.")]
    public virtual IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        int classes = OutputClassCount;
        if (inputRank != 4 || classes <= 0 || MaxSequenceLength <= 0) return null;
        return
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(MaxSequenceLength)),
            new OutputAxisContract(TensorAxis.Classes, AxisRelation.Fixed(classes)),
        ];
    }

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

    #region Preprocessing

    /// <summary>
    /// Gets or sets the instance-level preprocessing transformer for this document model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When set, <see cref="PreprocessDocument"/> uses this transformer instead of
    /// <see cref="ApplyDefaultPreprocessing"/>. This replaces the former static
    /// <c>PreprocessingRegistry</c> approach, which caused race conditions when
    /// multiple models were built concurrently.
    /// </para>
    /// <para><b>For Beginners:</b> If you want to customize how images are processed
    /// before being fed into this model, set this property. Otherwise, the model
    /// uses its own industry-standard defaults automatically.</para>
    /// </remarks>
    protected IDataTransformer<T, Tensor<T>, Tensor<T>>? PreprocessingTransformer { get; set; }

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
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), maxGradNorm)
    {
        Options = new DocumentNeuralNetworkOptions();
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
    /// 1. If an instance-level <see cref="PreprocessingTransformer"/> has been set, use it
    /// 2. Otherwise, use industry-standard defaults for this specific model type
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Raw images need to be transformed before the model can process them.
    /// You can either let the model use its industry-standard defaults (recommended for most cases),
    /// or configure custom preprocessing:
    /// <code>
    /// var result = new AiModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
    ///     .ConfigurePreprocessing(pipeline => pipeline
    ///         .Add(new ImageResizer&lt;double&gt;(224, 224))
    ///         .Add(new ImageNormalizer&lt;double&gt;()))
    ///     .Build(X, y);
    /// </code>
    /// </para>
    /// </remarks>
    protected Tensor<T> PreprocessDocument(Tensor<T> rawImage)
    {
        // Token-based document models (LayoutLM / LayoutXLM / LiLT / DocFormer / DocGCN / PICK /
        // TRIE / DocOwl / UDOP / InfographicVQA) consume a rank-1/2 sequence of TOKEN IDs, not raw
        // RGB pixels — their first layer is an EmbeddingLayer. Default image preprocessing (ImageNet
        // mean/std normalization over [C, H, W], which routes through EnsureBatchDimension and rejects
        // rank < 3) does not apply to token inputs, so pass them through unchanged. Genuine page-image
        // models (Donut, DocBank) are only ever fed rank-3/4 tensors, so this never bypasses their
        // normalization.
        if (rawImage.Rank < 3)
            return rawImage;

        // Priority 1: Instance-level transformer (set explicitly on this model)
        var transformer = PreprocessingTransformer;
        if (transformer is not null && transformer.IsFitted)
        {
            return transformer.Transform(rawImage);
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
    /// 1. If user configured a pipeline via AiModelBuilder.ConfigurePostprocessing() → use it
    /// 2. Otherwise → use industry-standard defaults for this specific model type
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Model outputs often need to be transformed into a usable format.
    /// You can either let the model use its industry-standard defaults (recommended for most cases),
    /// or configure custom postprocessing:
    /// <code>
    /// var result = new AiModelBuilder&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
    ///     .ConfigurePostprocessing(pipeline => pipeline
    ///         .Add(new SoftmaxTransformer&lt;double&gt;())
    ///         .Add(new LabelDecoder&lt;double&gt;(labels)))
    ///     .Build(X, y);
    /// </code>
    /// </para>
    /// </remarks>
    protected Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Model-specific industry-standard defaults
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
        Tensor<T>? encoderOutput = null;
        bool hasReshapedToSequence = false;
        bool hasPassedConvLayer = false;
        foreach (var layer in Layers)
        {
            // Track whether we've passed through any convolutional/pooling layer. The set mirrors
            // IsSpatialLayer below: residual conv blocks (BasicBlock/BottleneckBlock) and the
            // element-wise layers interleaved in a CNN backbone (activation, dropout) all keep the
            // [B,C,H,W] spatial layout, so passing one still counts as "inside the CNN stem".
            if (IsSpatialLayer(layer))
            {
                hasPassedConvLayer = true;
            }

            // PatchEmbeddingLayer performs the spatial→sequence flatten itself, and does it
            // tape-compatibly — so the training path (which replays the Layers through the
            // autodiff tape and never calls this inference Forward) gets the SAME flatten.
            // Mark the transition done so the inference-only auto-reshape below never fires
            // before OR after it; otherwise it would double-flatten the [C,H,W] map (handing
            // PatchEmbeddingLayer a rank-2 tensor it rejects, then re-mangling its output).
            if (layer is PatchEmbeddingLayer<T>)
                hasReshapedToSequence = true;

            // Auto-reshape once when transitioning from spatial (CNN) to non-spatial layers
            // Only reshape if we actually went through conv layers (not raw image input)
            // CNN outputs [B, C, H, W] or [C, H, W]; non-spatial layers expect [SeqLen, EmbDim].
            // A residual CNN backbone (PSENet's ResNet: BasicBlock/BottleneckBlock, with the usual
            // activation/dropout between stages) stays spatial the whole way; without treating those
            // as spatial the reshape fired at the FIRST residual block, handing the next MaxPool /
            // block a rank-2 [patches, channels] tensor ("MaxPooling requires rank-3/4; got rank 2").
            bool isNonSpatialLayer = !IsSpatialLayer(layer);
            if (!hasReshapedToSequence && hasPassedConvLayer && output.Shape.Length >= 3 && isNonSpatialLayer)
            {
                int channels = output.Shape.Length == 4 ? output.Shape[1] : output.Shape[0];
                int spatialH = output.Shape.Length == 4 ? output.Shape[2] : output.Shape[1];
                int spatialW = output.Shape.Length == 4 ? output.Shape[3] : output.Shape[2];
                int numPatches = spatialH * spatialW;
                output = new Tensor<T>(output.Data.ToArray(), [numPatches, channels]);
                hasReshapedToSequence = true;
            }

            // TransformerDecoderLayer requires encoder output as cross-attention context
            if (layer is TransformerDecoderLayer<T> decoderLayer)
            {
                encoderOutput ??= output;
                output = decoderLayer.Forward(output, encoderOutput);
            }
            else
            {
                output = layer.Forward(output);
            }
        }
        return output;
    }

    /// <summary>
    /// True when <paramref name="layer"/> preserves the CNN spatial layout ([B,C,H,W] / [C,H,W]),
    /// so the inference Forward's spatial→sequence auto-reshape must NOT fire on it. Covers the
    /// convolution / normalization / pooling primitives AND the residual conv blocks
    /// (<see cref="BasicBlock{T}"/>, <see cref="BottleneckBlock{T}"/>) and the element-wise layers
    /// (activation, dropout) that a residual CNN backbone interleaves between stages — a genuine
    /// paper-faithful ResNet backbone (e.g. PSENet) is entirely spatial. The reshape still fires at
    /// the first ACTUAL non-spatial layer (a transformer / attention / dense block), so transformer
    /// document models are unaffected.
    /// </summary>
    private static bool IsSpatialLayer(ILayer<T> layer) =>
        layer is ConvolutionalLayer<T> or BatchNormalizationLayer<T>
              or PoolingLayer<T> or MaxPoolingLayer<T> or AveragePoolingLayer<T>
              or ActivationLayer<T> or DropoutLayer<T>
              or BasicBlock<T> or BottleneckBlock<T>;

    /// <summary>
    /// Validates that an input image tensor has the correct shape.
    /// </summary>
    /// <param name="image">The tensor to validate.</param>
    /// <exception cref="ArgumentNullException">If image is null.</exception>
    /// <exception cref="ArgumentException">If the tensor shape is invalid.</exception>
    /// <summary>
    /// Safely serializes the model, returning an empty array if the model is too large.
    /// </summary>
    protected byte[] SafeSerialize()
    {
        try
        {
            return this.Serialize();
        }
        catch (OutOfMemoryException)
        {
            return Array.Empty<byte>();
        }
        catch (AiDotNet.Exceptions.LicenseRequiredException)
        {
            // #1830: AiModelResult's constructor captures metadata for every model it wraps, so an
            // eager serialization here turned AiModelBuilder.BuildAsync into a licensed operation and
            // threw out of the primary training entry point on an expired trial. Degrade to empty
            // model bytes exactly as NeuralNetworkBase.SerializeForMetadata already does -- metadata
            // without the serialized weights is still useful, a failed build is not.
            return Array.Empty<byte>();
        }
    }

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
            tensor.Data.Span.CopyTo(result.Data.Span);
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
