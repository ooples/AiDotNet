using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Models.Options;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;

namespace AiDotNet.Document.LayoutAware;

/// <summary>
/// DocFormer neural network for end-to-end document understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DocFormer is a multi-modal transformer that jointly learns text, visual, and spatial features
/// for document understanding tasks. It uses shared spatial encodings across all modalities.
/// </para>
/// <para>
/// <b>For Beginners:</b> DocFormer combines three types of information:
/// 1. Text content (what the words say)
/// 2. Visual features (what the document looks like)
/// 3. Spatial layout (where elements are positioned)
///
/// Unlike LayoutLM which adds position embeddings to text, DocFormer uses shared
/// spatial encodings that align all three modalities in the same coordinate space.
///
/// Example usage:
/// <code>
/// var model = new DocFormer&lt;float&gt;(architecture);
/// var result = model.DetectLayout(documentImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "DocFormer: End-to-End Transformer for Document Understanding" (ICCV 2021)
/// https://arxiv.org/abs/2106.11539
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[RankRoutedInputDomain(2, 8)]
[ResearchPaper("DocFormer: End-to-End Transformer for Document Understanding", "https://doi.org/10.48550/arXiv.2106.11539", Year = 2021, Authors = "Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, R. Manmatha")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, WarmupFraction = 0.1,
                MaxGradientNorm = 1.0, Phase = TrainingPhase.PreTraining,
                Source = "Appalaraju et al. 2021, training details table: AdamW at a pre-training "
                        + "learning rate of 5e-05 with warmup over 10 percent of iterations and gradient "
                        + "clipping at 1.0.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2.5e-5, MaxGradientNorm = 1.0,
                Phase = TrainingPhase.FineTuning,
                Source = "Appalaraju et al. 2021, training details table: fine-tuning uses a learning "
                        + "rate of 2.5e-05 with no warmup and the same gradient clipping of 1.0.")]
public partial class DocFormer<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IDocumentClassifier<T>
{
    private readonly DocFormerOptions _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly ITokenizer _tokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _hiddenDim;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _vocabSize;
    private readonly int _numClasses;
    private readonly int _spatialDim;



    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => true;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    public IReadOnlyList<LayoutElementType> SupportedElementTypes { get; } =
    [
        LayoutElementType.Text,
        LayoutElementType.Title,
        LayoutElementType.List,
        LayoutElementType.Table,
        LayoutElementType.Figure,
        LayoutElementType.Caption,
        LayoutElementType.Header,
        LayoutElementType.Footer,
        LayoutElementType.FormField
    ];

    /// <summary>
    /// Gets the available document classification categories.
    /// </summary>
    public IReadOnlyList<string> AvailableCategories { get; } =
    [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific", "specification", "file_folder", "news_article",
        "budget", "invoice", "presentation", "questionnaire", "resume", "memo"
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DocFormer model using a pre-trained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="tokenizer">Tokenizer for text processing.</param>
    /// <param name="numClasses">Number of output classes (default: 16 for RVL-CDIP).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522).</param>
    /// <param name="spatialDim">Spatial embedding dimension (default: 128).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    public DocFormer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        DocFormerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new DocFormerOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        Guard.NotNull(tokenizer);
        _tokenizer = tokenizer;
        _useNativeMode = false;
        // Validated after the path checks so a missing model file reports itself
        // as FileNotFoundException rather than being pre-empted by the options.
        _options.Validate();

        _numClasses = _options.NumClasses;
        _hiddenDim = _options.HiddenDim;
        _numLayers = _options.NumLayers;
        _numHeads = _options.NumHeads;
        _vocabSize = _options.VocabSize;
        _spatialDim = _options.SpatialDim;
        // DocFormer fine-tuning uses AdamW at 2.5e-5 with no warm-up and a 1.0
        // gradient-norm cap (Appalaraju et al., ICCV 2021, Table 1). Keep the
        // optimizer injectable so callers can fully customize the training recipe.
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = 2.5e-5,
                    WeightDecay = 0.01,
                    UseAMSGrad = false,
                    EnableGradientClipping = true,
                    MaxGradientNorm = 1.0
                }));

        ImageSize = _options.ImageSize;
        MaxSequenceLength = _options.MaxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a DocFormer model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="tokenizer">Tokenizer for text processing (optional).</param>
    /// <param name="numClasses">Number of output classes (default: 16 for RVL-CDIP).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522).</param>
    /// <param name="spatialDim">Spatial embedding dimension (default: 128).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (DocFormer-Base from ICCV 2021):</b>
    /// - Text encoder: BERT-base architecture
    /// - Visual encoder: ResNet-50 backbone
    /// - Shared spatial encodings for all modalities
    /// - Hidden dimension: 768
    /// - Layers: 12, Heads: 12
    /// - Image size: 224x224
    /// </para>
    /// </remarks>
    public DocFormer(
        NeuralNetworkArchitecture<T> architecture,
        DocFormerOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new DocFormerOptions();
        _options.Validate();
        Options = _options;

        _useNativeMode = true;
        _numClasses = _options.NumClasses;
        _hiddenDim = _options.HiddenDim;
        _numLayers = _options.NumLayers;
        _numHeads = _options.NumHeads;
        _vocabSize = _options.VocabSize;
        _spatialDim = _options.SpatialDim;
        // DocFormer fine-tuning uses AdamW at 2.5e-5 with no warm-up and a 1.0
        // gradient-norm cap (Appalaraju et al., ICCV 2021, Table 1). Keep the
        // optimizer injectable so callers can fully customize the training recipe.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 2.5e-5,
                WeightDecay = 0.01,
                UseAMSGrad = false,
                EnableGradientClipping = true,
                MaxGradientNorm = 1.0
            });

        ImageSize = _options.ImageSize;
        MaxSequenceLength = _options.MaxSequenceLength;

        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            return;
        }

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        // Streams (Appalaraju et al. 2021; reference ExtractFeatures + DocFormerEncoder), each a branch root:
        // text word embeddings, the text and visual spatial embeddings, the ResNet-50 visual trunk, then the
        // multi-modal encoder and the token classifier.
        IActivationFunction<T> identity = new IdentityActivation<T>();
        _textEmbeddingIndex = Layers.Count;
        Layers.Add(LayerGraphContract.FromExternalInput(new EmbeddingLayer<T>(_vocabSize, _hiddenDim)));
        _textSpatialIndex = Layers.Count;
        Layers.Add(LayerGraphContract.FromDerivedInput(new DocFormerSpatialEmbeddingLayer<T>(_hiddenDim, MaxPosition2D), "boxes"));
        _visualSpatialIndex = Layers.Count;
        Layers.Add(LayerGraphContract.FromDerivedInput(new DocFormerSpatialEmbeddingLayer<T>(_hiddenDim, MaxPosition2D), "boxes"));
        _visualStart = Layers.Count;
        bool first = true;
        foreach (var layer in LayerHelper<T>.CreateResNetBottleneckEncoderLayers(3, ImageSize, ImageSize,
                     new[] { 256, 512, 1024, 2048 }, new[] { 3, 4, 6, 3 }))
        {
            Layers.Add(first ? LayerGraphContract.FromExternalInput(layer) : layer);
            first = false;
        }
        // Conv1x1(2048 -> hidden) + ReLU, then Linear over the flattened spatial grid to MaxSequenceLength tokens.
        _visualProjectionIndex = Layers.Count;
        Layers.Add(new ConvolutionalLayer<T>(_hiddenDim, 1, 1, 0, (IActivationFunction<T>)new ReLUActivation<T>()));
        _visualTokenIndex = Layers.Count;
        Layers.Add(new DenseLayer<T>(MaxSequenceLength, identity));
        _encoderIndex = Layers.Count;
        Layers.Add(LayerGraphContract.FromDerivedInput(
            new DocFormerEncoderLayer<T>(_hiddenDim, _numHeads, 4 * _hiddenDim, _options.MaxRelativePositions, _numLayers), "text"));
        _headIndex = Layers.Count;
        Layers.Add(new DenseLayer<T>(_numClasses, identity));
        _defaultStack = true;
    }

    #endregion

    #region ILayoutDetector Implementation

    /// <inheritdoc/>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage)
    {
        return DetectLayout(documentImage, 0.5);
    }

    /// <inheritdoc/>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage, double confidenceThreshold)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var regions = ParseLayoutOutput(output, confidenceThreshold);

        return new DocumentLayoutResult<T>
        {
            Regions = regions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    private List<LayoutRegion<T>> ParseLayoutOutput(Tensor<T> output, double threshold)
    {
        int numClasses = output.Shape.Length > 1 ? output.Shape[1] : _numClasses;
        var (rows, classes, scores) = DocumentTensorOps.ForegroundRows(Engine, output, numClasses, NumOps.FromDouble(threshold));
        return rows.Select((i, k) => new LayoutRegion<T>
        {
            ElementType = (LayoutElementType)Math.Min(classes[k], (int)LayoutElementType.Other),
            Confidence = scores[k],
            Index = i,
            BoundingBox = Vector<T>.Empty()
        }).ToList();
    }

    #endregion

    #region IDocumentClassifier Implementation

    /// <inheritdoc/>
    public DocumentClassificationResult<T> ClassifyDocument(Tensor<T> documentImage)
    {
        return ClassifyDocument(documentImage, 5);
    }

    /// <inheritdoc/>
    public DocumentClassificationResult<T> ClassifyDocument(Tensor<T> documentImage, int topK)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        // Apply softmax for classification probabilities
        var probs = ApplySoftmax(output);

        // Get top-K predictions
        var topPredictions = GetTopKPredictions(probs, topK);

        return new DocumentClassificationResult<T>
        {
            PredictedCategory = topPredictions[0].Category,
            Confidence = topPredictions[0].Score,
            TopPredictions = topPredictions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    private List<(string Category, T Score)> GetTopKPredictions(Tensor<T> probs, int k)
    {
        int numClasses = Math.Min(probs.Length, AvailableCategories.Count);
        if (numClasses == 0)
            return new List<(string Category, T Score)>();
        var head = Engine.TensorSlice(Engine.Reshape(probs, new[] { probs.Length }), new[] { 0 }, new[] { numClasses });
        int[] order = DocumentTensorOps.DescendingOrder(Engine, head).Take(Math.Max(0, k)).ToArray();
        T[] scores = DocumentTensorOps.Pick(Engine, head, order);
        return order.Select((c, r) => (AvailableCategories[c], scores[r])).ToList();
    }

    private Tensor<T> ApplySoftmax(Tensor<T> input)
    {
        return Engine.Softmax(input, -1);
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
        var preprocessed = PreprocessDocument(documentImage);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("DocFormer Model Summary");
        sb.AppendLine("=======================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: Multi-modal Transformer with shared spatial encodings");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Number of Layers: {_numLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Spatial Embedding Dim: {_spatialDim}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"Uses Visual Features: Yes");
        sb.AppendLine($"Uses Shared Spatial Encodings: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies DocFormer's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// DocFormer uses ImageNet normalization with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
        => DocumentTensorOps.NormalizeChannels(Engine, EnsureBatchDimension(rawImage),
            DocumentTensorOps.ImageNetMean, DocumentTensorOps.ImageNetStd);

    /// <summary>
    /// Applies DocFormer's industry-standard postprocessing: pass-through (multimodal outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DocFormer",
            Description = "DocFormer with shared spatial encodings (ICCV 2021)",
            FeatureCount = _hiddenDim,
            Complexity = _numLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_layers", _numLayers },
                { "num_heads", _numHeads },
                { "vocab_size", _vocabSize },
                { "image_size", ImageSize },
                { "spatial_dim", _spatialDim },
                { "num_classes", _numClasses },
                { "use_native_mode", _useNativeMode }
            },
            ModelDataProvider = () => SafeSerialize()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    // Positions of each component in Layers, set by InitializeLayers for the default DocFormer stack.
    private const int MaxPosition2D = 1024;
    private bool _defaultStack;
    private int _textEmbeddingIndex;
    private int _textSpatialIndex;
    private int _visualSpatialIndex;
    private int _visualStart;
    private int _visualProjectionIndex;
    private int _visualTokenIndex;
    private int _encoderIndex;
    private int _headIndex;

    /// <summary>
    /// The DocFormer forward. Text rows are padded with token 0 and a zero box to MaxSequenceLength (the
    /// reference always runs at max_position_embeddings, because the visual stream has exactly that many
    /// tokens). The first <paramref name="keep"/> rows of token logits <c>[keep, numClasses]</c> are returned.
    /// Token ids <c>[S]</c> and boxes <c>[S, 4]</c> may each be null (no tokens, or zero boxes); ids are rounded
    /// and clamped to the vocabulary.
    /// </summary>
    private Tensor<T> RunDocFormer(Tensor<T>? tokens, Tensor<T>? boxes, Tensor<T>? pageImage, int keep, IDictionary<string, Tensor<T>>? activations = null)
    {
        int l = MaxSequenceLength;
        int s = Math.Min(tokens?.Length ?? 0, l);
        var ids = new Tensor<T>(new[] { l });
        var boxTensor = new Tensor<T>(new[] { l, 4 });
        if (s > 0 && tokens is not null)
        {
            var clamped = Engine.TensorClamp(Engine.TensorRound(Engine.TensorSlice(tokens, new[] { 0 }, new[] { s })),
                NumOps.Zero, NumOps.FromDouble(_vocabSize - 1));
            ids = s == l ? clamped : Engine.TensorConcatenate(new[] { clamped, new Tensor<T>(new[] { l - s }) }, 0);
            if (boxes is not null)
            {
                var head = Engine.TensorSlice(boxes, new[] { 0, 0 }, new[] { s, 4 });
                boxTensor = s == l ? head : Engine.TensorConcatenate(new[] { head, new Tensor<T>(new[] { l - s, 4 }) }, 0);
            }
        }

        var text = Layers[_textEmbeddingIndex].Forward(ids);
        var textSpatial = Layers[_textSpatialIndex].Forward(boxTensor);
        var visualSpatial = Layers[_visualSpatialIndex].Forward(boxTensor);
        Tensor<T> visual;
        if (pageImage is null)
        {
            visual = new Tensor<T>(new[] { l, _hiddenDim });
        }
        else
        {
            var image = pageImage.Rank == 3
                ? Engine.Reshape(pageImage, new[] { 1, pageImage.Shape[0], pageImage.Shape[1], pageImage.Shape[2] })
                : pageImage;
            var features = image;
            for (int i = _visualStart; i <= _visualProjectionIndex; i++) features = Layers[i].Forward(features);
            int spatial = features.Shape[2] * features.Shape[3];
            var grid = Engine.Reshape(features, new[] { _hiddenDim, spatial });                 // [hidden, h*w]
            visual = Engine.TensorPermute(Layers[_visualTokenIndex].Forward(grid), new[] { 1, 0 });  // [L, hidden]
        }
        if (activations is not null)
        {
            activations["text_embedding"] = text;
            activations["text_spatial"] = textSpatial;
            activations["visual_spatial"] = visualSpatial;
            activations["visual_tokens"] = visual;
        }

        var encoded = ((DocFormerEncoderLayer<T>)Layers[_encoderIndex]).Forward(text, visual, textSpatial, visualSpatial);
        var logits = Layers[_headIndex].Forward(encoded);
        if (activations is not null) activations["encoder"] = encoded;
        return keep == l ? logits : Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { keep, _numClasses });
    }

    /// <summary>
    /// Routes a public input:
    /// <list type="bullet">
    /// <item>Packed <c>[S, 5]</c> rows (token id, x0, y0, x1, y1).</item>
    /// <item>Token ids <c>[S]</c>, with zero boxes.</item>
    /// <item>A page image <c>[3, H, W]</c>, with an all-padding text stream; all MaxSequenceLength rows are
    /// returned.</item>
    /// </list>
    /// </summary>
    private Tensor<T> RouteDocFormer(Tensor<T> input, IDictionary<string, Tensor<T>>? activations = null)
    {
        if (input.Rank >= 3)
            return RunDocFormer(null, null, input, MaxSequenceLength, activations);
        int s = input.Shape[0];
        if (s > MaxSequenceLength)
            throw new ArgumentException($"DocFormer takes at most {MaxSequenceLength} tokens; got {s}.", nameof(input));
        bool packed = input.Rank == 2 && input.Shape[1] == 5;
        if (input.Rank == 2 && !packed && input.Shape[1] != 1)
            throw new ArgumentException("DocFormer expects packed rows [S, 5] (token, x0, y0, x1, y1), token ids [S], or a page image.", nameof(input));
        var matrix = Engine.Reshape(input, new[] { s, input.Length / Math.Max(1, s) });
        var tokens = Engine.Reshape(Engine.TensorSlice(matrix, new[] { 0, 0 }, new[] { s, 1 }), new[] { s });
        var boxes = packed ? Engine.TensorSlice(matrix, new[] { 0, 1 }, new[] { s, 4 }) : null;
        return RunDocFormer(tokens, boxes, null, s, activations);
    }

    /// <summary>
    /// Token logits <c>[S, numClasses]</c> for packed rows <c>[S, 5]</c> (token id, x0, y0, x1, y1) and the page
    /// image <c>[3, H, W]</c>, which drives the ResNet-50 visual stream.
    /// </summary>
    public Tensor<T> PredictDocument(Tensor<T> packedTokens, Tensor<T> pageImage)
    {
        if (packedTokens is null) throw new ArgumentNullException(nameof(packedTokens));
        if (pageImage is null) throw new ArgumentNullException(nameof(pageImage));
        if (!_useNativeMode) throw new NotSupportedException("PredictDocument needs the native model.");
        if (packedTokens.Rank != 2 || packedTokens.Shape[1] != 5)
            throw new ArgumentException("PredictDocument expects packed rows [S, 5] (token, x0, y0, x1, y1).", nameof(packedTokens));
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int s = packedTokens.Shape[0];
        var tokens = Engine.Reshape(Engine.TensorSlice(packedTokens, new[] { 0, 0 }, new[] { s, 1 }), new[] { s });
        var boxes = Engine.TensorSlice(packedTokens, new[] { 0, 1 }, new[] { s, 4 });
        return RunDocFormer(tokens, boxes, PreprocessDocument(pageImage), s);
    }

    /// <inheritdoc/>
    /// <remarks>Packed rows hold token ids and box coordinates on the 0-1023 grid; one bound covers both (token ids above the vocabulary are clamped).</remarks>
    public override LayerInputDomain GetInputDomain(int[]? inputShape)
        => _useNativeMode && _defaultStack && inputShape is { Length: 2 } && inputShape[1] == 5
            ? LayerInputDomain.Indices(Math.Max(_vocabSize, MaxPosition2D))
            : base.GetInputDomain(inputShape);

    /// <inheritdoc/>
    protected override Tensor<T> Forward(Tensor<T> input)
        => _useNativeMode && _defaultStack ? RouteDocFormer(input) : base.Forward(input);

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _useNativeMode && _defaultStack ? RouteDocFormer(input) : base.ForwardForTraining(input);

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (!_useNativeMode || !_defaultStack)
            return base.GetNamedLayerActivations(input);

        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var activations = new Dictionary<string, Tensor<T>>();
        activations["output"] = RouteDocFormer(input, activations);
        return activations;
    }
    // UpdateParameters applied a GRADIENT STEP, but its one-argument form is the value setter and every caller passes values -- the override corrupted the model. Removed under AIDN082.


    /// <summary>
    /// Parameters cannot be written while the model is backed by a loaded ONNX graph: the weights
    /// belong to that graph, not to this instance.
    /// </summary>
    /// <remarks>
    /// Replaces a hand-written throw that used to sit inside UpdateParameters. The base checks this
    /// on every mutating entry point rather than the one member the throw happened to guard, and
    /// reading -- ParameterCount and GetParameters -- stays available either way.
    /// </remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _onnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
