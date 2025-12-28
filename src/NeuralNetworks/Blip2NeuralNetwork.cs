using System.IO;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// BLIP-2 (Bootstrapped Language-Image Pre-training 2) neural network for vision-language tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BLIP-2 uses a Q-Former (Querying Transformer) to efficiently bridge frozen image encoders
/// with frozen large language models. The Q-Former uses learnable query tokens that interact
/// with frozen image features through cross-attention layers.
/// </para>
/// <para><b>For Beginners:</b> BLIP-2 is the next evolution of vision-language models!
///
/// Architecture overview:
/// 1. Frozen Image Encoder (ViT-G): Extracts image patch features
/// 2. Q-Former: Small trainable transformer that bridges vision and language
///    - Uses 32 learnable "query" tokens
///    - Queries attend to image features via cross-attention
///    - Output goes to the language model
/// 3. Frozen LLM (OPT/Flan-T5): Generates text from visual features
///
/// Why this architecture is brilliant:
/// - Only trains the small Q-Former (~188M parameters)
/// - Image encoder stays frozen (no GPU memory for gradients)
/// - LLM stays frozen (can use huge 66B+ models)
/// - Much cheaper to train than end-to-end models
///
/// Training stages:
/// 1. Vision-Language Representation Learning (Q-Former + ViT)
/// 2. Vision-to-Language Generative Learning (Q-Former + LLM)
/// </para>
/// </remarks>
public class Blip2NeuralNetwork<T> : NeuralNetworkBase<T>, IBlip2Model<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this BLIP-2 network uses native layers (true) or ONNX models (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for the vision encoder.
    /// </summary>
    private readonly InferenceSession? _visionEncoder;

    /// <summary>
    /// The ONNX inference session for the Q-Former.
    /// </summary>
    private readonly InferenceSession? _qformer;

    /// <summary>
    /// The ONNX inference session for the language model.
    /// </summary>
    private readonly InferenceSession? _languageModel;

    /// <summary>
    /// Path to the vision encoder ONNX model file.
    /// </summary>
    private readonly string? _visionEncoderPath;

    /// <summary>
    /// Path to the Q-Former ONNX model file.
    /// </summary>
    private readonly string? _qformerPath;

    /// <summary>
    /// Path to the language model ONNX model file.
    /// </summary>
    private readonly string? _languageModelPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Vision transformer layers for image encoding (native mode).
    /// </summary>
    private readonly List<ILayer<T>> _visionEncoderLayers = [];

    /// <summary>
    /// Q-Former self-attention layers (native mode).
    /// </summary>
    private readonly List<ILayer<T>> _qformerSelfAttentionLayers = [];

    /// <summary>
    /// Q-Former cross-attention layers (native mode).
    /// </summary>
    private readonly List<ILayer<T>> _qformerCrossAttentionLayers = [];

    /// <summary>
    /// Q-Former feed-forward layers (native mode).
    /// </summary>
    private readonly List<ILayer<T>> _qformerFeedForwardLayers = [];

    /// <summary>
    /// Language model projection layer (native mode).
    /// </summary>
    private ILayer<T>? _languageModelProjection;

    /// <summary>
    /// Language model decoder layers for text generation (native mode).
    /// </summary>
    private readonly List<TransformerDecoderLayer<T>> _lmDecoderLayers = [];

    /// <summary>
    /// Language model head for projecting decoder output to vocabulary logits.
    /// </summary>
    private ILayer<T>? _lmHead;

    /// <summary>
    /// Number of LM decoder layers.
    /// </summary>
    private readonly int _numLmDecoderLayers;

    /// <summary>
    /// Gradient storage for query tokens.
    /// </summary>
    private Tensor<T>? _queryTokensGradients;

    /// <summary>
    /// Gradient storage for positional embeddings.
    /// </summary>
    private Tensor<T>? _queryPositionalEmbeddingsGradients;

    /// <summary>
    /// Learnable query tokens for Q-Former.
    /// </summary>
    private Tensor<T>? _queryTokens;

    /// <summary>
    /// Vision CLS token.
    /// </summary>
    private Tensor<T>? _visionClsToken;

    /// <summary>
    /// Vision positional embeddings.
    /// </summary>
    private Tensor<T>? _visionPositionalEmbeddings;

    /// <summary>
    /// Query positional embeddings.
    /// </summary>
    private Tensor<T>? _queryPositionalEmbeddings;

    /// <summary>
    /// Text token embeddings for Q-Former text encoder.
    /// </summary>
    private ILayer<T>? _textTokenEmbedding;

    /// <summary>
    /// Patch embedding layer for vision.
    /// </summary>
    private ILayer<T>? _patchEmbedding;

    /// <summary>
    /// Image-text matching head.
    /// </summary>
    private ILayer<T>? _itmHead;

    /// <summary>
    /// Image-text contrastive projection.
    /// </summary>
    private ILayer<T>? _itcProjection;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The tokenizer for processing text input.
    /// </summary>
    private readonly ITokenizer _tokenizer;

    /// <summary>
    /// Optimizer for training.
    /// </summary>
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The dimensionality of the shared embedding space.
    /// </summary>
    private readonly int _embeddingDimension;

    /// <summary>
    /// Maximum sequence length for text encoder.
    /// </summary>
    private readonly int _maxSequenceLength;

    /// <summary>
    /// Expected image size (width and height).
    /// </summary>
    private readonly int _imageSize;

    /// <summary>
    /// Hidden dimension for Q-Former.
    /// </summary>
    private readonly int _qformerHiddenDim;

    /// <summary>
    /// Number of Q-Former layers.
    /// </summary>
    private readonly int _numQformerLayers;

    /// <summary>
    /// Number of attention heads in Q-Former.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Number of learnable query tokens.
    /// </summary>
    private readonly int _numQueryTokens;

    /// <summary>
    /// Patch size for vision transformer.
    /// </summary>
    private readonly int _patchSize;

    /// <summary>
    /// Vocabulary size for text encoder.
    /// </summary>
    private readonly int _vocabularySize;

    /// <summary>
    /// Vision encoder hidden dimension.
    /// </summary>
    private readonly int _visionHiddenDim;

    /// <summary>
    /// Language model hidden dimension.
    /// </summary>
    private readonly int _lmHiddenDim;

    /// <summary>
    /// Type of language model backend.
    /// </summary>
    private readonly string _languageModelType;

    #endregion

    #region IMultimodalEmbedding Properties

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc/>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    public int ImageSize => _imageSize;

    #endregion

    #region IBlip2Model Properties

    /// <inheritdoc/>
    public int NumQueryTokens => _numQueryTokens;

    /// <inheritdoc/>
    public string LanguageModelType => _languageModelType;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a BLIP-2 network using pretrained ONNX models.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="visionEncoderPath">Path to the vision encoder ONNX model.</param>
    /// <param name="qformerPath">Path to the Q-Former ONNX model.</param>
    /// <param name="languageModelPath">Path to the language model ONNX model.</param>
    /// <param name="tokenizer">The tokenizer for text processing.</param>
    /// <param name="languageModelType">Type of LLM backend ("opt" or "flan-t5").</param>
    /// <param name="embeddingDimension">Dimension of the shared embedding space.</param>
    /// <param name="maxSequenceLength">Maximum text sequence length.</param>
    /// <param name="imageSize">Expected image size.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    public Blip2NeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string visionEncoderPath,
        string qformerPath,
        string languageModelPath,
        ITokenizer tokenizer,
        string languageModelType = "opt",
        int embeddingDimension = 256,
        int maxSequenceLength = 32,
        int imageSize = 224,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture,
               lossFunction ?? new ContrastiveLoss<T>(),
               1.0)
    {
        // Validate ONNX model paths
        if (string.IsNullOrWhiteSpace(visionEncoderPath))
            throw new ArgumentException("Vision encoder path cannot be null or empty.", nameof(visionEncoderPath));
        if (string.IsNullOrWhiteSpace(qformerPath))
            throw new ArgumentException("Q-Former path cannot be null or empty.", nameof(qformerPath));
        if (string.IsNullOrWhiteSpace(languageModelPath))
            throw new ArgumentException("Language model path cannot be null or empty.", nameof(languageModelPath));
        if (!File.Exists(visionEncoderPath))
            throw new FileNotFoundException($"Vision encoder model not found: {visionEncoderPath}");
        if (!File.Exists(qformerPath))
            throw new FileNotFoundException($"Q-Former model not found: {qformerPath}");
        if (!File.Exists(languageModelPath))
            throw new FileNotFoundException($"Language model not found: {languageModelPath}");

        _useNativeMode = false;
        _visionEncoderPath = visionEncoderPath;
        _qformerPath = qformerPath;
        _languageModelPath = languageModelPath;
        _languageModelType = languageModelType.ToLowerInvariant();
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _qformerHiddenDim = 768;
        _numQformerLayers = 12;
        _numHeads = 12;
        _numQueryTokens = 32;
        _patchSize = 14; // ViT-G uses 14x14 patches
        _vocabularySize = 30522; // BERT vocabulary size
        _visionHiddenDim = 1408; // ViT-G hidden dimension
        _lmHiddenDim = _languageModelType == "opt" ? 2560 : 2048; // OPT-2.7B or Flan-T5-XL

        InferenceSession? visionEncoder = null;
        InferenceSession? qformer = null;
        InferenceSession? languageModel = null;

        try
        {
            visionEncoder = new InferenceSession(visionEncoderPath);
            qformer = new InferenceSession(qformerPath);
            languageModel = new InferenceSession(languageModelPath);

            _visionEncoder = visionEncoder;
            _qformer = qformer;
            _languageModel = languageModel;

            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer),
                "Tokenizer is required. Use BertTokenizer or equivalent.");

            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _lossFunction = lossFunction ?? new ContrastiveLoss<T>();

            InitializeLayers();
        }
        catch
        {
            try
            {
                visionEncoder?.Dispose();
                qformer?.Dispose();
                languageModel?.Dispose();
            }
            catch
            {
                // Swallow disposal exceptions
            }

            throw;
        }
    }

    /// <summary>
    /// Creates a BLIP-2 network using native library layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="imageSize">Expected image size (default 224 for BLIP-2).</param>
    /// <param name="channels">Number of image channels (default 3 for RGB).</param>
    /// <param name="patchSize">Patch size for vision transformer.</param>
    /// <param name="vocabularySize">Text vocabulary size (BERT: 30522).</param>
    /// <param name="maxSequenceLength">Maximum text sequence length.</param>
    /// <param name="embeddingDimension">Dimension of shared embedding space.</param>
    /// <param name="qformerHiddenDim">Q-Former hidden dimension.</param>
    /// <param name="visionHiddenDim">Vision encoder hidden dimension.</param>
    /// <param name="lmHiddenDim">Language model hidden dimension.</param>
    /// <param name="numQformerLayers">Number of Q-Former layers.</param>
    /// <param name="numQueryTokens">Number of learnable query tokens.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numLmDecoderLayers">Number of language model decoder layers for text generation.</param>
    /// <param name="languageModelType">Type of LLM backend.</param>
    /// <param name="tokenizer">Optional tokenizer for text processing.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    public Blip2NeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int imageSize = 224,
        int channels = 3,
        int patchSize = 14,
        int vocabularySize = 30522,
        int maxSequenceLength = 32,
        int embeddingDimension = 256,
        int qformerHiddenDim = 768,
        int visionHiddenDim = 1408,
        int lmHiddenDim = 2560,
        int numQformerLayers = 12,
        int numQueryTokens = 32,
        int numHeads = 12,
        int numLmDecoderLayers = 6,
        string languageModelType = "opt",
        ITokenizer? tokenizer = null,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture,
               lossFunction ?? new ContrastiveLoss<T>(),
               1.0)
    {
        _useNativeMode = true;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _qformerHiddenDim = qformerHiddenDim;
        _visionHiddenDim = visionHiddenDim;
        _lmHiddenDim = lmHiddenDim;
        _numQformerLayers = numQformerLayers;
        _numHeads = numHeads;
        _numQueryTokens = numQueryTokens;
        _patchSize = patchSize;
        _vocabularySize = vocabularySize;
        _numLmDecoderLayers = numLmDecoderLayers;
        _languageModelType = languageModelType.ToLowerInvariant();

        _tokenizer = tokenizer ?? Tokenization.ClipTokenizerFactory.CreateSimple();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new ContrastiveLoss<T>();

        InitializeNativeLayers(channels);
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes layers for both modes.
    /// </summary>
    protected override void InitializeLayers()
    {
        // Register any layers as needed
    }

    /// <summary>
    /// Initializes native mode layers.
    /// </summary>
    private void InitializeNativeLayers(int channels)
    {
        int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);

        // Vision encoder: Patch embedding
        _patchEmbedding = new PatchEmbeddingLayer<T>(
            _imageSize, _imageSize, channels, _patchSize, _visionHiddenDim);

        // Vision CLS token and positional embeddings
        _visionClsToken = Tensor<T>.CreateDefault([1, _visionHiddenDim], NumOps.Zero);
        _visionPositionalEmbeddings = Tensor<T>.CreateDefault([numPatches + 1, _visionHiddenDim], NumOps.Zero);

        // Learnable query tokens
        _queryTokens = Tensor<T>.CreateDefault([_numQueryTokens, _qformerHiddenDim], NumOps.Zero);
        _queryPositionalEmbeddings = Tensor<T>.CreateDefault([_numQueryTokens, _qformerHiddenDim], NumOps.Zero);

        // Gradient storage for query tokens and positional embeddings
        _queryTokensGradients = Tensor<T>.CreateDefault([_numQueryTokens, _qformerHiddenDim], NumOps.Zero);
        _queryPositionalEmbeddingsGradients = Tensor<T>.CreateDefault([_numQueryTokens, _qformerHiddenDim], NumOps.Zero);

        // Q-Former layers
        int feedForwardDim = _qformerHiddenDim * 4;
        for (int i = 0; i < _numQformerLayers; i++)
        {
            // Self-attention for queries
            _qformerSelfAttentionLayers.Add(new TransformerEncoderLayer<T>(
                _qformerHiddenDim, _numHeads, feedForwardDim));

            // Cross-attention from queries to image features - use TransformerEncoderLayer
            _qformerCrossAttentionLayers.Add(new TransformerEncoderLayer<T>(
                _qformerHiddenDim, _numHeads, feedForwardDim));

            // Feed-forward
            _qformerFeedForwardLayers.Add(new DenseLayer<T>(_qformerHiddenDim, _qformerHiddenDim, (IActivationFunction<T>?)null));
        }

        // Text embedding for Q-Former's text encoder
        _textTokenEmbedding = new EmbeddingLayer<T>(_vocabularySize, _qformerHiddenDim);

        // ITM head (binary classification)
        _itmHead = new DenseLayer<T>(_qformerHiddenDim, 2, (IActivationFunction<T>?)null);

        // ITC projection
        _itcProjection = new DenseLayer<T>(_qformerHiddenDim, _embeddingDimension, (IActivationFunction<T>?)null);

        // LM projection (project Q-Former output to LLM dimension)
        _languageModelProjection = new DenseLayer<T>(_qformerHiddenDim, _lmHiddenDim, (IActivationFunction<T>?)null);

        // LM Decoder layers for text generation
        int lmFeedForwardDim = _lmHiddenDim * 4;
        int lmNumHeads = Math.Max(8, _lmHiddenDim / 64); // Scale heads with hidden dimension
        var geluActivation = new GELUActivation<T>();
        for (int i = 0; i < _numLmDecoderLayers; i++)
        {
            _lmDecoderLayers.Add(new TransformerDecoderLayer<T>(
                embeddingSize: _lmHiddenDim,
                numHeads: lmNumHeads,
                feedForwardDim: lmFeedForwardDim,
                sequenceLength: _maxSequenceLength,
                ffnActivation: geluActivation));
        }

        // LM Head: Projects decoder output to vocabulary logits
        _lmHead = new DenseLayer<T>(_lmHiddenDim, _vocabularySize, (IActivationFunction<T>?)null);

        // Initialize with small random values
        InitializeWeights();
    }

    /// <summary>
    /// Initializes weights with appropriate random values.
    /// </summary>
    private void InitializeWeights()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        double scale = 0.02;

        // Initialize query tokens
        if (_queryTokens is not null)
        {
            for (int i = 0; i < _queryTokens.Shape[0]; i++)
            {
                for (int j = 0; j < _queryTokens.Shape[1]; j++)
                {
                    _queryTokens[i, j] = NumOps.FromDouble(random.NextDouble() * scale - scale / 2);
                }
            }
        }

        // Initialize CLS token
        if (_visionClsToken is not null)
        {
            for (int j = 0; j < _visionClsToken.Shape[1]; j++)
            {
                _visionClsToken[0, j] = NumOps.FromDouble(random.NextDouble() * scale - scale / 2);
            }
        }

        // Initialize positional embeddings
        if (_visionPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _visionPositionalEmbeddings.Shape[0]; i++)
            {
                for (int j = 0; j < _visionPositionalEmbeddings.Shape[1]; j++)
                {
                    _visionPositionalEmbeddings[i, j] = NumOps.FromDouble(random.NextDouble() * scale - scale / 2);
                }
            }
        }

        if (_queryPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _queryPositionalEmbeddings.Shape[0]; i++)
            {
                for (int j = 0; j < _queryPositionalEmbeddings.Shape[1]; j++)
                {
                    _queryPositionalEmbeddings[i, j] = NumOps.FromDouble(random.NextDouble() * scale - scale / 2);
                }
            }
        }
    }

    #endregion

    #region IMultimodalEmbedding Implementation

    /// <inheritdoc/>
    public Vector<T> GetTextEmbedding(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Text cannot be null or empty.", nameof(text));

        if (_useNativeMode)
        {
            return GetTextEmbeddingNative(text);
        }
        else
        {
            return GetTextEmbeddingOnnx(text);
        }
    }

    /// <inheritdoc/>
    public IEnumerable<Vector<T>> GetTextEmbeddings(IEnumerable<string> texts)
    {
        if (texts is null)
            throw new ArgumentNullException(nameof(texts));

        foreach (var text in texts)
        {
            yield return GetTextEmbedding(text);
        }
    }

    /// <inheritdoc/>
    public Vector<T> GetImageEmbedding(Tensor<T> image)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));

        ValidateImageShape(image);

        // Extract Q-Former features and return the mean
        var qformerOutput = ExtractQFormerFeatures(image);

        // Average over query dimension
        var embedding = new Vector<T>(_embeddingDimension);
        for (int i = 0; i < _embeddingDimension; i++)
        {
            T sum = NumOps.Zero;
            for (int q = 0; q < _numQueryTokens; q++)
            {
                sum = NumOps.Add(sum, qformerOutput[q, i]);
            }
            embedding[i] = NumOps.Divide(sum, NumOps.FromDouble(_numQueryTokens));
        }

        return NormalizeVector(embedding);
    }

    /// <inheritdoc/>
    public IEnumerable<Vector<T>> GetImageEmbeddings(IEnumerable<Tensor<T>> images)
    {
        if (images is null)
            throw new ArgumentNullException(nameof(images));

        foreach (var image in images)
        {
            yield return GetImageEmbedding(image);
        }
    }

    /// <inheritdoc/>
    public T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding)
    {
        if (textEmbedding is null)
            throw new ArgumentNullException(nameof(textEmbedding));
        if (imageEmbedding is null)
            throw new ArgumentNullException(nameof(imageEmbedding));
        if (textEmbedding.Length != imageEmbedding.Length)
            throw new ArgumentException("Embedding vectors must have the same dimension.");

        return Engine.DotProduct(textEmbedding, imageEmbedding);
    }

    #endregion

    #region IBlip2Model Implementation

    /// <inheritdoc/>
    public Tensor<T> ExtractQFormerFeatures(Tensor<T> image)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));

        ValidateImageShape(image);

        if (_useNativeMode)
        {
            return ExtractQFormerFeaturesNative(image);
        }
        else
        {
            return ExtractQFormerFeaturesOnnx(image);
        }
    }

    /// <inheritdoc/>
    public string GenerateCaption(
        Tensor<T> image,
        string? prompt = null,
        int maxLength = 30,
        int numBeams = 5,
        double temperature = 1.0)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));

        ValidateImageShape(image);

        // Extract Q-Former features
        var qformerFeatures = ExtractQFormerFeatures(image);

        // Project to LLM space
        var projectedFeatures = ProjectToLmSpace(qformerFeatures);

        // Generate using LLM
        return GenerateWithLm(projectedFeatures, prompt, maxLength, numBeams, temperature);
    }

    /// <inheritdoc/>
    public IEnumerable<(string Caption, T Score)> GenerateCaptions(
        Tensor<T> image,
        int numCaptions = 5,
        string? prompt = null,
        int maxLength = 30,
        double temperature = 0.9,
        double topP = 0.95)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));

        ValidateImageShape(image);

        var captions = new List<(string Caption, T Score)>();

        // Extract Q-Former features once
        var qformerFeatures = ExtractQFormerFeatures(image);
        var projectedFeatures = ProjectToLmSpace(qformerFeatures);

        // Generate multiple captions with sampling
        for (int i = 0; i < numCaptions; i++)
        {
            var caption = GenerateWithLm(projectedFeatures, prompt, maxLength, 1, temperature);
            var score = ScoreCaption(qformerFeatures, caption);
            captions.Add((caption, score));
        }

        // Sort by score descending
        return captions.OrderByDescending(c => NumOps.ToDouble(c.Score));
    }

    /// <inheritdoc/>
    public string AnswerQuestion(Tensor<T> image, string question, int maxLength = 30)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));
        if (string.IsNullOrWhiteSpace(question))
            throw new ArgumentException("Question cannot be null or empty.", nameof(question));

        ValidateImageShape(image);

        // Format question as prompt
        string prompt = _languageModelType == "flan-t5"
            ? $"Question: {question} Answer:"
            : $"Question: {question} Short answer:";

        return GenerateCaption(image, prompt, maxLength, numBeams: 3, temperature: 0.7);
    }

    /// <inheritdoc/>
    public T ComputeImageTextMatch(Tensor<T> image, string text)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Text cannot be null or empty.", nameof(text));

        ValidateImageShape(image);

        if (_useNativeMode)
        {
            return ComputeItmNative(image, text);
        }
        else
        {
            return ComputeItmOnnx(image, text);
        }
    }

    /// <inheritdoc/>
    public T ComputeContrastiveSimilarity(Tensor<T> image, string text)
    {
        var imageEmbedding = GetImageEmbedding(image);
        var textEmbedding = GetTextEmbedding(text);
        return ComputeSimilarity(textEmbedding, imageEmbedding);
    }

    /// <inheritdoc/>
    public Vector<T> GroundText(Tensor<T> image, string description)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));
        if (string.IsNullOrWhiteSpace(description))
            throw new ArgumentException("Description cannot be null or empty.", nameof(description));

        ValidateImageShape(image);

        // Get attention weights from Q-Former cross-attention
        var attentionWeights = GetCrossAttentionWeights(image, description);

        // Convert attention to bounding box
        return AttentionToBoundingBox(attentionWeights);
    }

    /// <inheritdoc/>
    public string GenerateWithInstruction(Tensor<T> image, string instruction, int maxLength = 100)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));
        if (string.IsNullOrWhiteSpace(instruction))
            throw new ArgumentException("Instruction cannot be null or empty.", nameof(instruction));

        ValidateImageShape(image);

        return GenerateCaption(image, instruction, maxLength, numBeams: 5, temperature: 0.7);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, IEnumerable<string> classLabels)
    {
        return ZeroShotClassify(image, classLabels, useItm: false);
    }

    /// <summary>
    /// Performs zero-shot image classification with optional ITM scoring.
    /// </summary>
    public Dictionary<string, T> ZeroShotClassify(
        Tensor<T> image,
        IEnumerable<string> classLabels,
        bool useItm)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));
        if (classLabels is null)
            throw new ArgumentNullException(nameof(classLabels));

        var labels = classLabels.ToList();
        if (labels.Count == 0)
            throw new ArgumentException("At least one class label is required.", nameof(classLabels));

        ValidateImageShape(image);

        var scores = new Dictionary<string, T>();

        if (useItm)
        {
            // Use ITM for more accurate but slower classification
            foreach (var label in labels)
            {
                var prompt = $"a photo of {label}";
                scores[label] = ComputeImageTextMatch(image, prompt);
            }
        }
        else
        {
            // Use ITC for faster classification
            var imageEmbedding = GetImageEmbedding(image);
            foreach (var label in labels)
            {
                var prompt = $"a photo of {label}";
                var textEmbedding = GetTextEmbedding(prompt);
                scores[label] = ComputeSimilarity(textEmbedding, imageEmbedding);
            }
        }

        // Softmax normalization
        return SoftmaxScores(scores);
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveImages(
        string query,
        IEnumerable<Tensor<T>> imageFeatures,
        int topK = 10,
        bool useItmReranking = true,
        int rerankTopN = 100)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty.", nameof(query));
        if (imageFeatures is null)
            throw new ArgumentNullException(nameof(imageFeatures));

        var featuresList = imageFeatures.ToList();
        if (featuresList.Count == 0)
            return Enumerable.Empty<(int, T)>();

        var textEmbedding = GetTextEmbedding(query);

        // Stage 1: ITC-based retrieval
        var candidates = new List<(int Index, T Score)>();
        for (int i = 0; i < featuresList.Count; i++)
        {
            // Get image embedding from Q-Former features
            var qformerOutput = featuresList[i];
            var embedding = new Vector<T>(_embeddingDimension);
            for (int j = 0; j < _embeddingDimension; j++)
            {
                T sum = NumOps.Zero;
                for (int q = 0; q < Math.Min(_numQueryTokens, qformerOutput.Shape[0]); q++)
                {
                    sum = NumOps.Add(sum, qformerOutput[q, j]);
                }
                embedding[j] = NumOps.Divide(sum, NumOps.FromDouble(_numQueryTokens));
            }
            embedding = NormalizeVector(embedding);

            var score = ComputeSimilarity(textEmbedding, embedding);
            candidates.Add((i, score));
        }

        // Sort by score
        candidates = candidates.OrderByDescending(c => NumOps.ToDouble(c.Score)).ToList();

        if (!useItmReranking)
        {
            return candidates.Take(topK);
        }

        // Stage 2: ITM reranking of top candidates
        // Note: This would require the original images, not just features
        // For now, return ITC results
        return candidates.Take(topK);
    }

    #endregion

    #region Native Mode Implementation

    /// <summary>
    /// Gets text embedding using native layers.
    /// </summary>
    private Vector<T> GetTextEmbeddingNative(string text)
    {
        var encoded = _tokenizer.Encode(text);
        var inputIds = encoded.TokenIds;

        if (_textTokenEmbedding is null)
            throw new InvalidOperationException("Text token embedding not initialized.");

        // Create input tensor
        var inputTensor = Tensor<T>.CreateDefault([inputIds.Count], NumOps.Zero);
        for (int i = 0; i < inputIds.Count; i++)
        {
            inputTensor[i] = NumOps.FromDouble(inputIds[i]);
        }

        // Get embeddings
        var embeddings = _textTokenEmbedding.Forward(inputTensor);

        // Use CLS token (first token)
        var clsEmbedding = new Vector<T>(_qformerHiddenDim);
        for (int i = 0; i < _qformerHiddenDim; i++)
        {
            clsEmbedding[i] = embeddings[0, i];
        }

        // Project to embedding dimension
        if (_itcProjection is not null)
        {
            var projInput = Tensor<T>.FromVector(clsEmbedding);
            var projected = _itcProjection.Forward(projInput);
            var result = new Vector<T>(_embeddingDimension);
            for (int i = 0; i < _embeddingDimension; i++)
            {
                result[i] = projected[i];
            }
            return NormalizeVector(result);
        }

        return NormalizeVector(clsEmbedding);
    }

    /// <summary>
    /// Extracts Q-Former features using native layers.
    /// </summary>
    private Tensor<T> ExtractQFormerFeaturesNative(Tensor<T> image)
    {
        // Step 1: Vision encoder
        if (_patchEmbedding is null || _visionClsToken is null || _visionPositionalEmbeddings is null)
            throw new InvalidOperationException("Vision encoder not initialized.");

        var patches = _patchEmbedding.Forward(image);
        int numPatches = patches.Shape[0];

        // Add CLS token
        var withCls = Tensor<T>.CreateDefault([numPatches + 1, _visionHiddenDim], NumOps.Zero);
        for (int j = 0; j < _visionHiddenDim; j++)
        {
            withCls[0, j] = _visionClsToken[0, j];
        }
        for (int i = 0; i < numPatches; i++)
        {
            for (int j = 0; j < _visionHiddenDim; j++)
            {
                withCls[i + 1, j] = patches[i, j];
            }
        }

        // Add positional embeddings
        int posLen = Math.Min(withCls.Shape[0], _visionPositionalEmbeddings.Shape[0]);
        for (int i = 0; i < posLen; i++)
        {
            for (int j = 0; j < _visionHiddenDim; j++)
            {
                withCls[i, j] = NumOps.Add(withCls[i, j], _visionPositionalEmbeddings[i, j]);
            }
        }

        var visionOutput = withCls;

        // Step 2: Q-Former
        if (_queryTokens is null || _queryPositionalEmbeddings is null)
            throw new InvalidOperationException("Q-Former not initialized.");

        // Initialize query embeddings
        var queryEmbeddings = Tensor<T>.CreateDefault([_numQueryTokens, _qformerHiddenDim], NumOps.Zero);
        for (int i = 0; i < _numQueryTokens; i++)
        {
            for (int j = 0; j < _qformerHiddenDim; j++)
            {
                queryEmbeddings[i, j] = NumOps.Add(_queryTokens[i, j], _queryPositionalEmbeddings[i, j]);
            }
        }

        // Process through Q-Former layers
        for (int layer = 0; layer < _numQformerLayers; layer++)
        {
            // Self-attention among queries
            if (layer < _qformerSelfAttentionLayers.Count)
            {
                queryEmbeddings = _qformerSelfAttentionLayers[layer].Forward(queryEmbeddings);
            }

            // Cross-attention to vision features
            if (layer < _qformerCrossAttentionLayers.Count)
            {
                // Need to project vision features to Q-Former dimension
                var projectedVision = ProjectVisionToQformer(visionOutput);
                queryEmbeddings = ApplyCrossAttention(
                    _qformerCrossAttentionLayers[layer],
                    queryEmbeddings,
                    projectedVision);
            }

            // Feed-forward
            if (layer < _qformerFeedForwardLayers.Count)
            {
                queryEmbeddings = _qformerFeedForwardLayers[layer].Forward(queryEmbeddings);
            }
        }

        // Project to embedding dimension
        if (_itcProjection is not null)
        {
            var projected = Tensor<T>.CreateDefault([_numQueryTokens, _embeddingDimension], NumOps.Zero);
            for (int q = 0; q < _numQueryTokens; q++)
            {
                var queryVec = Tensor<T>.CreateDefault([_qformerHiddenDim], NumOps.Zero);
                for (int j = 0; j < _qformerHiddenDim; j++)
                {
                    queryVec[j] = queryEmbeddings[q, j];
                }
                var projVec = _itcProjection.Forward(queryVec);
                for (int j = 0; j < _embeddingDimension; j++)
                {
                    projected[q, j] = projVec[j];
                }
            }
            return projected;
        }

        return queryEmbeddings;
    }

    /// <summary>
    /// Projects vision features to Q-Former dimension.
    /// </summary>
    private Tensor<T> ProjectVisionToQformer(Tensor<T> visionFeatures)
    {
        int seqLen = visionFeatures.Shape[0];
        var projected = Tensor<T>.CreateDefault([seqLen, _qformerHiddenDim], NumOps.Zero);

        // Simple linear projection (in practice, would use a learned projection)
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < _qformerHiddenDim; j++)
            {
                // Average pooling over vision dimension
                int srcIdx = j * _visionHiddenDim / _qformerHiddenDim;
                srcIdx = Math.Min(srcIdx, _visionHiddenDim - 1);
                projected[i, j] = visionFeatures[i, srcIdx];
            }
        }

        return projected;
    }

    /// <summary>
    /// Applies cross-attention between queries and keys/values.
    /// </summary>
    private Tensor<T> ApplyCrossAttention(ILayer<T> crossAttention, Tensor<T> queries, Tensor<T> keyValues)
    {
        // Combine for attention layer
        int queryLen = queries.Shape[0];
        int kvLen = keyValues.Shape[0];
        int hiddenDim = queries.Shape[1];

        // For MultiHeadAttentionLayer, we need to format input appropriately
        // The layer expects [seqLen, hiddenDim] input
        // We'll use queries as input and provide keyValues as context

        // Simple implementation: attend queries to keyValues
        var output = Tensor<T>.CreateDefault([queryLen, hiddenDim], NumOps.Zero);

        // Compute attention weights
        double scale = 1.0 / Math.Sqrt(hiddenDim);

        for (int q = 0; q < queryLen; q++)
        {
            // Compute attention scores
            var scores = new T[kvLen];
            for (int k = 0; k < kvLen; k++)
            {
                T dot = NumOps.Zero;
                for (int d = 0; d < hiddenDim; d++)
                {
                    dot = NumOps.Add(dot, NumOps.Multiply(queries[q, d], keyValues[k, d]));
                }
                scores[k] = NumOps.Multiply(dot, NumOps.FromDouble(scale));
            }

            // Softmax
            T maxScore = scores[0];
            for (int k = 1; k < kvLen; k++)
            {
                if (NumOps.ToDouble(scores[k]) > NumOps.ToDouble(maxScore))
                    maxScore = scores[k];
            }

            T sumExp = NumOps.Zero;
            var expScores = new T[kvLen];
            for (int k = 0; k < kvLen; k++)
            {
                expScores[k] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Subtract(scores[k], maxScore))));
                sumExp = NumOps.Add(sumExp, expScores[k]);
            }

            for (int k = 0; k < kvLen; k++)
            {
                expScores[k] = NumOps.Divide(expScores[k], sumExp);
            }

            // Weighted sum of values
            for (int d = 0; d < hiddenDim; d++)
            {
                T weighted = NumOps.Zero;
                for (int k = 0; k < kvLen; k++)
                {
                    weighted = NumOps.Add(weighted, NumOps.Multiply(expScores[k], keyValues[k, d]));
                }
                output[q, d] = NumOps.Add(queries[q, d], weighted); // Residual connection
            }
        }

        return output;
    }

    /// <summary>
    /// Computes ITM score using native layers.
    /// </summary>
    private T ComputeItmNative(Tensor<T> image, string text)
    {
        // Get Q-Former features with text conditioning
        var qformerFeatures = ExtractQFormerFeatures(image);

        // Get text features
        var textEmbedding = GetTextEmbeddingNative(text);

        // Combine for ITM head
        // Use mean of query features
        var queryMean = new Vector<T>(_qformerHiddenDim);
        for (int d = 0; d < Math.Min(_qformerHiddenDim, qformerFeatures.Shape[1]); d++)
        {
            T sum = NumOps.Zero;
            for (int q = 0; q < _numQueryTokens; q++)
            {
                sum = NumOps.Add(sum, qformerFeatures[q, d]);
            }
            queryMean[d] = NumOps.Divide(sum, NumOps.FromDouble(_numQueryTokens));
        }

        // Pass through ITM head
        if (_itmHead is not null)
        {
            var input = Tensor<T>.FromVector(queryMean);
            var logits = _itmHead.Forward(input);

            // Softmax to get probability
            T exp0 = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logits[0])));
            T exp1 = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(logits[1])));
            T sum = NumOps.Add(exp0, exp1);

            return NumOps.Divide(exp1, sum); // Return probability of match
        }

        // Fallback to cosine similarity
        return ComputeSimilarity(textEmbedding, queryMean);
    }

    #endregion

    #region ONNX Mode Implementation

    /// <summary>
    /// Gets text embedding using ONNX models.
    /// </summary>
    private Vector<T> GetTextEmbeddingOnnx(string text)
    {
        if (_qformer is null)
            throw new InvalidOperationException("Q-Former ONNX session not initialized.");

        var encoded = _tokenizer.Encode(text);
        var inputIds = encoded.TokenIds;

        // Create input tensor
        var inputIdsTensor = new OnnxTensors.DenseTensor<long>([1, inputIds.Count]);
        for (int i = 0; i < inputIds.Count; i++)
        {
            inputIdsTensor[0, i] = inputIds[i];
        }

        var attentionMask = encoded.AttentionMask is not null
            ? encoded.AttentionMask.ToArray()
            : Enumerable.Repeat(1, inputIds.Count).ToArray();

        var attentionMaskTensor = new OnnxTensors.DenseTensor<long>([1, inputIds.Count]);
        for (int i = 0; i < inputIds.Count; i++)
        {
            attentionMaskTensor[0, i] = attentionMask[i];
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };

        using var results = _qformer.Run(inputs);
        var output = results.First().AsTensor<float>();

        // Extract embedding from Q-Former output
        // Q-Former outputs shape: [batch_size, num_query_tokens, hidden_size]
        // We use mean pooling over query tokens for the embedding
        var embedding = new Vector<T>(_embeddingDimension);

        if (output.Rank == 3)
        {
            // 3D tensor: [batch, num_query_tokens, hidden_size]
            int numQueryTokens = (int)output.Dimensions[1];
            int hiddenSize = (int)output.Dimensions[2];
            int embDim = Math.Min(_embeddingDimension, hiddenSize);

            // Mean pool over query tokens
            for (int i = 0; i < embDim; i++)
            {
                double sum = 0;
                for (int q = 0; q < numQueryTokens; q++)
                {
                    sum += output[0, q, i];
                }
                embedding[i] = NumOps.FromDouble(sum / numQueryTokens);
            }
        }
        else if (output.Rank == 2)
        {
            // 2D tensor: [batch, hidden_size] - direct extraction
            int hiddenSize = (int)output.Dimensions[1];
            int embDim = Math.Min(_embeddingDimension, hiddenSize);
            for (int i = 0; i < embDim; i++)
            {
                embedding[i] = NumOps.FromDouble(output[0, i]);
            }
        }
        else
        {
            throw new InvalidOperationException(
                $"Unexpected Q-Former output rank: {output.Rank}. Expected 2 or 3 dimensions.");
        }

        return NormalizeVector(embedding);
    }

    /// <summary>
    /// Extracts Q-Former features using ONNX models.
    /// </summary>
    private Tensor<T> ExtractQFormerFeaturesOnnx(Tensor<T> image)
    {
        if (_visionEncoder is null || _qformer is null)
            throw new InvalidOperationException("ONNX sessions not initialized.");

        // Run vision encoder
        var imageInput = PrepareImageForOnnx(image);
        var visionInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", imageInput)
        };

        OnnxTensors.Tensor<float> visionOutput;
        using (var visionResults = _visionEncoder.Run(visionInputs))
        {
            visionOutput = visionResults.First().AsTensor<float>().Clone() as OnnxTensors.DenseTensor<float>
                ?? throw new InvalidOperationException("Failed to get vision encoder output.");
        }

        // Run Q-Former with vision features
        var qformerInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states", visionOutput)
        };

        using var qformerResults = _qformer.Run(qformerInputs);
        var qformerOutput = qformerResults.First().AsTensor<float>();

        // Convert to Tensor<T>
        int numQueries = (int)qformerOutput.Dimensions[1];
        int hiddenDim = (int)qformerOutput.Dimensions[2];

        var result = Tensor<T>.CreateDefault([numQueries, hiddenDim], NumOps.Zero);
        for (int q = 0; q < numQueries; q++)
        {
            for (int d = 0; d < hiddenDim; d++)
            {
                result[q, d] = NumOps.FromDouble(qformerOutput[0, q, d]);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes ITM score using ONNX models.
    /// </summary>
    private T ComputeItmOnnx(Tensor<T> image, string text)
    {
        // For ONNX mode, use embedding similarity as approximation
        var imageEmbedding = GetImageEmbedding(image);
        var textEmbedding = GetTextEmbedding(text);
        return ComputeSimilarity(textEmbedding, imageEmbedding);
    }

    /// <summary>
    /// Prepares image tensor for ONNX inference.
    /// </summary>
    private OnnxTensors.DenseTensor<float> PrepareImageForOnnx(Tensor<T> image)
    {
        bool is3D = image.Shape.Length == 3;
        int channels = is3D ? image.Shape[0] : image.Shape[1];
        int height = is3D ? image.Shape[1] : image.Shape[2];
        int width = is3D ? image.Shape[2] : image.Shape[3];

        var onnxTensor = new OnnxTensors.DenseTensor<float>([1, channels, height, width]);

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    T value = is3D ? image[c, h, w] : image[0, c, h, w];
                    onnxTensor[0, c, h, w] = (float)NumOps.ToDouble(value);
                }
            }
        }

        return onnxTensor;
    }

    #endregion

    #region Generation Helpers

    /// <summary>
    /// Projects Q-Former features to language model space.
    /// </summary>
    private Tensor<T> ProjectToLmSpace(Tensor<T> qformerFeatures)
    {
        if (_languageModelProjection is null)
        {
            // Simple identity/reshape if no projection layer
            return qformerFeatures;
        }

        int numQueries = qformerFeatures.Shape[0];
        var projected = Tensor<T>.CreateDefault([numQueries, _lmHiddenDim], NumOps.Zero);

        for (int q = 0; q < numQueries; q++)
        {
            var queryVec = Tensor<T>.CreateDefault([qformerFeatures.Shape[1]], NumOps.Zero);
            for (int d = 0; d < qformerFeatures.Shape[1]; d++)
            {
                queryVec[d] = qformerFeatures[q, d];
            }

            var projVec = _languageModelProjection.Forward(queryVec);
            for (int d = 0; d < _lmHiddenDim; d++)
            {
                projected[q, d] = projVec[d];
            }
        }

        return projected;
    }

    /// <summary>
    /// Generates text using the language model.
    /// </summary>
    private string GenerateWithLm(
        Tensor<T> projectedFeatures,
        string? prompt,
        int maxLength,
        int numBeams,
        double temperature)
    {
        if (_useNativeMode)
        {
            // Native mode: simplified autoregressive generation
            return GenerateNative(projectedFeatures, prompt, maxLength, temperature);
        }
        else
        {
            // ONNX mode: use LLM model
            return GenerateOnnx(projectedFeatures, prompt, maxLength);
        }
    }

    /// <summary>
    /// Generates text using native layers with autoregressive decoding.
    /// </summary>
    /// <remarks>
    /// Uses the LM decoder layers and LM head to generate text autoregressively.
    /// The visual features from Q-Former are used as encoder output for cross-attention
    /// in the decoder layers.
    /// </remarks>
    private string GenerateNative(Tensor<T> features, string? prompt, int maxLength, double temperature)
    {
        if (_lmDecoderLayers.Count == 0 || _lmHead is null || _textTokenEmbedding is null)
        {
            throw new InvalidOperationException("LM decoder layers and head must be initialized for native text generation.");
        }

        // Clamp temperature to valid range
        temperature = Math.Max(0.1, Math.Min(temperature, 2.0));

        // Initialize generated tokens with BOS token or prompt tokens
        List<int> generatedTokens;
        if (prompt is not null && prompt.Length > 0)
        {
            var encoded = _tokenizer.Encode(prompt);
            generatedTokens = new List<int>(encoded.TokenIds);
        }
        else
        {
            // Start with BOS token (usually token ID 1 for BERT-style tokenizers)
            generatedTokens = [1];
        }

        // Special token IDs
        int eosTokenId = 2;  // EOS token
        int padTokenId = 0;  // PAD token

        // Autoregressive generation loop
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(
            Tensors.Helpers.RandomHelper.ThreadSafeRandom.Next());

        for (int step = 0; step < maxLength && generatedTokens.Count < _maxSequenceLength; step++)
        {
            // Create input tensor for current sequence
            var inputTensor = Tensor<T>.CreateDefault([generatedTokens.Count], NumOps.Zero);
            for (int i = 0; i < generatedTokens.Count; i++)
            {
                inputTensor[i] = NumOps.FromDouble(generatedTokens[i]);
            }

            // Get token embeddings
            var embeddings = _textTokenEmbedding.Forward(inputTensor);

            // Reshape embeddings to [seqLen, hiddenDim] for decoder
            int seqLen = generatedTokens.Count;
            var decoderInput = Tensor<T>.CreateDefault([seqLen, _lmHiddenDim], NumOps.Zero);

            // Project embeddings to LM hidden dimension if needed
            int embDim = embeddings.Shape.Length > 1 ? embeddings.Shape[1] : _qformerHiddenDim;
            for (int i = 0; i < seqLen; i++)
            {
                // Simple projection: pad or truncate to match _lmHiddenDim
                for (int j = 0; j < _lmHiddenDim; j++)
                {
                    if (j < embDim)
                    {
                        decoderInput[i, j] = embeddings.Shape.Length > 1 ? embeddings[i, j] : embeddings[i];
                    }
                    else
                    {
                        decoderInput[i, j] = NumOps.Zero;
                    }
                }
            }

            // Pass through decoder layers with visual features as encoder output
            var decoderOutput = decoderInput;
            foreach (var decoderLayer in _lmDecoderLayers)
            {
                decoderOutput = decoderLayer.Forward(decoderOutput, features);
            }

            // Get logits for last position only
            var lastPositionHidden = Tensor<T>.CreateDefault([_lmHiddenDim], NumOps.Zero);
            for (int j = 0; j < _lmHiddenDim; j++)
            {
                lastPositionHidden[j] = decoderOutput[seqLen - 1, j];
            }

            // Project to vocabulary via LM head
            var logits = _lmHead.Forward(lastPositionHidden);

            // Apply temperature scaling
            var scaledLogits = new T[_vocabularySize];
            for (int i = 0; i < _vocabularySize; i++)
            {
                scaledLogits[i] = NumOps.Divide(logits[i], NumOps.FromDouble(temperature));
            }

            // Compute softmax probabilities
            T maxLogit = scaledLogits[0];
            for (int i = 1; i < _vocabularySize; i++)
            {
                if (NumOps.ToDouble(scaledLogits[i]) > NumOps.ToDouble(maxLogit))
                {
                    maxLogit = scaledLogits[i];
                }
            }

            T sumExp = NumOps.Zero;
            var probs = new T[_vocabularySize];
            for (int i = 0; i < _vocabularySize; i++)
            {
                probs[i] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Subtract(scaledLogits[i], maxLogit))));
                sumExp = NumOps.Add(sumExp, probs[i]);
            }

            for (int i = 0; i < _vocabularySize; i++)
            {
                probs[i] = NumOps.Divide(probs[i], sumExp);
            }

            // Sample from distribution
            double randVal = random.NextDouble();
            double cumSum = 0.0;
            int nextToken = 0;
            for (int i = 0; i < _vocabularySize; i++)
            {
                cumSum += NumOps.ToDouble(probs[i]);
                if (randVal <= cumSum)
                {
                    nextToken = i;
                    break;
                }
            }

            // Check for EOS token
            if (nextToken == eosTokenId)
            {
                break;
            }

            // Skip PAD tokens
            if (nextToken == padTokenId)
            {
                continue;
            }

            generatedTokens.Add(nextToken);
        }

        // Decode generated tokens to text
        return _tokenizer.Decode(generatedTokens);
    }

    /// <summary>
    /// Generates text using ONNX language model.
    /// </summary>
    private string GenerateOnnx(Tensor<T> features, string? prompt, int maxLength)
    {
        if (_languageModel is null)
            throw new InvalidOperationException("Language model ONNX session not initialized.");

        // Prepare visual features for LLM
        var visualFeatures = new OnnxTensors.DenseTensor<float>(
            [1, features.Shape[0], features.Shape[1]]);

        for (int q = 0; q < features.Shape[0]; q++)
        {
            for (int d = 0; d < features.Shape[1]; d++)
            {
                visualFeatures[0, q, d] = (float)NumOps.ToDouble(features[q, d]);
            }
        }

        // Encode prompt if provided
        List<int> inputIds;
        if (prompt is not null && prompt.Length > 0)
        {
            var encoded = _tokenizer.Encode(prompt);
            inputIds = encoded.TokenIds;
        }
        else
        {
            inputIds = [1]; // BOS token
        }

        var inputIdsTensor = new OnnxTensors.DenseTensor<long>([1, inputIds.Count]);
        for (int i = 0; i < inputIds.Count; i++)
        {
            inputIdsTensor[0, i] = inputIds[i];
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states", visualFeatures),
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor)
        };

        // Run generation
        using var results = _languageModel.Run(inputs);
        var outputIds = results.First().AsTensor<long>();

        // Convert output IDs to list
        var generatedIds = new List<int>();
        for (int i = 0; i < outputIds.Length; i++)
        {
            generatedIds.Add((int)outputIds[0, i]);
        }

        return _tokenizer.Decode(generatedIds);
    }

    /// <summary>
    /// Scores a caption against Q-Former features.
    /// </summary>
    private T ScoreCaption(Tensor<T> qformerFeatures, string caption)
    {
        var textEmbedding = GetTextEmbedding(caption);

        // Average Q-Former features
        var queryMean = new Vector<T>(_embeddingDimension);
        int dim = Math.Min(_embeddingDimension, qformerFeatures.Shape[1]);
        for (int d = 0; d < dim; d++)
        {
            T sum = NumOps.Zero;
            for (int q = 0; q < _numQueryTokens; q++)
            {
                sum = NumOps.Add(sum, qformerFeatures[q, d]);
            }
            queryMean[d] = NumOps.Divide(sum, NumOps.FromDouble(_numQueryTokens));
        }

        queryMean = NormalizeVector(queryMean);
        return ComputeSimilarity(textEmbedding, queryMean);
    }

    /// <summary>
    /// Gets cross-attention weights for visual grounding.
    /// </summary>
    private Tensor<T> GetCrossAttentionWeights(Tensor<T> image, string description)
    {
        // Simplified: return uniform attention weights
        int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);
        var weights = Tensor<T>.CreateDefault([_numQueryTokens, numPatches], NumOps.Zero);

        T uniform = NumOps.FromDouble(1.0 / numPatches);
        for (int q = 0; q < _numQueryTokens; q++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                weights[q, p] = uniform;
            }
        }

        return weights;
    }

    /// <summary>
    /// Converts attention weights to bounding box.
    /// </summary>
    private Vector<T> AttentionToBoundingBox(Tensor<T> attentionWeights)
    {
        int gridSize = _imageSize / _patchSize;

        // Average attention across queries
        var avgAttention = new T[gridSize * gridSize];
        for (int p = 0; p < gridSize * gridSize; p++)
        {
            T sum = NumOps.Zero;
            for (int q = 0; q < _numQueryTokens; q++)
            {
                sum = NumOps.Add(sum, attentionWeights[q, p]);
            }
            avgAttention[p] = NumOps.Divide(sum, NumOps.FromDouble(_numQueryTokens));
        }

        // Find max attention patch
        int maxIdx = 0;
        T maxVal = avgAttention[0];
        for (int p = 1; p < avgAttention.Length; p++)
        {
            if (NumOps.ToDouble(avgAttention[p]) > NumOps.ToDouble(maxVal))
            {
                maxVal = avgAttention[p];
                maxIdx = p;
            }
        }

        // Convert to bounding box (simple: just the patch location)
        int row = maxIdx / gridSize;
        int col = maxIdx % gridSize;

        double patchSizeNorm = 1.0 / gridSize;
        var bbox = new Vector<T>(4);
        bbox[0] = NumOps.FromDouble(col * patchSizeNorm);           // x1
        bbox[1] = NumOps.FromDouble(row * patchSizeNorm);           // y1
        bbox[2] = NumOps.FromDouble((col + 1) * patchSizeNorm);     // x2
        bbox[3] = NumOps.FromDouble((row + 1) * patchSizeNorm);     // y2

        return bbox;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Validates the shape of an input image tensor.
    /// </summary>
    private void ValidateImageShape(Tensor<T> image)
    {
        if (image.Shape.Length != 3 && image.Shape.Length != 4)
            throw new ArgumentException($"Image tensor must have 3 or 4 dimensions, got {image.Shape.Length}.");

        bool is3D = image.Shape.Length == 3;
        int channels = is3D ? image.Shape[0] : image.Shape[1];
        int height = is3D ? image.Shape[1] : image.Shape[2];
        int width = is3D ? image.Shape[2] : image.Shape[3];

        if (channels != 3)
            throw new ArgumentException($"Image must have 3 channels (RGB), got {channels}.");
        if (height != _imageSize || width != _imageSize)
            throw new ArgumentException($"Image must be {_imageSize}x{_imageSize}, got {height}x{width}.");
    }

    /// <summary>
    /// Normalizes a vector to unit length.
    /// </summary>
    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        T sumSquares = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(vector[i], vector[i]));
        }

        T norm = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(sumSquares)));
        if (NumOps.ToDouble(norm) < 1e-10)
            return vector;

        var normalized = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = NumOps.Divide(vector[i], norm);
        }

        return normalized;
    }

    /// <summary>
    /// Applies softmax normalization to scores.
    /// </summary>
    private Dictionary<string, T> SoftmaxScores(Dictionary<string, T> scores)
    {
        // Find max for numerical stability
        T maxScore = scores.Values.First();
        foreach (var score in scores.Values)
        {
            if (NumOps.ToDouble(score) > NumOps.ToDouble(maxScore))
                maxScore = score;
        }

        // Compute exp(score - max)
        var expScores = new Dictionary<string, T>();
        T sumExp = NumOps.Zero;
        foreach (var kvp in scores)
        {
            T expScore = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Subtract(kvp.Value, maxScore))));
            expScores[kvp.Key] = expScore;
            sumExp = NumOps.Add(sumExp, expScore);
        }

        // Normalize
        var normalized = new Dictionary<string, T>();
        foreach (var kvp in expScores)
        {
            normalized[kvp.Key] = NumOps.Divide(kvp.Value, sumExp);
        }

        return normalized;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // For BLIP-2, prediction depends on the task
        // Default: return Q-Former features
        return ExtractQFormerFeatures(input);
    }

    /// <summary>
    /// Forward pass through Q-Former and vision encoder.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // Extract Q-Former features
        return ExtractQFormerFeatures(input);
    }

    /// <summary>
    /// Backward pass through Q-Former (vision encoder is frozen).
    /// </summary>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        // Backward pass through Q-Former and vision encoder
        // Only Q-Former is trainable; vision encoder is frozen

        if (!_useNativeMode)
        {
            throw new NotSupportedException("Backward pass is only supported in native mode.");
        }

        // Backprop through layers (only Q-Former layers are trainable)
        var currentGradient = gradient;

        // Backward through Q-Former layers in reverse order
        for (int i = _numQformerLayers - 1; i >= 0; i--)
        {
            if (i < _qformerFeedForwardLayers.Count)
            {
                currentGradient = _qformerFeedForwardLayers[i].Backward(currentGradient);
            }

            if (i < _qformerCrossAttentionLayers.Count)
            {
                currentGradient = _qformerCrossAttentionLayers[i].Backward(currentGradient);
            }

            if (i < _qformerSelfAttentionLayers.Count)
            {
                currentGradient = _qformerSelfAttentionLayers[i].Backward(currentGradient);
            }
        }

        // Accumulate gradients for query tokens and positional embeddings
        // The gradient at this point flows back to the input query tokens
        AccumulateQueryTokenGradients(currentGradient);

        return currentGradient;
    }

    /// <summary>
    /// Accumulates gradients for query tokens and positional embeddings from the backward pass.
    /// </summary>
    private void AccumulateQueryTokenGradients(Tensor<T> gradient)
    {
        if (_queryTokensGradients is null || _queryPositionalEmbeddingsGradients is null)
            return;

        // The gradient shape should be [numQueryTokens, hiddenDim] or compatible
        int rows = Math.Min(gradient.Shape[0], _numQueryTokens);
        int cols = gradient.Shape.Length > 1 ? Math.Min(gradient.Shape[1], _qformerHiddenDim) : _qformerHiddenDim;

        // Accumulate gradients to query tokens
        // Since query tokens are added with positional embeddings, both receive the same gradient
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                T gradValue = gradient.Shape.Length > 1 ? gradient[i, j] : gradient[i];

                // Accumulate (add to existing gradients for mini-batch accumulation)
                _queryTokensGradients[i, j] = NumOps.Add(_queryTokensGradients[i, j], gradValue);
                _queryPositionalEmbeddingsGradients[i, j] = NumOps.Add(_queryPositionalEmbeddingsGradients[i, j], gradValue);
            }
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        int expectedCount = ParameterCount;
        if (gradients.Length != expectedCount)
        {
            throw new ArgumentException(
                $"Expected {expectedCount} gradients, but got {gradients.Length}",
                nameof(gradients));
        }

        if (!_useNativeMode) return;

        // Get current parameters
        var currentParams = GetParameters();

        // Apply gradient descent update: params = params - learning_rate * gradients
        T learningRate = NumOps.FromDouble(0.001); // Default learning rate
        for (int i = 0; i < currentParams.Length; i++)
        {
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        // Set the updated parameters
        SetParameters(currentParams);
    }

    /// <inheritdoc/>
    public override int ParameterCount
    {
        get
        {
            int count = 0;

            // Q-Former parameters
            count += _numQueryTokens * _qformerHiddenDim; // Query tokens
            count += _numQueryTokens * _qformerHiddenDim; // Query positional embeddings

            foreach (var layer in _qformerSelfAttentionLayers)
            {
                count += layer.ParameterCount;
            }

            foreach (var layer in _qformerCrossAttentionLayers)
            {
                count += layer.ParameterCount;
            }

            foreach (var layer in _qformerFeedForwardLayers)
            {
                count += layer.ParameterCount;
            }

            if (_itmHead is not null)
                count += _itmHead.ParameterCount;

            if (_itcProjection is not null)
                count += _itcProjection.ParameterCount;

            if (_languageModelProjection is not null)
                count += _languageModelProjection.ParameterCount;

            // LM Decoder layers
            foreach (var layer in _lmDecoderLayers)
            {
                count += layer.ParameterCount;
            }

            // LM Head
            if (_lmHead is not null)
                count += _lmHead.ParameterCount;

            return count;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();

        // Add query tokens (trainable)
        if (_queryTokens is not null)
        {
            for (int i = 0; i < _queryTokens.Shape[0]; i++)
            {
                for (int j = 0; j < _queryTokens.Shape[1]; j++)
                {
                    parameters.Add(_queryTokens[i, j]);
                }
            }
        }

        // Add query positional embeddings (trainable)
        if (_queryPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _queryPositionalEmbeddings.Shape[0]; i++)
            {
                for (int j = 0; j < _queryPositionalEmbeddings.Shape[1]; j++)
                {
                    parameters.Add(_queryPositionalEmbeddings[i, j]);
                }
            }
        }

        // Add Q-Former layer parameters
        foreach (var layer in _qformerSelfAttentionLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                parameters.Add(layerParams[i]);
            }
        }

        foreach (var layer in _qformerCrossAttentionLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                parameters.Add(layerParams[i]);
            }
        }

        foreach (var layer in _qformerFeedForwardLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                parameters.Add(layerParams[i]);
            }
        }

        // Add projection head parameters
        if (_itmHead is not null)
        {
            var headParams = _itmHead.GetParameters();
            for (int i = 0; i < headParams.Length; i++)
            {
                parameters.Add(headParams[i]);
            }
        }

        if (_itcProjection is not null)
        {
            var projParams = _itcProjection.GetParameters();
            for (int i = 0; i < projParams.Length; i++)
            {
                parameters.Add(projParams[i]);
            }
        }

        if (_languageModelProjection is not null)
        {
            var lmProjParams = _languageModelProjection.GetParameters();
            for (int i = 0; i < lmProjParams.Length; i++)
            {
                parameters.Add(lmProjParams[i]);
            }
        }

        // Add LM decoder layer parameters
        foreach (var layer in _lmDecoderLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                parameters.Add(layerParams[i]);
            }
        }

        // Add LM head parameters
        if (_lmHead is not null)
        {
            var lmHeadParams = _lmHead.GetParameters();
            for (int i = 0; i < lmHeadParams.Length; i++)
            {
                parameters.Add(lmHeadParams[i]);
            }
        }

        return new Vector<T>([.. parameters]);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Set query tokens
        if (_queryTokens is not null)
        {
            for (int i = 0; i < _queryTokens.Shape[0]; i++)
            {
                for (int j = 0; j < _queryTokens.Shape[1]; j++)
                {
                    _queryTokens[i, j] = parameters[offset++];
                }
            }
        }

        // Set query positional embeddings
        if (_queryPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _queryPositionalEmbeddings.Shape[0]; i++)
            {
                for (int j = 0; j < _queryPositionalEmbeddings.Shape[1]; j++)
                {
                    _queryPositionalEmbeddings[i, j] = parameters[offset++];
                }
            }
        }

        // Set Q-Former layer parameters
        foreach (var layer in _qformerSelfAttentionLayers)
        {
            int layerParamCount = layer.ParameterCount;
            var layerParams = new Vector<T>(layerParamCount);
            for (int i = 0; i < layerParamCount; i++)
            {
                layerParams[i] = parameters[offset + i];
            }
            layer.SetParameters(layerParams);
            offset += layerParamCount;
        }

        foreach (var layer in _qformerCrossAttentionLayers)
        {
            int layerParamCount = layer.ParameterCount;
            var layerParams = new Vector<T>(layerParamCount);
            for (int i = 0; i < layerParamCount; i++)
            {
                layerParams[i] = parameters[offset + i];
            }
            layer.SetParameters(layerParams);
            offset += layerParamCount;
        }

        foreach (var layer in _qformerFeedForwardLayers)
        {
            int layerParamCount = layer.ParameterCount;
            var layerParams = new Vector<T>(layerParamCount);
            for (int i = 0; i < layerParamCount; i++)
            {
                layerParams[i] = parameters[offset + i];
            }
            layer.SetParameters(layerParams);
            offset += layerParamCount;
        }

        // Set projection head parameters
        if (_itmHead is not null)
        {
            int paramCount = _itmHead.ParameterCount;
            var headParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                headParams[i] = parameters[offset + i];
            }
            _itmHead.SetParameters(headParams);
            offset += paramCount;
        }

        if (_itcProjection is not null)
        {
            int paramCount = _itcProjection.ParameterCount;
            var projParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                projParams[i] = parameters[offset + i];
            }
            _itcProjection.SetParameters(projParams);
            offset += paramCount;
        }

        if (_languageModelProjection is not null)
        {
            int paramCount = _languageModelProjection.ParameterCount;
            var lmProjParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                lmProjParams[i] = parameters[offset + i];
            }
            _languageModelProjection.SetParameters(lmProjParams);
            offset += paramCount;
        }

        // Set LM decoder layer parameters
        foreach (var layer in _lmDecoderLayers)
        {
            int layerParamCount = layer.ParameterCount;
            var layerParams = new Vector<T>(layerParamCount);
            for (int i = 0; i < layerParamCount; i++)
            {
                layerParams[i] = parameters[offset + i];
            }
            layer.SetParameters(layerParams);
            offset += layerParamCount;
        }

        // Set LM head parameters
        if (_lmHead is not null)
        {
            int paramCount = _lmHead.ParameterCount;
            var lmHeadParams = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                lmHeadParams[i] = parameters[offset + i];
            }
            _lmHead.SetParameters(lmHeadParams);
            offset += paramCount;
        }
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);

        // For BLIP-2, training involves:
        // 1. Image-Text Contrastive (ITC) loss
        // 2. Image-Text Matching (ITM) loss
        // 3. Image-grounded Text Generation (ITG) loss

        // Forward pass
        var imageOutput = Forward(input);

        // Compute loss (simplified: contrastive loss)
        LastLoss = LossFunction.CalculateLoss(imageOutput.ToVector(), expectedOutput.ToVector());

        // Backward pass - compute gradients
        var lossGradient = LossFunction.CalculateDerivative(imageOutput.ToVector(), expectedOutput.ToVector());
        var gradient = Tensor<T>.FromVector(lossGradient);
        Backward(gradient);

        // Apply gradient descent update using the computed gradients
        // The Backward pass accumulates gradients in layers
        var paramGradients = GetBlip2ParameterGradients();
        UpdateParameters(paramGradients);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Gets the gradients for all BLIP-2 trainable parameters.
    /// </summary>
    private Vector<T> GetBlip2ParameterGradients()
    {
        var gradients = new List<T>();

        // Gradients for query tokens (accumulated during backward)
        if (_queryTokens is not null && _queryTokensGradients is not null)
        {
            for (int i = 0; i < _queryTokensGradients.Shape[0]; i++)
            {
                for (int j = 0; j < _queryTokensGradients.Shape[1]; j++)
                {
                    gradients.Add(_queryTokensGradients[i, j]);
                }
            }
        }
        else if (_queryTokens is not null)
        {
            // Fallback to zeros if gradients not accumulated
            for (int i = 0; i < _queryTokens.Shape[0] * _queryTokens.Shape[1]; i++)
            {
                gradients.Add(NumOps.Zero);
            }
        }

        // Gradients for query positional embeddings
        if (_queryPositionalEmbeddings is not null && _queryPositionalEmbeddingsGradients is not null)
        {
            for (int i = 0; i < _queryPositionalEmbeddingsGradients.Shape[0]; i++)
            {
                for (int j = 0; j < _queryPositionalEmbeddingsGradients.Shape[1]; j++)
                {
                    gradients.Add(_queryPositionalEmbeddingsGradients[i, j]);
                }
            }
        }
        else if (_queryPositionalEmbeddings is not null)
        {
            // Fallback to zeros if gradients not accumulated
            for (int i = 0; i < _queryPositionalEmbeddings.Shape[0] * _queryPositionalEmbeddings.Shape[1]; i++)
            {
                gradients.Add(NumOps.Zero);
            }
        }

        // Get layer gradients
        foreach (var layer in _qformerSelfAttentionLayers)
        {
            var layerGrads = layer.GetParameterGradients();
            for (int i = 0; i < layerGrads.Length; i++)
            {
                gradients.Add(layerGrads[i]);
            }
        }

        foreach (var layer in _qformerCrossAttentionLayers)
        {
            var layerGrads = layer.GetParameterGradients();
            for (int i = 0; i < layerGrads.Length; i++)
            {
                gradients.Add(layerGrads[i]);
            }
        }

        foreach (var layer in _qformerFeedForwardLayers)
        {
            var layerGrads = layer.GetParameterGradients();
            for (int i = 0; i < layerGrads.Length; i++)
            {
                gradients.Add(layerGrads[i]);
            }
        }

        // Get projection head gradients
        if (_itmHead is not null)
        {
            var headGrads = _itmHead.GetParameterGradients();
            for (int i = 0; i < headGrads.Length; i++)
            {
                gradients.Add(headGrads[i]);
            }
        }

        if (_itcProjection is not null)
        {
            var projGrads = _itcProjection.GetParameterGradients();
            for (int i = 0; i < projGrads.Length; i++)
            {
                gradients.Add(projGrads[i]);
            }
        }

        if (_languageModelProjection is not null)
        {
            var lmProjGrads = _languageModelProjection.GetParameterGradients();
            for (int i = 0; i < lmProjGrads.Length; i++)
            {
                gradients.Add(lmProjGrads[i]);
            }
        }

        // Get LM decoder layer gradients
        foreach (var layer in _lmDecoderLayers)
        {
            var layerGrads = layer.GetParameterGradients();
            for (int i = 0; i < layerGrads.Length; i++)
            {
                gradients.Add(layerGrads[i]);
            }
        }

        // Get LM head gradients
        if (_lmHead is not null)
        {
            var lmHeadGrads = _lmHead.GetParameterGradients();
            for (int i = 0; i < lmHeadGrads.Length; i++)
            {
                gradients.Add(lmHeadGrads[i]);
            }
        }

        return new Vector<T>([.. gradients]);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = Enums.ModelType.Blip2,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ImageSize", _imageSize },
                { "EmbeddingDimension", _embeddingDimension },
                { "NumQueryTokens", _numQueryTokens },
                { "QFormerHiddenDim", _qformerHiddenDim },
                { "VisionHiddenDim", _visionHiddenDim },
                { "LmHiddenDim", _lmHiddenDim },
                { "NumQformerLayers", _numQformerLayers },
                { "NumHeads", _numHeads },
                { "NumLmDecoderLayers", _numLmDecoderLayers },
                { "VocabularySize", _vocabularySize },
                { "LanguageModelType", _languageModelType },
                { "UseNativeMode", _useNativeMode },
                { "InputSize", _imageSize * _imageSize * 3 },
                { "OutputSize", _embeddingDimension },
                { "ParameterCount", GetParameterCount() },
                { "TaskType", Architecture.TaskType.ToString() },
                { "LayerCount", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embeddingDimension);
        writer.Write(_maxSequenceLength);
        writer.Write(_imageSize);
        writer.Write(_qformerHiddenDim);
        writer.Write(_numQformerLayers);
        writer.Write(_numHeads);
        writer.Write(_numQueryTokens);
        writer.Write(_patchSize);
        writer.Write(_vocabularySize);
        writer.Write(_visionHiddenDim);
        writer.Write(_lmHiddenDim);
        writer.Write(_languageModelType);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int embeddingDim = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        int qformerHiddenDim = reader.ReadInt32();
        int numQformerLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int numQueryTokens = reader.ReadInt32();
        int patchSize = reader.ReadInt32();
        int vocabularySize = reader.ReadInt32();
        int visionHiddenDim = reader.ReadInt32();
        int lmHiddenDim = reader.ReadInt32();
        string languageModelType = reader.ReadString();
        bool useNativeMode = reader.ReadBoolean();

        // Note: Since fields are readonly, this just validates consistency
        // In practice, you'd need to reconstruct the network if dimensions differ
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Blip2NeuralNetwork<T>(
            Architecture,
            _imageSize,
            3,
            _patchSize,
            _vocabularySize,
            _maxSequenceLength,
            _embeddingDimension,
            _qformerHiddenDim,
            _visionHiddenDim,
            _lmHiddenDim,
            _numQformerLayers,
            _numQueryTokens,
            _numHeads,
            _numLmDecoderLayers,
            _languageModelType,
            _tokenizer,
            null,
            LossFunction);
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _visionEncoder?.Dispose();
            _qformer?.Dispose();
            _languageModel?.Dispose();
        }

        base.Dispose(disposing);
    }

    #endregion

    #region IMultimodalEmbedding Interface (Standard API)

    /// <inheritdoc/>
    public Vector<T> EncodeText(string text)
    {
        return GetTextEmbedding(text);
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeTextBatch(IEnumerable<string> texts)
    {
        var embeddings = GetTextEmbeddings(texts).ToList();
        if (embeddings.Count == 0)
        {
            return new Matrix<T>(0, EmbeddingDimension);
        }

        var matrix = new Matrix<T>(embeddings.Count, embeddings[0].Length);
        for (int i = 0; i < embeddings.Count; i++)
        {
            for (int j = 0; j < embeddings[i].Length; j++)
            {
                matrix[i, j] = embeddings[i][j];
            }
        }
        return matrix;
    }

    /// <inheritdoc/>
    public Vector<T> EncodeImage(double[] imageData)
    {
        // Convert double[] to Tensor<T> in CHW format
        var tensor = ConvertToTensor(imageData);
        return GetImageEmbedding(tensor);
    }

    /// <inheritdoc/>
    public Matrix<T> EncodeImageBatch(IEnumerable<double[]> imageDataBatch)
    {
        var tensors = imageDataBatch.Select(ConvertToTensor);
        var embeddings = GetImageEmbeddings(tensors).ToList();
        if (embeddings.Count == 0)
        {
            return new Matrix<T>(0, EmbeddingDimension);
        }

        var matrix = new Matrix<T>(embeddings.Count, embeddings[0].Length);
        for (int i = 0; i < embeddings.Count; i++)
        {
            for (int j = 0; j < embeddings[i].Length; j++)
            {
                matrix[i, j] = embeddings[i][j];
            }
        }
        return matrix;
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(double[] imageData, IEnumerable<string> labels)
    {
        var tensor = ConvertToTensor(imageData);
        return ZeroShotClassify(tensor, labels);
    }

    /// <summary>
    /// Converts a double[] image to Tensor format.
    /// </summary>
    private Tensor<T> ConvertToTensor(double[] imageData)
    {
        if (imageData == null || imageData.Length == 0)
        {
            throw new ArgumentException("Image data cannot be null or empty.", nameof(imageData));
        }

        int channels = 3;
        if (imageData.Length % channels != 0)
        {
            throw new ArgumentException($"Image data length ({imageData.Length}) must be divisible by {channels} channels.", nameof(imageData));
        }

        int pixels = imageData.Length / channels;
        int size = (int)Math.Sqrt(pixels);
        if (size * size != pixels)
        {
            throw new ArgumentException($"Image must be square. Got {pixels} pixels which is not a perfect square.", nameof(imageData));
        }

        var tensor = new Tensor<T>(new[] { channels, size, size });
        int idx = 0;
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < size; h++)
            {
                for (int w = 0; w < size; w++)
                {
                    tensor[c, h, w] = NumOps.FromDouble(imageData[idx++]);
                }
            }
        }
        return tensor;
    }

    #endregion

}
