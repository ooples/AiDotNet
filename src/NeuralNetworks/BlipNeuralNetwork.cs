using System.IO;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// BLIP (Bootstrapped Language-Image Pre-training) neural network for vision-language tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BLIP extends CLIP's capabilities with image captioning, image-text matching, and visual
/// question answering. It uses a unified framework with both understanding and generation tasks.
/// This implementation supports both ONNX pretrained models and native library layers.
/// </para>
/// <para><b>For Beginners:</b> BLIP is a more powerful version of CLIP!
///
/// CLIP can:
/// - Match images with text descriptions
/// - Zero-shot classification
///
/// BLIP adds:
/// - Generate captions ("a dog playing in the park")
/// - Answer questions ("What color is the car?" -> "Red")
/// - More accurate image-text matching
///
/// Training innovation:
/// - BLIP was trained on noisy web data
/// - It learned to filter out bad captions automatically
/// - Then it generated better captions to train on!
/// - This "bootstrapping" creates a cleaner dataset
///
/// Use cases:
/// - Accessibility (auto-generate alt-text for images)
/// - Content moderation (answer "is there violence in this image?")
/// - Visual search (find images matching a description)
/// - Image organization (auto-tag photos)
/// </para>
/// </remarks>
public class BlipNeuralNetwork<T> : NeuralNetworkBase<T>, IBlipModel<T>
{
    private readonly BlipOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Execution Mode

    /// <summary>
    /// Indicates whether this BLIP network uses native layers (true) or ONNX models (false).
    /// </summary>
    private bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for the vision encoder.
    /// </summary>
    private readonly InferenceSession? _visionEncoder;

    /// <summary>
    /// The ONNX inference session for the text encoder.
    /// </summary>
    private readonly InferenceSession? _textEncoder;

    /// <summary>
    /// The ONNX inference session for the text decoder (for caption generation).
    /// </summary>
    private readonly InferenceSession? _textDecoder;

    /// <summary>
    /// Path to the vision encoder ONNX model file (for ONNX mode).
    /// </summary>
    private readonly string? _visionEncoderPath;

    /// <summary>
    /// Path to the text encoder ONNX model file (for ONNX mode).
    /// </summary>
    private readonly string? _textEncoderPath;

    /// <summary>
    /// Path to the text decoder ONNX model file (for ONNX mode).
    /// </summary>
    private readonly string? _textDecoderPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Vision transformer layers for image encoding (native mode).
    /// </summary>
    private readonly List<ILayer<T>> _visionEncoderLayers = [];

    /// <summary>
    /// Text encoder layers (native mode).
    /// </summary>
    private readonly List<ILayer<T>> _textEncoderLayers = [];

    /// <summary>
    /// Text decoder layers for caption generation (native mode).
    /// </summary>
    private readonly List<ILayer<T>> _textDecoderLayers = [];

    /// <summary>
    /// Cross-attention layers for ITM (native mode).
    /// </summary>
    private readonly List<ILayer<T>> _crossAttentionLayers = [];

    /// <summary>
    /// Image-Text Matching (ITM) classification head.
    /// </summary>
    private ILayer<T>? _itmHead;

    /// <summary>
    /// Learnable CLS token for vision encoder.
    /// </summary>
    private Matrix<T>? _visionClsToken;

    /// <summary>
    /// Learnable CLS token for text encoder.
    /// </summary>
    private Matrix<T>? _textClsToken;

    /// <summary>
    /// Vision positional embeddings.
    /// </summary>
    private Matrix<T>? _visionPositionalEmbeddings;

    /// <summary>
    /// Text positional embeddings.
    /// </summary>
    private Matrix<T>? _textPositionalEmbeddings;

    /// <summary>
    /// Text token embeddings (vocabulary lookup).
    /// </summary>
    private ILayer<T>? _textTokenEmbedding;

    /// <summary>
    /// Patch embedding layer for vision.
    /// </summary>
    private ILayer<T>? _patchEmbedding;

    /// <summary>
    /// Language model head - projects hidden states to vocabulary logits.
    /// </summary>
    private ILayer<T>? _lmHead;

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
    private int _embeddingDimension;

    /// <summary>
    /// Maximum sequence length for text encoder.
    /// </summary>
    private int _maxSequenceLength;

    /// <summary>
    /// Expected image size (width and height).
    /// </summary>
    private int _imageSize;

    /// <summary>
    /// Hidden dimension for transformer layers.
    /// </summary>
    private int _hiddenDim;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    private int _numLayers;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private int _numHeads;

    /// <summary>
    /// MLP hidden dimension.
    /// </summary>
    private int _mlpDim;

    /// <summary>
    /// Patch size for vision transformer.
    /// </summary>
    private int _patchSize;

    /// <summary>
    /// Vocabulary size for text encoder.
    /// </summary>
    private int _vocabularySize;

    /// <summary>
    /// Number of decoder layers.
    /// </summary>
    private int _numDecoderLayers;

    #endregion

    #region IMultimodalEmbedding Properties

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc/>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    public int ImageSize => _imageSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a BLIP network using pretrained ONNX models.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="visionEncoderPath">Path to the vision encoder ONNX model.</param>
    /// <param name="textEncoderPath">Path to the text encoder ONNX model.</param>
    /// <param name="textDecoderPath">Path to the text decoder ONNX model.</param>
    /// <param name="tokenizer">The tokenizer for text processing.</param>
    /// <param name="embeddingDimension">Dimension of the shared embedding space.</param>
    /// <param name="maxSequenceLength">Maximum text sequence length.</param>
    /// <param name="imageSize">Expected image size.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    public BlipNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string visionEncoderPath,
        string textEncoderPath,
        string textDecoderPath,
        ITokenizer tokenizer,
        int embeddingDimension = 256,
        int maxSequenceLength = 35,
        int imageSize = 384,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        BlipOptions? options = null)
        : base(architecture,
               lossFunction ?? new ContrastiveLoss<T>(),
               1.0)
    {
        _options = options ?? new BlipOptions();
        Options = _options;

        // Validate ONNX model paths
        if (string.IsNullOrWhiteSpace(visionEncoderPath))
            throw new ArgumentException("Vision encoder path cannot be null or empty.", nameof(visionEncoderPath));
        if (string.IsNullOrWhiteSpace(textEncoderPath))
            throw new ArgumentException("Text encoder path cannot be null or empty.", nameof(textEncoderPath));
        if (string.IsNullOrWhiteSpace(textDecoderPath))
            throw new ArgumentException("Text decoder path cannot be null or empty.", nameof(textDecoderPath));
        if (!File.Exists(visionEncoderPath))
            throw new FileNotFoundException($"Vision encoder model not found: {visionEncoderPath}");
        if (!File.Exists(textEncoderPath))
            throw new FileNotFoundException($"Text encoder model not found: {textEncoderPath}");
        if (!File.Exists(textDecoderPath))
            throw new FileNotFoundException($"Text decoder model not found: {textDecoderPath}");

        _useNativeMode = false;
        _visionEncoderPath = visionEncoderPath;
        _textEncoderPath = textEncoderPath;
        _textDecoderPath = textDecoderPath;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _hiddenDim = 768;
        _numLayers = 12;
        _numHeads = 12;
        _mlpDim = 3072;
        _patchSize = 16;
        _vocabularySize = 30522; // BERT vocabulary size
        _numDecoderLayers = 12;

        InferenceSession? visionEncoder = null;
        InferenceSession? textEncoder = null;
        InferenceSession? textDecoder = null;

        try
        {
            visionEncoder = new InferenceSession(visionEncoderPath);
            textEncoder = new InferenceSession(textEncoderPath);
            textDecoder = new InferenceSession(textDecoderPath);

            _visionEncoder = visionEncoder;
            _textEncoder = textEncoder;
            _textDecoder = textDecoder;

            Guard.NotNull(tokenizer);
            _tokenizer = tokenizer;

            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _lossFunction = lossFunction ?? new ContrastiveLoss<T>();

            InitializeLayers();
        }
        catch
        {
            try
            {
                visionEncoder?.Dispose();
                textEncoder?.Dispose();
                textDecoder?.Dispose();
            }
            catch
            {
                // Swallow disposal exceptions
            }

            throw;
        }
    }

    /// <summary>
    /// Creates a BLIP network using native library layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="imageSize">Expected image size (default 384 for BLIP).</param>
    /// <param name="channels">Number of image channels (default 3 for RGB).</param>
    /// <param name="patchSize">Patch size for vision transformer.</param>
    /// <param name="vocabularySize">Text vocabulary size (BERT: 30522).</param>
    /// <param name="maxSequenceLength">Maximum text sequence length.</param>
    /// <param name="embeddingDimension">Dimension of shared embedding space.</param>
    /// <param name="hiddenDim">Hidden dimension for transformers.</param>
    /// <param name="numEncoderLayers">Number of encoder transformer layers.</param>
    /// <param name="numDecoderLayers">Number of decoder transformer layers.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="mlpDim">MLP hidden dimension.</param>
    /// <param name="tokenizer">Optional tokenizer for text processing.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a fully trainable BLIP network using the library's
    /// native layers. All operations use the Engine for CPU/GPU acceleration.
    /// </para>
    /// </remarks>
    public BlipNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int imageSize = 384,
        int channels = 3,
        int patchSize = 16,
        int vocabularySize = 30522,
        int maxSequenceLength = 35,
        int embeddingDimension = 256,
        int hiddenDim = 768,
        int numEncoderLayers = 12,
        int numDecoderLayers = 12,
        int numHeads = 12,
        int mlpDim = 3072,
        ITokenizer? tokenizer = null,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        BlipOptions? options = null)
        : base(architecture,
               lossFunction ?? new ContrastiveLoss<T>(),
               1.0)
    {
        _options = options ?? new BlipOptions();
        Options = _options;

        _useNativeMode = true;
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _hiddenDim = hiddenDim;
        _numLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numHeads = numHeads;
        _mlpDim = mlpDim;
        _patchSize = patchSize;
        _vocabularySize = vocabularySize;

        // Create simple tokenizer if not provided
        _tokenizer = tokenizer ?? CreateDefaultTokenizer();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new ContrastiveLoss<T>();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (_useNativeMode)
        {
            InitializeNativeLayers();
        }
        else
        {
            // ONNX mode - layers are in the ONNX models
        }
    }

    /// <summary>
    /// Initializes native layers for the BLIP network.
    /// </summary>
    private void InitializeNativeLayers()
    {
        int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);

        Layers.Clear();

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateBlipLayers(
                _imageSize, _patchSize, _hiddenDim, _embeddingDimension,
                _numLayers, _numDecoderLayers, _numHeads, _mlpDim, _vocabularySize));
        }

        // Distribute layers to internal fields
        int idx = 0;

        // Vision encoder: PatchEmbed + numLayers × TransformerEncoder + projection
        _patchEmbedding = Layers[idx++];
        _visionEncoderLayers.Clear();
        _visionEncoderLayers.Add(_patchEmbedding);
        for (int i = 0; i < _numLayers; i++)
            _visionEncoderLayers.Add(Layers[idx++]);
        _visionEncoderLayers.Add(Layers[idx++]); // vision projection

        // Text encoder: EmbeddingLayer + numLayers × TransformerEncoder + projection
        _textTokenEmbedding = Layers[idx++];
        _textEncoderLayers.Clear();
        _textEncoderLayers.Add(_textTokenEmbedding);
        for (int i = 0; i < _numLayers; i++)
            _textEncoderLayers.Add(Layers[idx++]);
        _textEncoderLayers.Add(Layers[idx++]); // text projection

        // Text decoder layers
        _textDecoderLayers.Clear();
        for (int i = 0; i < _numDecoderLayers; i++)
            _textDecoderLayers.Add(Layers[idx++]);

        // Cross-attention layers (6)
        _crossAttentionLayers.Clear();
        for (int i = 0; i < 6; i++)
            _crossAttentionLayers.Add(Layers[idx++]);

        // ITM head + LM head
        _itmHead = Layers[idx++];
        _lmHead = Layers[idx++];

        // Initialize learnable tokens
        _visionClsToken = Matrix<T>.CreateDefault(1, _hiddenDim, NumOps.Zero);
        _textClsToken = Matrix<T>.CreateDefault(1, _hiddenDim, NumOps.Zero);

        // Initialize positional embeddings
        _visionPositionalEmbeddings = Matrix<T>.CreateDefault(numPatches + 1, _hiddenDim, NumOps.Zero);
        _textPositionalEmbeddings = Matrix<T>.CreateDefault(_maxSequenceLength, _hiddenDim, NumOps.Zero);

        // Initialize with small random values
        InitializeParameters();
    }

    /// <summary>
    /// Initialize parameters with small random values.
    /// </summary>
    private void InitializeParameters()
    {
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        double scale = 0.02;

        // Initialize CLS tokens
        if (_visionClsToken is not null)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                _visionClsToken[0, j] = NumOps.FromDouble((random.NextDouble() - 0.5) * scale);
            }
        }

        if (_textClsToken is not null)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                _textClsToken[0, j] = NumOps.FromDouble((random.NextDouble() - 0.5) * scale);
            }
        }

        // Initialize positional embeddings
        if (_visionPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _visionPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < _visionPositionalEmbeddings.Columns; j++)
                {
                    _visionPositionalEmbeddings[i, j] = NumOps.FromDouble((random.NextDouble() - 0.5) * scale);
                }
            }
        }

        if (_textPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _textPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < _textPositionalEmbeddings.Columns; j++)
                {
                    _textPositionalEmbeddings[i, j] = NumOps.FromDouble((random.NextDouble() - 0.5) * scale);
                }
            }
        }
    }

    /// <summary>
    /// Creates a default simple tokenizer for testing.
    /// </summary>
    private static ITokenizer CreateDefaultTokenizer()
    {
        return Tokenization.ClipTokenizerFactory.CreateSimple();
    }

    #endregion

    #region IMultimodalEmbedding Implementation

    /// <inheritdoc/>
    public Vector<T> GetTextEmbedding(string text)
    {
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
        var textList = texts.ToList();
        var embeddings = new List<Vector<T>>(textList.Count);

        foreach (var text in textList)
        {
            embeddings.Add(GetTextEmbedding(text));
        }

        return embeddings;
    }

    /// <inheritdoc/>
    public Vector<T> GetImageEmbedding(Tensor<T> image)
    {
        if (_useNativeMode)
        {
            return GetImageEmbeddingNative(image);
        }
        else
        {
            return GetImageEmbeddingOnnx(image);
        }
    }

    /// <inheritdoc/>
    public IEnumerable<Vector<T>> GetImageEmbeddings(IEnumerable<Tensor<T>> images)
    {
        var imageList = images.ToList();
        var embeddings = new List<Vector<T>>(imageList.Count);

        foreach (var image in imageList)
        {
            embeddings.Add(GetImageEmbedding(image));
        }

        return embeddings;
    }

    /// <inheritdoc/>
    public T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding)
    {
        // Both embeddings should already be L2-normalized
        // Cosine similarity = dot product for unit vectors
        T dotProduct = NumOps.Zero;

        for (int i = 0; i < textEmbedding.Length && i < imageEmbedding.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(textEmbedding[i], imageEmbedding[i]));
        }

        return dotProduct;
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, IEnumerable<string> classLabels)
    {
        if (classLabels is null)
        {
            throw new ArgumentNullException(nameof(classLabels));
        }

        var labelList = classLabels.ToList();
        if (labelList.Count == 0)
        {
            throw new ArgumentException("At least one class label must be provided.", nameof(classLabels));
        }

        var imageEmbedding = GetImageEmbedding(image);
        var scores = new Dictionary<string, T>();

        // Get embeddings for all labels
        var labelEmbeddings = new List<Vector<T>>();
        foreach (var label in labelList)
        {
            string prompt = $"a photo of {label}";
            labelEmbeddings.Add(GetTextEmbedding(prompt));
        }

        // Compute similarities
        var similarities = new T[labelList.Count];
        for (int i = 0; i < labelList.Count; i++)
        {
            similarities[i] = ComputeSimilarity(labelEmbeddings[i], imageEmbedding);
        }

        // Apply softmax
        T maxSim = similarities[0];
        for (int i = 1; i < similarities.Length; i++)
        {
            if (NumOps.ToDouble(similarities[i]) > NumOps.ToDouble(maxSim))
            {
                maxSim = similarities[i];
            }
        }

        T sumExp = NumOps.Zero;
        var expSims = new T[similarities.Length];
        for (int i = 0; i < similarities.Length; i++)
        {
            T shifted = NumOps.Subtract(similarities[i], maxSim);
            expSims[i] = NumOps.Exp(NumOps.Multiply(shifted, NumOps.FromDouble(100.0))); // Temperature
            sumExp = NumOps.Add(sumExp, expSims[i]);
        }

        for (int i = 0; i < labelList.Count; i++)
        {
            scores[labelList[i]] = NumOps.Divide(expSims[i], sumExp);
        }

        return scores;
    }

    #endregion

    #region IBlipModel Implementation

    /// <inheritdoc/>
    public string GenerateCaption(Tensor<T> image, int maxLength = 30, int numBeams = 3)
    {
        if (_useNativeMode)
        {
            return GenerateCaptionNative(image, maxLength, numBeams);
        }
        else
        {
            return GenerateCaptionOnnx(image, maxLength, numBeams);
        }
    }

    /// <inheritdoc/>
    public IEnumerable<string> GenerateCaptions(Tensor<T> image, int numCaptions = 5, int maxLength = 30)
    {
        var captions = new List<string>();

        for (int i = 0; i < numCaptions; i++)
        {
            // Use different random seeds for diversity
            string caption = GenerateCaptionWithSampling(image, maxLength, temperature: 0.7);
            if (!string.IsNullOrWhiteSpace(caption))
            {
                captions.Add(caption);
            }
        }

        return captions.Distinct().ToList();
    }

    /// <inheritdoc/>
    public T ComputeImageTextMatch(Tensor<T> image, string text)
    {
        if (_useNativeMode)
        {
            return ComputeImageTextMatchNative(image, text);
        }
        else
        {
            return ComputeImageTextMatchOnnx(image, text);
        }
    }

    /// <inheritdoc/>
    public string AnswerQuestion(Tensor<T> image, string question, int maxLength = 20)
    {
        if (_useNativeMode)
        {
            return AnswerQuestionNative(image, question, maxLength);
        }
        else
        {
            return AnswerQuestionOnnx(image, question, maxLength);
        }
    }

    /// <inheritdoc/>
    public IEnumerable<(string Caption, T Score)> RankCaptions(Tensor<T> image, IEnumerable<string> candidates)
    {
        var candidateList = candidates.ToList();
        var scored = new List<(string Caption, T Score)>();

        foreach (var caption in candidateList)
        {
            T score = ComputeImageTextMatch(image, caption);
            scored.Add((caption, score));
        }

        return scored.OrderByDescending(x => NumOps.ToDouble(x.Score)).ToList();
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveImages(
        string query,
        IEnumerable<Vector<T>> imageEmbeddings,
        int topK = 10)
    {
        var queryEmbedding = GetTextEmbedding(query);
        var embeddings = imageEmbeddings.ToList();
        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < embeddings.Count; i++)
        {
            T score = ComputeSimilarity(queryEmbedding, embeddings[i]);
            scores.Add((i, score));
        }

        return scores.OrderByDescending(x => NumOps.ToDouble(x.Score)).Take(topK).ToList();
    }

    /// <inheritdoc/>
    public IEnumerable<(int Index, T Score)> RetrieveTexts(
        Tensor<T> image,
        IEnumerable<Vector<T>> textEmbeddings,
        int topK = 10)
    {
        var imageEmbedding = GetImageEmbedding(image);
        var embeddings = textEmbeddings.ToList();
        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < embeddings.Count; i++)
        {
            T score = ComputeSimilarity(embeddings[i], imageEmbedding);
            scores.Add((i, score));
        }

        return scores.OrderByDescending(x => NumOps.ToDouble(x.Score)).Take(topK).ToList();
    }

    #endregion

    #region Native Mode Implementations

    /// <summary>
    /// Gets text embedding using native layers.
    /// </summary>
    private Vector<T> GetTextEmbeddingNative(string text)
    {
        // Tokenize the text
        var options = new EncodingOptions
        {
            MaxLength = _maxSequenceLength,
            Padding = true,
            PaddingSide = "right",
            Truncation = true,
            TruncationSide = "right",
            AddSpecialTokens = true,
            ReturnAttentionMask = true
        };

        var encoded = _tokenizer.Encode(text, options);
        var tokenIds = encoded.TokenIds;

        // Create input tensor [1, seqLen]
        var inputTensor = new Tensor<T>(new[] { 1, tokenIds.Count });
        for (int i = 0; i < tokenIds.Count; i++)
        {
            inputTensor[0, i] = NumOps.FromDouble(tokenIds[i]);
        }

        // Get token embeddings
        Tensor<T> hidden = inputTensor;
        if (_textTokenEmbedding is not null)
        {
            hidden = _textTokenEmbedding.Forward(inputTensor);
        }

        // Add positional embeddings using Engine
        if (_textPositionalEmbeddings is not null)
        {
            var posEmbTensor = Tensor<T>.FromMatrix(_textPositionalEmbeddings);
            int seqLen = Math.Min(hidden.Shape[1], posEmbTensor.Shape[0]);

            // Slice to actual sequence length using proper int[] parameters
            var posSlice = Engine.TensorSlice(posEmbTensor, new[] { 0, 0 }, new[] { seqLen, _hiddenDim });
            var posExpanded = Engine.TensorExpandDims<T>(posSlice, 0);
            hidden = Engine.TensorBroadcastAdd<T>(hidden, posExpanded);
        }

        // Process through transformer layers
        foreach (var layer in _textEncoderLayers.Skip(1)) // Skip embedding layer
        {
            hidden = layer.Forward(hidden);
        }

        // Extract CLS token embedding (first position)
        var clsEmbedding = new T[_embeddingDimension];
        int embDim = Math.Min(_embeddingDimension, hidden.Shape[^1]);

        for (int i = 0; i < embDim; i++)
        {
            clsEmbedding[i] = hidden.Shape.Length == 3 ? hidden[0, 0, i] : hidden[0, i];
        }

        // L2 normalize
        return NormalizeVector(new Vector<T>(clsEmbedding));
    }

    /// <summary>
    /// Gets image embedding using native layers.
    /// </summary>
    private Vector<T> GetImageEmbeddingNative(Tensor<T> image)
    {
        // Ensure batch dimension
        Tensor<T> batchedImage = image.Shape.Length == 3
            ? Engine.TensorExpandDims(image, 0)
            : image;

        // Patch embedding
        Tensor<T> hidden = batchedImage;
        if (_patchEmbedding is not null)
        {
            hidden = _patchEmbedding.Forward(batchedImage);
        }

        // Prepend CLS token using Engine operations
        if (_visionClsToken is not null)
        {
            var clsTensor = Tensor<T>.FromMatrix(_visionClsToken);
            var clsExpanded = Engine.TensorExpandDims<T>(clsTensor, 0);

            // Concatenate CLS token with patch embeddings
            hidden = Engine.TensorConcatenate(new[] { clsExpanded, hidden }, axis: 1);
        }

        // Add positional embeddings
        if (_visionPositionalEmbeddings is not null)
        {
            var posEmbTensor = Tensor<T>.FromMatrix(_visionPositionalEmbeddings);
            int seqLen = Math.Min(hidden.Shape[1], posEmbTensor.Shape[0]);

            var posSlice = Engine.TensorSlice(posEmbTensor, new[] { 0, 0 }, new[] { seqLen, _hiddenDim });
            var posExpanded = Engine.TensorExpandDims<T>(posSlice, 0);
            hidden = Engine.TensorBroadcastAdd<T>(hidden, posExpanded);
        }

        // Process through transformer layers
        foreach (var layer in _visionEncoderLayers.Skip(1)) // Skip patch embedding
        {
            hidden = layer.Forward(hidden);
        }

        // Extract CLS token embedding
        var clsEmbedding = new T[_embeddingDimension];
        int embDim = Math.Min(_embeddingDimension, hidden.Shape[^1]);

        for (int i = 0; i < embDim; i++)
        {
            clsEmbedding[i] = hidden.Shape.Length == 3 ? hidden[0, 0, i] : hidden[0, i];
        }

        return NormalizeVector(new Vector<T>(clsEmbedding));
    }

    /// <summary>
    /// Generates a caption using native layers.
    /// </summary>
    private string GenerateCaptionNative(Tensor<T> image, int maxLength, int numBeams)
    {
        // Get image features
        Tensor<T> imageFeatures = GetImageFeaturesNative(image);

        // Start with BOS token
        var generatedTokens = new List<int> { GetBosTokenId() };

        // Simple greedy decoding (beam search would be more complex)
        for (int step = 0; step < maxLength; step++)
        {
            // Create decoder input
            var inputTensor = new Tensor<T>(new[] { 1, generatedTokens.Count });
            for (int i = 0; i < generatedTokens.Count; i++)
            {
                inputTensor[0, i] = NumOps.FromDouble(generatedTokens[i]);
            }

            // Get next token logits
            var logits = ForwardDecoderNative(inputTensor, imageFeatures);

            // Get most likely next token (greedy)
            int nextToken = ArgMax(logits);

            if (nextToken == GetEosTokenId())
            {
                break;
            }

            generatedTokens.Add(nextToken);
        }

        // Decode tokens to text
        return DecodeTokens(generatedTokens);
    }

    /// <summary>
    /// Generates caption with temperature sampling for diversity.
    /// </summary>
    private string GenerateCaptionWithSampling(Tensor<T> image, int maxLength, double temperature)
    {
        Tensor<T> imageFeatures = GetImageFeaturesNative(image);
        var generatedTokens = new List<int> { GetBosTokenId() };
        // Use thread-safe random with proper seed (not DateTime.Millisecond which only has 1000 values)
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(
            Tensors.Helpers.RandomHelper.ThreadSafeRandom.Next());

        for (int step = 0; step < maxLength; step++)
        {
            var inputTensor = new Tensor<T>(new[] { 1, generatedTokens.Count });
            for (int i = 0; i < generatedTokens.Count; i++)
            {
                inputTensor[0, i] = NumOps.FromDouble(generatedTokens[i]);
            }

            var logits = ForwardDecoderNative(inputTensor, imageFeatures);

            // Apply temperature and sample
            int nextToken = SampleWithTemperature(logits, temperature, random);

            if (nextToken == GetEosTokenId())
            {
                break;
            }

            generatedTokens.Add(nextToken);
        }

        return DecodeTokens(generatedTokens);
    }

    /// <summary>
    /// Computes ITM score using native layers.
    /// </summary>
    private T ComputeImageTextMatchNative(Tensor<T> image, string text)
    {
        // Get image and text features
        var imageFeatures = GetImageFeaturesNative(image);
        var textFeatures = GetTextFeaturesNative(text);

        // Apply cross-attention
        Tensor<T> fused = textFeatures;
        foreach (var layer in _crossAttentionLayers)
        {
            // Simple implementation: just use text features
            // Full implementation would use cross-attention between image and text
            fused = layer.Forward(fused);
        }

        // Apply ITM head
        if (_itmHead is not null)
        {
            var logits = _itmHead.Forward(fused);

            // Get probability of match (class 1)
            T logit0 = logits.Shape.Length == 3 ? logits[0, 0, 0] : logits[0, 0];
            T logit1 = logits.Shape.Length == 3 ? logits[0, 0, 1] : logits[0, 1];

            // Softmax
            T maxLogit = NumOps.ToDouble(logit0) > NumOps.ToDouble(logit1) ? logit0 : logit1;
            T exp0 = NumOps.Exp(NumOps.Subtract(logit0, maxLogit));
            T exp1 = NumOps.Exp(NumOps.Subtract(logit1, maxLogit));
            T sum = NumOps.Add(exp0, exp1);

            return NumOps.Divide(exp1, sum); // Return probability of match
        }

        return NumOps.FromDouble(0.5);
    }

    /// <summary>
    /// Answers a question using native layers.
    /// </summary>
    private string AnswerQuestionNative(Tensor<T> image, string question, int maxLength)
    {
        var imageFeatures = GetImageFeaturesNative(image);

        // Encode question
        var options = new EncodingOptions
        {
            MaxLength = _maxSequenceLength,
            Padding = true,
            Truncation = true,
            AddSpecialTokens = true
        };

        var encoded = _tokenizer.Encode(question, options);
        var generatedTokens = new List<int>(encoded.TokenIds);

        // Generate answer tokens
        for (int step = 0; step < maxLength; step++)
        {
            var inputTensor = new Tensor<T>(new[] { 1, generatedTokens.Count });
            for (int i = 0; i < generatedTokens.Count; i++)
            {
                inputTensor[0, i] = NumOps.FromDouble(generatedTokens[i]);
            }

            var logits = ForwardDecoderNative(inputTensor, imageFeatures);
            int nextToken = ArgMax(logits);

            if (nextToken == GetEosTokenId())
            {
                break;
            }

            generatedTokens.Add(nextToken);
        }

        // Decode only the answer portion
        var answerTokens = generatedTokens.Skip(encoded.TokenIds.Count).ToList();
        return DecodeTokens(answerTokens);
    }

    /// <summary>
    /// Gets image features without final projection (for cross-attention).
    /// </summary>
    private Tensor<T> GetImageFeaturesNative(Tensor<T> image)
    {
        Tensor<T> batchedImage = image.Shape.Length == 3
            ? Engine.TensorExpandDims(image, 0)
            : image;

        Tensor<T> hidden = batchedImage;
        if (_patchEmbedding is not null)
        {
            hidden = _patchEmbedding.Forward(batchedImage);
        }

        if (_visionClsToken is not null)
        {
            var clsTensor = Tensor<T>.FromMatrix(_visionClsToken);
            var clsExpanded = Engine.TensorExpandDims<T>(clsTensor, 0);
            hidden = Engine.TensorConcatenate(new[] { clsExpanded, hidden }, axis: 1);
        }

        if (_visionPositionalEmbeddings is not null)
        {
            var posEmbTensor = Tensor<T>.FromMatrix(_visionPositionalEmbeddings);
            int seqLen = Math.Min(hidden.Shape[1], posEmbTensor.Shape[0]);

            var posSlice = Engine.TensorSlice(posEmbTensor, new[] { 0, 0 }, new[] { seqLen, _hiddenDim });
            var posExpanded = Engine.TensorExpandDims<T>(posSlice, 0);
            hidden = Engine.TensorBroadcastAdd<T>(hidden, posExpanded);
        }

        // Process through transformer layers (except final projection)
        // Skip first layer (embedding) and last layer (projection), handling edge cases
        int layerCount = _visionEncoderLayers.Count;
        if (layerCount > 2)
        {
            foreach (var layer in _visionEncoderLayers.Skip(1).Take(layerCount - 2))
            {
                hidden = layer.Forward(hidden);
            }
        }

        return hidden;
    }

    /// <summary>
    /// Gets text features without final projection.
    /// </summary>
    private Tensor<T> GetTextFeaturesNative(string text)
    {
        var options = new EncodingOptions
        {
            MaxLength = _maxSequenceLength,
            Padding = true,
            Truncation = true,
            AddSpecialTokens = true
        };

        var encoded = _tokenizer.Encode(text, options);
        var tokenIds = encoded.TokenIds;

        var inputTensor = new Tensor<T>(new[] { 1, tokenIds.Count });
        for (int i = 0; i < tokenIds.Count; i++)
        {
            inputTensor[0, i] = NumOps.FromDouble(tokenIds[i]);
        }

        Tensor<T> hidden = inputTensor;
        if (_textTokenEmbedding is not null)
        {
            hidden = _textTokenEmbedding.Forward(inputTensor);
        }

        if (_textPositionalEmbeddings is not null)
        {
            var posEmbTensor = Tensor<T>.FromMatrix(_textPositionalEmbeddings);
            int seqLen = Math.Min(hidden.Shape[1], posEmbTensor.Shape[0]);

            var posSlice = Engine.TensorSlice(posEmbTensor, new[] { 0, 0 }, new[] { seqLen, _hiddenDim });
            var posExpanded = Engine.TensorExpandDims<T>(posSlice, 0);
            hidden = Engine.TensorBroadcastAdd<T>(hidden, posExpanded);
        }

        // Process through transformer layers (except final projection)
        // Skip first layer (embedding) and last layer (projection), handling edge cases
        int textLayerCount = _textEncoderLayers.Count;
        if (textLayerCount > 2)
        {
            foreach (var layer in _textEncoderLayers.Skip(1).Take(textLayerCount - 2))
            {
                hidden = layer.Forward(hidden);
            }
        }

        return hidden;
    }

    /// <summary>
    /// Forward pass through decoder with image conditioning.
    /// </summary>
    private Tensor<T> ForwardDecoderNative(Tensor<T> input, Tensor<T> imageFeatures)
    {
        // Get embeddings
        Tensor<T> hidden = input;
        if (_textTokenEmbedding is not null)
        {
            hidden = _textTokenEmbedding.Forward(input);
        }

        // Add positional embeddings
        if (_textPositionalEmbeddings is not null)
        {
            var posEmbTensor = Tensor<T>.FromMatrix(_textPositionalEmbeddings);
            int seqLen = Math.Min(hidden.Shape[1], posEmbTensor.Shape[0]);

            var posSlice = Engine.TensorSlice(posEmbTensor, new[] { 0, 0 }, new[] { seqLen, _hiddenDim });
            var posExpanded = Engine.TensorExpandDims<T>(posSlice, 0);
            hidden = Engine.TensorBroadcastAdd<T>(hidden, posExpanded);
        }

        // Process through decoder layers
        foreach (var layer in _textDecoderLayers)
        {
            hidden = layer.Forward(hidden);
        }

        // Project hidden states to vocabulary logits using the language model head
        if (_lmHead is not null)
        {
            // Reshape for dense layer: [batch, seq_len, hidden_dim] -> [batch * seq_len, hidden_dim]
            int batchSize = hidden.Shape[0];
            int seqLen = hidden.Shape[1];
            int hiddenDim = hidden.Shape[2];

            var flatHidden = Engine.Reshape(hidden, [batchSize * seqLen, hiddenDim]);
            var logits = _lmHead.Forward(flatHidden);

            // Reshape back: [batch * seq_len, vocab_size] -> [batch, seq_len, vocab_size]
            int vocabSize = logits.Shape[1];
            return Engine.Reshape(logits, [batchSize, seqLen, vocabSize]);
        }

        // Fallback: return hidden states (won't produce valid tokens)
        return hidden;
    }

    #endregion

    #region ONNX Mode Implementations

    /// <summary>
    /// Gets text embedding using ONNX.
    /// </summary>
    private Vector<T> GetTextEmbeddingOnnx(string text)
    {
        if (_textEncoder is null)
        {
            throw new InvalidOperationException("Text encoder not initialized.");
        }

        var options = new EncodingOptions
        {
            MaxLength = _maxSequenceLength,
            Padding = true,
            Truncation = true,
            AddSpecialTokens = true
        };

        var encoded = _tokenizer.Encode(text, options);
        var inputIds = encoded.TokenIds;
        var attentionMask = encoded.AttentionMask is not null
            ? encoded.AttentionMask.ToArray()
            : Enumerable.Repeat(1, inputIds.Count).ToArray();

        var inputTensor = new OnnxTensors.DenseTensor<long>(new[] { 1, inputIds.Count });
        var maskTensor = new OnnxTensors.DenseTensor<long>(new[] { 1, attentionMask.Length });

        for (int i = 0; i < inputIds.Count; i++)
        {
            inputTensor[0, i] = inputIds[i];
            maskTensor[0, i] = attentionMask[i];
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor)
        };

        using var results = _textEncoder.Run(inputs);
        var output = results.First().AsTensor<float>();

        return ExtractEmbeddingFromOnnxTensor(output);
    }

    /// <summary>
    /// Gets image embedding using ONNX.
    /// </summary>
    private Vector<T> GetImageEmbeddingOnnx(Tensor<T> image)
    {
        if (_visionEncoder is null)
        {
            throw new InvalidOperationException("Vision encoder not initialized.");
        }

        int channels = image.Shape.Length == 4 ? image.Shape[1] : image.Shape[0];
        int height = image.Shape.Length == 4 ? image.Shape[2] : image.Shape[1];
        int width = image.Shape.Length == 4 ? image.Shape[3] : image.Shape[2];

        var inputTensor = new OnnxTensors.DenseTensor<float>(new[] { 1, channels, height, width });

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    T value = image.Shape.Length == 4 ? image[0, c, h, w] : image[c, h, w];
                    inputTensor[0, c, h, w] = (float)NumOps.ToDouble(value);
                }
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", inputTensor)
        };

        using var results = _visionEncoder.Run(inputs);
        var output = results.First().AsTensor<float>();

        return ExtractEmbeddingFromOnnxTensor(output);
    }

    /// <summary>
    /// Generates caption using ONNX decoder.
    /// </summary>
    private string GenerateCaptionOnnx(Tensor<T> image, int maxLength, int numBeams)
    {
        // Simplified implementation - actual ONNX captioning would need the decoder model
        return "[Caption generation requires ONNX decoder model]";
    }

    /// <summary>
    /// Computes ITM score using ONNX.
    /// </summary>
    private T ComputeImageTextMatchOnnx(Tensor<T> image, string text)
    {
        // Use embedding similarity as fallback
        var imageEmb = GetImageEmbeddingOnnx(image);
        var textEmb = GetTextEmbeddingOnnx(text);
        return ComputeSimilarity(textEmb, imageEmb);
    }

    /// <summary>
    /// Answers question using ONNX.
    /// </summary>
    private string AnswerQuestionOnnx(Tensor<T> image, string question, int maxLength)
    {
        // Simplified implementation
        return "[VQA requires ONNX decoder model]";
    }

    /// <summary>
    /// Extracts embedding from ONNX tensor.
    /// </summary>
    private Vector<T> ExtractEmbeddingFromOnnxTensor(OnnxTensors.Tensor<float> output)
    {
        int rank = output.Dimensions.Length;
        int hiddenSize = rank > 0 ? output.Dimensions[rank - 1] : (int)output.Length;
        int vectorSize = Math.Min(_embeddingDimension, hiddenSize);

        var data = new T[vectorSize];

        for (int i = 0; i < vectorSize; i++)
        {
            float value = rank switch
            {
                1 => output.GetValue(i),
                2 => output[0, i],
                3 => output[0, 0, i],
                _ => output.GetValue(i)
            };

            data[i] = NumOps.FromDouble(value);
        }

        return NormalizeVector(new Vector<T>(data));
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// L2 normalizes a vector.
    /// </summary>
    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        T sumSquared = NumOps.Zero;

        for (int i = 0; i < vector.Length; i++)
        {
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(vector[i], vector[i]));
        }

        T norm = NumOps.Sqrt(sumSquared);
        T epsilon = NumOps.FromDouble(1e-12);

        if (NumOps.ToDouble(norm) < NumOps.ToDouble(epsilon))
        {
            return vector;
        }

        var normalized = new T[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = NumOps.Divide(vector[i], norm);
        }

        return new Vector<T>(normalized);
    }

    /// <summary>
    /// Gets the BOS token ID.
    /// </summary>
    private int GetBosTokenId()
    {
        var specialTokens = _tokenizer.SpecialTokens;
        if (specialTokens is not null && !string.IsNullOrEmpty(specialTokens.BosToken))
        {
            return _tokenizer.Vocabulary.GetTokenId(specialTokens.BosToken);
        }
        // Fallback: use CLS token if BOS is not defined
        if (specialTokens is not null && !string.IsNullOrEmpty(specialTokens.ClsToken))
        {
            return _tokenizer.Vocabulary.GetTokenId(specialTokens.ClsToken);
        }
        return 101; // Default BERT [CLS]
    }

    /// <summary>
    /// Gets the EOS token ID.
    /// </summary>
    private int GetEosTokenId()
    {
        var specialTokens = _tokenizer.SpecialTokens;
        if (specialTokens is not null && !string.IsNullOrEmpty(specialTokens.EosToken))
        {
            return _tokenizer.Vocabulary.GetTokenId(specialTokens.EosToken);
        }
        // Fallback: use SEP token if EOS is not defined
        if (specialTokens is not null && !string.IsNullOrEmpty(specialTokens.SepToken))
        {
            return _tokenizer.Vocabulary.GetTokenId(specialTokens.SepToken);
        }
        return 102; // Default BERT [SEP]
    }

    /// <summary>
    /// Decodes token IDs to text.
    /// </summary>
    private string DecodeTokens(IEnumerable<int> tokenIds)
    {
        return _tokenizer.Decode(tokenIds.ToList());
    }

    /// <summary>
    /// Finds argmax of last dimension at the last sequence position for autoregressive decoding.
    /// </summary>
    private int ArgMax(Tensor<T> logits)
    {
        int lastDim = logits.Shape[^1];
        int bestIdx = 0;

        // For 3D tensor [batch, seq_len, vocab_size], use last sequence position
        // For 2D tensor [batch, vocab_size], use first batch
        int seqPos = logits.Shape.Length == 3 ? logits.Shape[1] - 1 : 0;
        T bestVal = logits.Shape.Length == 3 ? logits[0, seqPos, 0] : logits[0, 0];

        for (int i = 1; i < lastDim; i++)
        {
            T val = logits.Shape.Length == 3 ? logits[0, seqPos, i] : logits[0, i];
            if (NumOps.ToDouble(val) > NumOps.ToDouble(bestVal))
            {
                bestVal = val;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /// <summary>
    /// Samples token with temperature at the last sequence position for autoregressive decoding.
    /// </summary>
    private int SampleWithTemperature(Tensor<T> logits, double temperature, Random random)
    {
        int lastDim = logits.Shape[^1];
        var probs = new double[lastDim];
        double maxLogit = double.MinValue;

        // For 3D tensor [batch, seq_len, vocab_size], use last sequence position
        int seqPos = logits.Shape.Length == 3 ? logits.Shape[1] - 1 : 0;

        // Find max for numerical stability
        for (int i = 0; i < lastDim; i++)
        {
            double val = NumOps.ToDouble(logits.Shape.Length == 3 ? logits[0, seqPos, i] : logits[0, i]);
            if (val > maxLogit) maxLogit = val;
        }

        // Compute softmax with temperature
        double sum = 0;
        for (int i = 0; i < lastDim; i++)
        {
            double val = NumOps.ToDouble(logits.Shape.Length == 3 ? logits[0, seqPos, i] : logits[0, i]);
            probs[i] = Math.Exp((val - maxLogit) / temperature);
            sum += probs[i];
        }

        // Normalize
        for (int i = 0; i < lastDim; i++)
        {
            probs[i] /= sum;
        }

        // Sample
        double r = random.NextDouble();
        double cumSum = 0;
        for (int i = 0; i < lastDim; i++)
        {
            cumSum += probs[i];
            if (r < cumSum) return i;
        }

        return lastDim - 1;
    }

    #endregion

    #region Training Support

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Training is only supported in native mode.");
        }

        // BLIP uses multiple training objectives:
        // 1. Image-Text Contrastive (ITC) - like CLIP
        // 2. Image-Text Matching (ITM) - binary classification
        // 3. Language Modeling (LM) - caption generation

        // For now, implement ITC like CLIP
        // Full implementation would include all three objectives
        SetTrainingMode(true);

        try
        {
            // Forward pass through vision encoder
            var imageOutput = input;
            foreach (var layer in _visionEncoderLayers)
            {
                imageOutput = layer.Forward(imageOutput);
            }

            // Forward pass through text encoder (target contains text tokens)
            var textOutput = target;
            foreach (var layer in _textEncoderLayers)
            {
                textOutput = layer.Forward(textOutput);
            }

            // Compute contrastive loss and backpropagate
            LastLoss = LossFunction.CalculateLoss(imageOutput.ToVector(), textOutput.ToVector());
            var lossGradient = LossFunction.CalculateDerivative(imageOutput.ToVector(), textOutput.ToVector());
            var gradient = Tensor<T>.FromVector(lossGradient);

            // Backward pass through text encoder
            foreach (var layer in _textEncoderLayers.AsEnumerable().Reverse())
            {
                gradient = layer.Backward(gradient);
            }

            // Backward pass through vision encoder
            var visionGradient = Tensor<T>.FromVector(LossFunction.CalculateDerivative(imageOutput.ToVector(), textOutput.ToVector()));
            foreach (var layer in _visionEncoderLayers.AsEnumerable().Reverse())
            {
                visionGradient = layer.Backward(visionGradient);
            }
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Default prediction returns image embedding
        var embedding = GetImageEmbedding(input);
        return Tensor<T>.FromVector(embedding);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new BlipNeuralNetwork<T>(
                Architecture,
                _imageSize,
                3,
                _patchSize,
                _vocabularySize,
                _maxSequenceLength,
                _embeddingDimension,
                _hiddenDim,
                _numLayers,
                _numDecoderLayers,
                _numHeads,
                _mlpDim,
                _tokenizer,
                null,
                null);
        }
        else
        {
            string visionPath = _visionEncoderPath ?? string.Empty;
            string textPath = _textEncoderPath ?? string.Empty;
            string decoderPath = _textDecoderPath ?? string.Empty;

            if (string.IsNullOrEmpty(visionPath) || string.IsNullOrEmpty(textPath) || string.IsNullOrEmpty(decoderPath))
            {
                throw new InvalidOperationException("ONNX model paths required for ONNX mode.");
            }

            return new BlipNeuralNetwork<T>(
                Architecture,
                visionPath,
                textPath,
                decoderPath,
                _tokenizer,
                _embeddingDimension,
                _maxSequenceLength,
                _imageSize,
                null,
                null);
        }
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount
    {
        get
        {
            if (!_useNativeMode) return 0;

            int count = 0;

            // CLS tokens and positional embeddings first
            if (_visionClsToken is not null) count += _visionClsToken.Rows * _visionClsToken.Columns;
            if (_textClsToken is not null) count += _textClsToken.Rows * _textClsToken.Columns;
            if (_visionPositionalEmbeddings is not null) count += _visionPositionalEmbeddings.Rows * _visionPositionalEmbeddings.Columns;
            if (_textPositionalEmbeddings is not null) count += _textPositionalEmbeddings.Rows * _textPositionalEmbeddings.Columns;

            // Add layer parameters from all native layer lists
            foreach (var layer in _visionEncoderLayers)
            {
                count += layer.ParameterCount;
            }

            foreach (var layer in _textEncoderLayers)
            {
                count += layer.ParameterCount;
            }

            foreach (var layer in _textDecoderLayers)
            {
                count += layer.ParameterCount;
            }

            foreach (var layer in _crossAttentionLayers)
            {
                count += layer.ParameterCount;
            }

            return count;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(ParameterCount);
        int index = 0;

        if (!_useNativeMode) return parameters;

        // CLS tokens and positional embeddings first
        if (_visionClsToken is not null)
        {
            for (int i = 0; i < _visionClsToken.Rows; i++)
            {
                for (int j = 0; j < _visionClsToken.Columns; j++)
                {
                    parameters[index++] = _visionClsToken[i, j];
                }
            }
        }

        if (_textClsToken is not null)
        {
            for (int i = 0; i < _textClsToken.Rows; i++)
            {
                for (int j = 0; j < _textClsToken.Columns; j++)
                {
                    parameters[index++] = _textClsToken[i, j];
                }
            }
        }

        if (_visionPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _visionPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < _visionPositionalEmbeddings.Columns; j++)
                {
                    parameters[index++] = _visionPositionalEmbeddings[i, j];
                }
            }
        }

        if (_textPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _textPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < _textPositionalEmbeddings.Columns; j++)
                {
                    parameters[index++] = _textPositionalEmbeddings[i, j];
                }
            }
        }

        // Layer parameters from all native layer lists
        foreach (var layer in _visionEncoderLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                parameters[index++] = layerParams[i];
            }
        }

        foreach (var layer in _textEncoderLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                parameters[index++] = layerParams[i];
            }
        }

        foreach (var layer in _textDecoderLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                parameters[index++] = layerParams[i];
            }
        }

        foreach (var layer in _crossAttentionLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                parameters[index++] = layerParams[i];
            }
        }

        return parameters;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int expectedCount = ParameterCount;
        if (parameters.Length != expectedCount)
        {
            throw new ArgumentException(
                $"Expected {expectedCount} parameters, but got {parameters.Length}",
                nameof(parameters));
        }

        if (!_useNativeMode) return;

        int index = 0;

        // CLS tokens and positional embeddings first
        if (_visionClsToken is not null)
        {
            for (int i = 0; i < _visionClsToken.Rows; i++)
            {
                for (int j = 0; j < _visionClsToken.Columns; j++)
                {
                    _visionClsToken[i, j] = parameters[index++];
                }
            }
        }

        if (_textClsToken is not null)
        {
            for (int i = 0; i < _textClsToken.Rows; i++)
            {
                for (int j = 0; j < _textClsToken.Columns; j++)
                {
                    _textClsToken[i, j] = parameters[index++];
                }
            }
        }

        if (_visionPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _visionPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < _visionPositionalEmbeddings.Columns; j++)
                {
                    _visionPositionalEmbeddings[i, j] = parameters[index++];
                }
            }
        }

        if (_textPositionalEmbeddings is not null)
        {
            for (int i = 0; i < _textPositionalEmbeddings.Rows; i++)
            {
                for (int j = 0; j < _textPositionalEmbeddings.Columns; j++)
                {
                    _textPositionalEmbeddings[i, j] = parameters[index++];
                }
            }
        }

        // Update layer parameters from all native layer lists
        foreach (var layer in _visionEncoderLayers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.Slice(index, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                index += layerParameterCount;
            }
        }

        foreach (var layer in _textEncoderLayers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.Slice(index, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                index += layerParameterCount;
            }
        }

        foreach (var layer in _textDecoderLayers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.Slice(index, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                index += layerParameterCount;
            }
        }

        foreach (var layer in _crossAttentionLayers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = parameters.Slice(index, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                index += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Retrieves metadata about the BLIP neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.Blip,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "EmbeddingDimension", _embeddingDimension },
                { "MaxSequenceLength", _maxSequenceLength },
                { "ImageSize", _imageSize },
                { "HiddenDimension", _hiddenDim },
                { "NumEncoderLayers", _numLayers },
                { "NumDecoderLayers", _numDecoderLayers },
                { "NumHeads", _numHeads },
                { "VocabularySize", _vocabularySize },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
                { "TaskType", Architecture.TaskType.ToString() },
                { "ParameterCount", ParameterCount },
                { "UseNativeMode", _useNativeMode }
            },
            ModelData = this.Serialize()
        };
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embeddingDimension);
        writer.Write(_maxSequenceLength);
        writer.Write(_imageSize);
        writer.Write(_hiddenDim);
        writer.Write(_numLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_numHeads);
        writer.Write(_mlpDim);
        writer.Write(_patchSize);
        writer.Write(_vocabularySize);
        writer.Write(_useNativeMode);
        writer.Write(_optimizer?.GetType().Name ?? "Adam");
        writer.Write(_lossFunction?.GetType().Name ?? "Contrastive");
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _embeddingDimension = reader.ReadInt32();
        _maxSequenceLength = reader.ReadInt32();
        _imageSize = reader.ReadInt32();
        _hiddenDim = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numDecoderLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _mlpDim = reader.ReadInt32();
        _patchSize = reader.ReadInt32();
        _vocabularySize = reader.ReadInt32();
        _useNativeMode = reader.ReadBoolean();
        _ = reader.ReadString(); // optimizer type
        _ = reader.ReadString(); // loss function type
    }

    #endregion

    #region IDisposable

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _visionEncoder?.Dispose();
            _textEncoder?.Dispose();
            _textDecoder?.Dispose();
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
    public Task<Vector<T>> EmbedAsync(string text)
    {
        return Task.FromResult(EncodeText(text));
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
    public Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts)
    {
        return Task.FromResult(EncodeTextBatch(texts));
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
