using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// CLIP (Contrastive Language-Image Pre-training) neural network for joint text-image embeddings.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLIP was introduced by OpenAI and trained on 400 million (image, text) pairs from the internet.
/// It learns to associate images with their textual descriptions in a shared embedding space.
/// This implementation uses pre-trained ONNX models for the vision and text encoders.
/// </para>
/// <para><b>For Beginners:</b> CLIP learned by looking at millions of images with captions.
///
/// Training process (simplified):
/// 1. Show CLIP an image of a dog and the caption "a golden retriever"
/// 2. CLIP learns that these should have similar embeddings
/// 3. Show CLIP the same dog image and "a sports car"
/// 4. CLIP learns that these should have different embeddings
/// 5. Repeat 400 million times!
///
/// Now CLIP can:
/// - Match any image to any text description
/// - Classify images without seeing examples (zero-shot)
/// - Power image search engines
/// - Find the best caption for any image
///
/// This implementation loads pre-trained ONNX models, so you get all this capability
/// without having to train the model yourself!
/// </para>
/// </remarks>
public class ClipNeuralNetwork<T> : NeuralNetworkBase<T>, IMultimodalEmbedding<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this CLIP network uses native layers (true) or ONNX models (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for the image encoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The image encoder is typically a Vision Transformer (ViT) or ResNet that processes
    /// images and outputs embedding vectors. Only used when _useNativeMode is false.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "eye" of CLIP - it looks at images
    /// and converts them into numbers (vectors) that represent their visual content.
    /// </para>
    /// </remarks>
    private readonly InferenceSession? _imageEncoder;

    /// <summary>
    /// The ONNX inference session for the text encoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The text encoder is typically a Transformer that processes tokenized text
    /// and outputs embedding vectors. Only used when _useNativeMode is false.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "reader" of CLIP - it reads text
    /// and converts it into numbers (vectors) that represent its meaning.
    /// </para>
    /// </remarks>
    private readonly InferenceSession? _textEncoder;

    /// <summary>
    /// Path to the image encoder ONNX model (stored for CreateNewInstance).
    /// </summary>
    private readonly string? _imageEncoderPath;

    /// <summary>
    /// Path to the text encoder ONNX model (stored for CreateNewInstance).
    /// </summary>
    private readonly string? _textEncoderPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// The layers that compose the vision encoder (when using native mode).
    /// </summary>
    private readonly List<ILayer<T>> _visionEncoderLayers;

    /// <summary>
    /// The layers that compose the text encoder (when using native mode).
    /// </summary>
    private readonly List<ILayer<T>> _textEncoderLayers;

    /// <summary>
    /// The CLS token for the vision encoder (learnable parameter).
    /// </summary>
    private Vector<T>? _visionClsToken;

    /// <summary>
    /// The CLS token for the text encoder (learnable parameter).
    /// </summary>
    private Vector<T>? _textClsToken;

    /// <summary>
    /// Positional embeddings for the vision encoder.
    /// </summary>
    private Matrix<T>? _visionPositionalEmbeddings;

    /// <summary>
    /// The number of image patches for native mode.
    /// </summary>
    private readonly int _numPatches;

    /// <summary>
    /// The patch size for native vision encoder.
    /// </summary>
    private readonly int _patchSize;

    /// <summary>
    /// Vocabulary size for native text encoder.
    /// </summary>
    private readonly int _vocabularySize;

    /// <summary>
    /// Number of channels in input images.
    /// </summary>
    private readonly int _channels;

    /// <summary>
    /// Number of transformer layers in each encoder.
    /// </summary>
    private readonly int _numLayers;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Feed-forward dimension in transformer layers.
    /// </summary>
    private readonly int _mlpDim;

    /// <summary>
    /// Hidden dimension for transformer layers.
    /// </summary>
    private readonly int _hiddenDim;

    #endregion

    #region Common Fields

    /// <summary>
    /// The tokenizer for converting text to token IDs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP uses a BPE (Byte Pair Encoding) tokenizer with a vocabulary of 49408 tokens.
    /// </para>
    /// <para><b>For Beginners:</b> Before CLIP can read text, it needs to break it
    /// into pieces called "tokens". A tokenizer is like a text splitter that breaks
    /// "Hello World" into ["Hello", "World"] or even smaller pieces.
    /// </para>
    /// </remarks>
    private readonly ITokenizer? _tokenizer;

    /// <summary>
    /// The optimizer used for fine-tuning the network.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The loss function for contrastive learning.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The dimensionality of the embedding space.
    /// </summary>
    private readonly int _embeddingDimension;

    /// <summary>
    /// The maximum sequence length for text input.
    /// </summary>
    private readonly int _maxSequenceLength;

    /// <summary>
    /// The expected image size (height and width).
    /// </summary>
    private readonly int _imageSize;

    #endregion

    /// <summary>
    /// Gets the dimensionality of the embedding vectors produced by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Both text and image embeddings will have this same dimension.
    /// Common values are 512 (CLIP ViT-B/32) or 768 (CLIP ViT-L/14).
    /// </para>
    /// </remarks>
    public int EmbeddingDimension => _embeddingDimension;

    /// <summary>
    /// Gets the maximum number of tokens the text encoder can process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP models typically have a maximum sequence length of 77 tokens.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <summary>
    /// Gets the expected image size (height and width) for the vision encoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP models expect square images of a specific size (e.g., 224x224).
    /// </para>
    /// </remarks>
    public int ImageSize => _imageSize;

    #region Constructors

    /// <summary>
    /// Initializes a new CLIP network with native layers (no ONNX dependencies).
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="imageSize">The expected image size (height and width). Default is 224.</param>
    /// <param name="channels">The number of image channels. Default is 3 (RGB).</param>
    /// <param name="patchSize">The patch size for the vision encoder. Default is 16.</param>
    /// <param name="vocabularySize">The vocabulary size for the text encoder. Default is 49408 (CLIP vocab).</param>
    /// <param name="maxSequenceLength">The maximum sequence length for text. Default is 77.</param>
    /// <param name="embeddingDimension">The output embedding dimension. Default is 512.</param>
    /// <param name="hiddenDim">The hidden dimension for transformer layers. Default is 768.</param>
    /// <param name="numLayers">The number of transformer layers. Default is 12.</param>
    /// <param name="numHeads">The number of attention heads. Default is 12.</param>
    /// <param name="mlpDim">The feed-forward dimension. Default is 3072.</param>
    /// <param name="tokenizer">Optional tokenizer for text processing.</param>
    /// <param name="optimizer">The optimization algorithm. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function. If null, ContrastiveLoss is used.</param>
    /// <param name="maxGradNorm">The maximum gradient norm for gradient clipping.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a fully native CLIP network using the library's built-in layers.
    /// It supports full training and customization without requiring external ONNX models.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a CLIP network that you can train from scratch
    /// or fine-tune for your specific use case. Unlike the ONNX version, this uses native
    /// layers and supports full backpropagation and training.
    ///
    /// Key components:
    /// - Vision Encoder: Converts images to embeddings using patch embedding + transformer layers
    /// - Text Encoder: Converts text to embeddings using token embedding + positional encoding + transformer layers
    /// - Both encoders output vectors in the same embedding space for similarity comparison
    /// </para>
    /// </remarks>
    public ClipNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int imageSize = 224,
        int channels = 3,
        int patchSize = 16,
        int vocabularySize = 49408,
        int maxSequenceLength = 77,
        int embeddingDimension = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int mlpDim = 3072,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture,
               lossFunction ?? new ContrastiveLoss<T>(),
               maxGradNorm)
    {
        // Validate parameters
        if (imageSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageSize), "Image size must be positive.");
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), "Channels must be positive.");
        if (patchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(patchSize), "Patch size must be positive.");
        if (imageSize % patchSize != 0)
            throw new ArgumentException($"Image size ({imageSize}) must be divisible by patch size ({patchSize}).", nameof(patchSize));
        if (vocabularySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabularySize), "Vocabulary size must be positive.");
        if (maxSequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSequenceLength), "Max sequence length must be positive.");
        if (embeddingDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(embeddingDimension), "Embedding dimension must be positive.");
        if (hiddenDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenDim), "Hidden dimension must be positive.");
        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be positive.");
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "Number of heads must be positive.");
        if (hiddenDim % numHeads != 0)
            throw new ArgumentException($"Hidden dimension ({hiddenDim}) must be divisible by number of heads ({numHeads}).", nameof(numHeads));
        if (mlpDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(mlpDim), "MLP dimension must be positive.");

        // Set native mode
        _useNativeMode = true;

        // Store configuration
        _imageSize = imageSize;
        _channels = channels;
        _patchSize = patchSize;
        _vocabularySize = vocabularySize;
        _maxSequenceLength = maxSequenceLength;
        _embeddingDimension = embeddingDimension;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _mlpDim = mlpDim;
        _numPatches = (imageSize / patchSize) * (imageSize / patchSize);

        // Initialize common components
        _tokenizer = tokenizer;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new ContrastiveLoss<T>();

        // Initialize encoder layer lists
        _visionEncoderLayers = new List<ILayer<T>>();
        _textEncoderLayers = new List<ILayer<T>>();

        // ONNX fields are null in native mode
        _imageEncoder = null;
        _textEncoder = null;
        _imageEncoderPath = null;
        _textEncoderPath = null;

        // Initialize native layers
        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the ClipNeuralNetwork class using ONNX models.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <param name="imageEncoderPath">Path to the ONNX model for the image encoder.</param>
    /// <param name="textEncoderPath">Path to the ONNX model for the text encoder.</param>
    /// <param name="tokenizer">The tokenizer for text processing. Required - use ClipTokenizerFactory to create one.</param>
    /// <param name="optimizer">The optimization algorithm. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">The loss function. If null, ContrastiveLoss is used.</param>
    /// <param name="embeddingDimension">The embedding dimension. Default is 512.</param>
    /// <param name="maxSequenceLength">The maximum sequence length. Default is 77.</param>
    /// <param name="imageSize">The expected image size. Default is 224.</param>
    /// <param name="maxGradNorm">The maximum gradient norm for gradient clipping.</param>
    /// <remarks>
    /// <para>
    /// This constructor loads pre-trained ONNX models for both the vision and text encoders.
    /// The models should be exported from a pre-trained CLIP model (e.g., from HuggingFace or OpenAI).
    /// </para>
    /// <para><b>For Beginners:</b> To use CLIP, you need two model files:
    /// 1. An image encoder (converts images to vectors)
    /// 2. A text encoder (converts text to vectors)
    ///
    /// These are typically downloaded from model repositories like HuggingFace.
    /// You also need a tokenizer to break text into tokens that the model understands.
    /// </para>
    /// </remarks>
    public ClipNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture,
        string imageEncoderPath,
        string textEncoderPath,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int embeddingDimension = 512,
        int maxSequenceLength = 77,
        int imageSize = 224,
        double maxGradNorm = 1.0)
        : base(architecture,
               lossFunction ?? new ContrastiveLoss<T>(),
               maxGradNorm)
    {
        if (string.IsNullOrWhiteSpace(imageEncoderPath))
            throw new ArgumentException("Image encoder path cannot be null or empty.", nameof(imageEncoderPath));
        if (string.IsNullOrWhiteSpace(textEncoderPath))
            throw new ArgumentException("Text encoder path cannot be null or empty.", nameof(textEncoderPath));
        if (!File.Exists(imageEncoderPath))
            throw new FileNotFoundException($"Image encoder model not found: {imageEncoderPath}");
        if (!File.Exists(textEncoderPath))
            throw new FileNotFoundException($"Text encoder model not found: {textEncoderPath}");
        if (embeddingDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(embeddingDimension), "Embedding dimension must be positive.");
        if (maxSequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSequenceLength), "Max sequence length must be positive.");
        if (imageSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageSize), "Image size must be positive.");

        // Set ONNX mode
        _useNativeMode = false;

        // Store configuration
        _embeddingDimension = embeddingDimension;
        _maxSequenceLength = maxSequenceLength;
        _imageSize = imageSize;
        _imageEncoderPath = imageEncoderPath;
        _textEncoderPath = textEncoderPath;

        // Initialize native-mode fields to defaults (unused in ONNX mode)
        _visionEncoderLayers = new List<ILayer<T>>();
        _textEncoderLayers = new List<ILayer<T>>();
        _numPatches = 0;
        _patchSize = 16;
        _vocabularySize = 49408;
        _channels = 3;
        _numLayers = 12;
        _numHeads = 12;
        _mlpDim = 3072;
        _hiddenDim = 768;

        // Load ONNX models with proper cleanup on failure
        InferenceSession? tempImageEncoder = null;
        InferenceSession? tempTextEncoder = null;
        try
        {
            tempImageEncoder = new InferenceSession(imageEncoderPath);
            tempTextEncoder = new InferenceSession(textEncoderPath);

            // Tokenizer is required; validate argument
            if (tokenizer is null)
                throw new ArgumentNullException(nameof(tokenizer));

            _tokenizer = tokenizer;

            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _lossFunction = lossFunction ?? new ContrastiveLoss<T>();

            // Transfer ownership after all validation passes
            _imageEncoder = tempImageEncoder;
            _textEncoder = tempTextEncoder;
            tempImageEncoder = null;
            tempTextEncoder = null;

            // Note: InitializeLayers is called from base class via overridden method pattern.
            // While this triggers a virtual call warning, it is intentional as part of the
            // template method pattern for neural network initialization.
            InitializeLayers();
        }
        catch
        {
            // Dispose resources if constructor fails partway through
            tempImageEncoder?.Dispose();
            tempTextEncoder?.Dispose();
            throw;
        }
    }

    #endregion

    /// <summary>
    /// Initializes the layers of the neural network based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In native mode, this builds the full vision and text encoders using library layers.
    /// In ONNX mode, layers are optional projection heads for fine-tuning.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the internal structure of CLIP.
    ///
    /// In native mode, it builds:
    /// - Vision Encoder: PatchEmbedding + Transformer layers + Projection
    /// - Text Encoder: TokenEmbedding + Positional + Transformer layers + Projection
    ///
    /// In ONNX mode, the encoders are loaded from external files, and this just adds
    /// optional projection layers for customization.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (_useNativeMode)
        {
            InitializeNativeLayers();
        }
        else
        {
            InitializeOnnxProjectionLayers();
        }
    }

    /// <summary>
    /// Initializes the native vision and text encoder layers.
    /// </summary>
    private void InitializeNativeLayers()
    {
        // Build Vision Encoder
        // 1. Patch Embedding Layer
        _visionEncoderLayers.Add(new PatchEmbeddingLayer<T>(
            _imageSize, _imageSize, _channels, _patchSize, _hiddenDim));

        // 2. Transformer Encoder Layers for vision
        for (int i = 0; i < _numLayers; i++)
        {
            _visionEncoderLayers.Add(new TransformerEncoderLayer<T>(_hiddenDim, _numHeads, _mlpDim));
        }

        // 3. Vision Projection Layer (project from hiddenDim to embeddingDim)
        _visionEncoderLayers.Add(new DenseLayer<T>(_hiddenDim, _embeddingDimension, (IActivationFunction<T>?)null));

        // Build Text Encoder
        // 1. Token Embedding Layer
        _textEncoderLayers.Add(new EmbeddingLayer<T>(_vocabularySize, _hiddenDim));

        // 2. Positional Encoding Layer
        _textEncoderLayers.Add(new PositionalEncodingLayer<T>(_maxSequenceLength, _hiddenDim));

        // 3. Transformer Encoder Layers for text
        for (int i = 0; i < _numLayers; i++)
        {
            _textEncoderLayers.Add(new TransformerEncoderLayer<T>(_hiddenDim, _numHeads, _mlpDim));
        }

        // 4. Text Projection Layer (project from hiddenDim to embeddingDim)
        _textEncoderLayers.Add(new DenseLayer<T>(_hiddenDim, _embeddingDimension, (IActivationFunction<T>?)null));

        // Initialize CLS tokens and positional embeddings for native mode
        _visionClsToken = new Vector<T>(_hiddenDim);
        _textClsToken = new Vector<T>(_hiddenDim);
        _visionPositionalEmbeddings = new Matrix<T>(_numPatches + 1, _hiddenDim);

        InitializeVisionClsToken();
        InitializeTextClsToken();
        InitializeVisionPositionalEmbeddings();

        // Add all layers to the base Layers collection for parameter management
        Layers.AddRange(_visionEncoderLayers);
        Layers.AddRange(_textEncoderLayers);

        // Add optional user-provided layers
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
    }

    /// <summary>
    /// Initializes projection layers for ONNX mode.
    /// </summary>
    private void InitializeOnnxProjectionLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // CLIP uses ONNX encoders - layers are optional projection heads
            Layers.AddRange(LayerHelper<T>.CreateDefaultClipLayers(Architecture, _embeddingDimension));
        }
    }

    /// <summary>
    /// Initializes the vision CLS token with random values.
    /// </summary>
    private void InitializeVisionClsToken()
    {
        if (_visionClsToken is null) return;

        T scale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _hiddenDim));
        for (int i = 0; i < _hiddenDim; i++)
        {
            _visionClsToken[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    /// <summary>
    /// Initializes the text CLS token with random values.
    /// </summary>
    private void InitializeTextClsToken()
    {
        if (_textClsToken is null) return;

        T scale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _hiddenDim));
        for (int i = 0; i < _hiddenDim; i++)
        {
            _textClsToken[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }
    }

    /// <summary>
    /// Initializes the vision positional embeddings with random values.
    /// </summary>
    private void InitializeVisionPositionalEmbeddings()
    {
        if (_visionPositionalEmbeddings is null) return;

        T scale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _hiddenDim));
        for (int i = 0; i < _visionPositionalEmbeddings.Rows; i++)
        {
            for (int j = 0; j < _visionPositionalEmbeddings.Columns; j++)
            {
                _visionPositionalEmbeddings[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }
    }

    /// <summary>
    /// Converts a single text string into an embedding vector.
    /// </summary>
    /// <param name="text">The text to embed.</param>
    /// <returns>A normalized embedding vector.</returns>
    /// <remarks>
    /// <para>
    /// The text is tokenized, padded/truncated to MaxSequenceLength, processed through
    /// the text encoder, and L2-normalized.
    /// </para>
    /// </remarks>
    public Vector<T> GetTextEmbedding(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Text cannot be null or empty.", nameof(text));

        if (_useNativeMode)
        {
            return GetTextEmbeddingNative(text);
        }

        return GetTextEmbeddingOnnx(text);
    }

    /// <summary>
    /// Gets text embedding using native layers with Engine-accelerated operations.
    /// </summary>
    private Vector<T> GetTextEmbeddingNative(string text)
    {
        // Get token IDs
        int[] tokenIds;
        if (_tokenizer is not null)
        {
            var encodingOptions = new EncodingOptions
            {
                MaxLength = _maxSequenceLength,
                Truncation = true,
                Padding = true,
                PaddingSide = "right",
                TruncationSide = "right",
                AddSpecialTokens = true,
                ReturnAttentionMask = true
            };
            var tokenResult = _tokenizer.Encode(text, encodingOptions);
            tokenIds = tokenResult.TokenIds.ToArray();
        }
        else
        {
            // Simple fallback tokenization: convert characters to token IDs
            tokenIds = new int[Math.Min(text.Length, _maxSequenceLength)];
            for (int i = 0; i < tokenIds.Length; i++)
            {
                tokenIds[i] = Math.Min((int)text[i], _vocabularySize - 1);
            }
        }

        // Create input tensor [1, seqLen] using vectorized fill
        int seqLen = tokenIds.Length;
        var inputTensor = new Tensor<T>(new[] { 1, seqLen });
        for (int i = 0; i < seqLen; i++)
        {
            inputTensor[0, i] = NumOps.FromDouble(tokenIds[i]);
        }

        // Forward through text encoder layers
        // Layer 0: EmbeddingLayer - outputs [batch, seqLen, hiddenDim]
        var embedded = _textEncoderLayers[0].Forward(inputTensor);

        // Layer 1: PositionalEncodingLayer
        var withPositional = _textEncoderLayers[1].Forward(embedded);

        // Prepare CLS token as tensor [1, 1, hiddenDim] for concatenation
        int batchSize = withPositional.Shape[0];
        var clsTokenTensor = new Tensor<T>(new[] { batchSize, 1, _hiddenDim });
        if (_textClsToken is not null)
        {
            // Use Engine to set the CLS token
            var clsRow = Tensor<T>.FromVector(_textClsToken);
            var clsExpanded = Engine.TensorExpandDims(clsRow, 0); // [1, hiddenDim]
            clsExpanded = Engine.TensorExpandDims(clsExpanded, 0); // [1, 1, hiddenDim]
            // Broadcast to batch size
            for (int b = 0; b < batchSize; b++)
            {
                Engine.TensorCopy(clsExpanded, Engine.TensorSlice(clsTokenTensor, new[] { b, 0, 0 }, new[] { 1, 1, _hiddenDim }));
            }
        }

        // Concatenate CLS token with embedded tokens: [batch, 1, hidden] + [batch, seqLen, hidden]
        var withCls = Engine.TensorConcatenate(new[] { clsTokenTensor, withPositional }, axis: 1);

        // Layers 2 to N-1: TransformerEncoderLayers
        var current = withCls;
        for (int i = 2; i < _textEncoderLayers.Count - 1; i++)
        {
            current = _textEncoderLayers[i].Forward(current);
        }

        // Extract CLS token embedding (position 0) using Engine slice
        var clsEmbedding = Engine.TensorSlice(current, new[] { 0, 0, 0 }, new[] { batchSize, 1, _hiddenDim });
        clsEmbedding = Engine.TensorSqueeze(clsEmbedding, 1); // [batch, hiddenDim]

        // Last layer: Projection to embedding dimension
        // Reshape for DenseLayer: [batch, 1, hiddenDim]
        var reshapedForDense = Engine.TensorExpandDims(clsEmbedding, 1);
        var projected = _textEncoderLayers[^1].Forward(reshapedForDense);

        // Extract first batch item and convert to vector
        var firstBatch = Engine.TensorSlice(projected, new[] { 0, 0, 0 }, new[] { 1, 1, _embeddingDimension });
        var embedding = firstBatch.ToVector();

        return NormalizeVector(embedding);
    }

    /// <summary>
    /// Gets text embedding using ONNX model.
    /// </summary>
    private Vector<T> GetTextEmbeddingOnnx(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer is required for ONNX mode.");
        if (_textEncoder is null)
            throw new InvalidOperationException("Text encoder is not initialized.");

        // Tokenize text with CLIP-specific options
        var encodingOptions = new EncodingOptions
        {
            MaxLength = _maxSequenceLength,
            Truncation = true,
            Padding = true,
            PaddingSide = "right",
            TruncationSide = "right",
            AddSpecialTokens = true,
            ReturnAttentionMask = true
        };
        var tokenResult = _tokenizer.Encode(text, encodingOptions);

        // Create input tensor for ONNX
        var inputIds = tokenResult.TokenIds.ToArray();
        var inputTensor = new OnnxTensors.DenseTensor<long>(inputIds.Select(i => (long)i).ToArray(), new[] { 1, inputIds.Length });

        // Run text encoder
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
        };

        using var results = _textEncoder.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Convert to Vector<T> and normalize
        var embedding = ConvertToVector(outputTensor);
        return NormalizeVector(embedding);
    }

    /// <summary>
    /// Converts multiple text strings into embedding vectors in a batch operation.
    /// </summary>
    /// <param name="texts">The texts to embed.</param>
    /// <returns>A collection of normalized embedding vectors.</returns>
    /// <remarks>
    /// <para>
    /// This method uses true batched inference when all tokenized sequences have the same length.
    /// When sequence lengths differ, it falls back to sequential processing.
    /// </para>
    /// </remarks>
    public IEnumerable<Vector<T>> GetTextEmbeddings(IEnumerable<string> texts)
    {
        if (texts is null)
            throw new ArgumentNullException(nameof(texts));

        var textList = texts.ToList();
        if (textList.Count == 0)
            return Enumerable.Empty<Vector<T>>();

        if (_useNativeMode)
        {
            // Native mode: process each text individually
            return textList.Select(text => GetTextEmbeddingNative(text)).ToList();
        }

        return GetTextEmbeddingsOnnx(textList);
    }

    /// <summary>
    /// Gets text embeddings using ONNX model with batching support.
    /// </summary>
    private IEnumerable<Vector<T>> GetTextEmbeddingsOnnx(List<string> textList)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer is required for ONNX mode.");
        if (_textEncoder is null)
            throw new InvalidOperationException("Text encoder is not initialized.");

        // Tokenize all texts with CLIP-specific options (padding ensures uniform length)
        var encodingOptions = new EncodingOptions
        {
            MaxLength = _maxSequenceLength,
            Truncation = true,
            Padding = true,
            PaddingSide = "right",
            TruncationSide = "right",
            AddSpecialTokens = true,
            ReturnAttentionMask = true
        };
        var tokenResults = _tokenizer.EncodeBatch(textList, encodingOptions);
        var tokenResultList = tokenResults.ToList();

        // Check if all sequences have the same length for true batching
        int seqLength = tokenResultList.First().TokenIds.Count;
        bool canBatch = tokenResultList.All(r => r.TokenIds.Count == seqLength);

        if (canBatch && tokenResultList.Count > 1)
        {
            // True batched inference: create a single [batch_size, seq_length] tensor
            int batchSize = tokenResultList.Count;
            var batchedIds = new long[batchSize * seqLength];

            for (int b = 0; b < batchSize; b++)
            {
                var tokenIds = tokenResultList[b].TokenIds;
                for (int i = 0; i < seqLength; i++)
                {
                    batchedIds[b * seqLength + i] = tokenIds[i];
                }
            }

            var inputTensor = new OnnxTensors.DenseTensor<long>(batchedIds, new[] { batchSize, seqLength });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
            };

            using var results = _textEncoder.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();

            return ExtractBatchEmbeddings(outputTensor, batchSize);
        }
        else
        {
            // Fallback to sequential processing for varying sequence lengths
            return tokenResultList.Select(tokenResult =>
            {
                var inputIds = tokenResult.TokenIds.ToArray();
                var inputTensor = new OnnxTensors.DenseTensor<long>(inputIds.Select(i => (long)i).ToArray(), new[] { 1, inputIds.Length });

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
                };

                using var results = _textEncoder.Run(inputs);
                var outputTensor = results.First().AsTensor<float>();

                var embedding = ConvertToVector(outputTensor);
                return NormalizeVector(embedding);
            }).ToList();
        }
    }

    /// <summary>
    /// Converts a single image into an embedding vector.
    /// </summary>
    /// <param name="image">The preprocessed image tensor with shape [channels, height, width].</param>
    /// <returns>A normalized embedding vector.</returns>
    /// <remarks>
    /// <para>
    /// The image should be preprocessed (resized to ImageSize, normalized) before calling this method.
    /// </para>
    /// </remarks>
    public Vector<T> GetImageEmbedding(Tensor<T> image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));

        ValidateImageShape(image);

        if (_useNativeMode)
        {
            return GetImageEmbeddingNative(image);
        }

        return GetImageEmbeddingOnnx(image);
    }

    /// <summary>
    /// Gets image embedding using native layers with Engine-accelerated operations.
    /// </summary>
    private Vector<T> GetImageEmbeddingNative(Tensor<T> image)
    {
        // Add batch dimension if needed: [C, H, W] -> [1, C, H, W]
        Tensor<T> batchedImage = image.Shape.Length == 3
            ? Engine.TensorExpandDims(image, 0)
            : image;

        // Forward through vision encoder layers
        // Layer 0: PatchEmbeddingLayer - outputs [batch, numPatches, hiddenDim]
        var patchEmbeddings = _visionEncoderLayers[0].Forward(batchedImage);

        // Prepare CLS token as tensor [1, 1, hiddenDim] for concatenation
        int batchSize = patchEmbeddings.Shape[0];
        int numPatches = patchEmbeddings.Shape[1];

        // Create CLS token tensor and broadcast to batch
        var clsTokenTensor = new Tensor<T>(new[] { batchSize, 1, _hiddenDim });
        if (_visionClsToken is not null)
        {
            // Use Engine to set the CLS token for each batch item
            var clsRow = Tensor<T>.FromVector(_visionClsToken);
            var clsExpanded = Engine.TensorExpandDims(clsRow, 0); // [1, hiddenDim]
            clsExpanded = Engine.TensorExpandDims(clsExpanded, 0); // [1, 1, hiddenDim]
            // Broadcast to batch size
            for (int b = 0; b < batchSize; b++)
            {
                Engine.TensorCopy(clsExpanded, Engine.TensorSlice(clsTokenTensor, new[] { b, 0, 0 }, new[] { 1, 1, _hiddenDim }));
            }
        }

        // Concatenate CLS token with patch embeddings: [batch, 1, hidden] + [batch, numPatches, hidden]
        var withCls = Engine.TensorConcatenate(new[] { clsTokenTensor, patchEmbeddings }, axis: 1);

        // Add positional embeddings using broadcast
        if (_visionPositionalEmbeddings is not null)
        {
            // Convert positional embeddings matrix to tensor [numPatches+1, hiddenDim]
            var posEmbTensor = Tensor<T>.FromMatrix(_visionPositionalEmbeddings);
            // Expand to [1, numPatches+1, hiddenDim] for broadcasting
            var posEmbExpanded = Engine.TensorExpandDims<T>(posEmbTensor, 0);
            // Use broadcast add to add positional embeddings to all batch items
            withCls = Engine.TensorBroadcastAdd<T>(withCls, posEmbExpanded);
        }

        // Layers 1 to N-1: TransformerEncoderLayers
        var current = withCls;
        for (int i = 1; i < _visionEncoderLayers.Count - 1; i++)
        {
            current = _visionEncoderLayers[i].Forward(current);
        }

        // Extract CLS token embedding (position 0) using slice: [batch, hiddenDim]
        var clsEmbedding = Engine.TensorSlice(current, new[] { 0, 0, 0 }, new[] { batchSize, 1, _hiddenDim });
        clsEmbedding = Engine.TensorSqueeze(clsEmbedding, 1); // [batch, hiddenDim]

        // Last layer: Projection to embedding dimension
        // Reshape for DenseLayer: [batch, 1, hiddenDim]
        var reshapedForDense = Engine.TensorExpandDims(clsEmbedding, 1);
        var projected = _visionEncoderLayers[^1].Forward(reshapedForDense);

        // Extract first batch item and convert to vector
        var firstBatch = Engine.TensorSlice(projected, new[] { 0, 0, 0 }, new[] { 1, 1, _embeddingDimension });
        var embedding = firstBatch.ToVector();

        return NormalizeVector(embedding);
    }

    /// <summary>
    /// Gets image embedding using ONNX model.
    /// </summary>
    private Vector<T> GetImageEmbeddingOnnx(Tensor<T> image)
    {
        if (_imageEncoder is null)
            throw new InvalidOperationException("Image encoder is not initialized.");

        // Convert to ONNX tensor format
        var onnxTensor = ConvertToOnnxTensor(image);

        // Run image encoder
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", onnxTensor)
        };

        using var results = _imageEncoder.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Convert to Vector<T> and normalize
        var embedding = ConvertToVector(outputTensor);
        return NormalizeVector(embedding);
    }

    /// <summary>
    /// Converts multiple images into embedding vectors in a batch operation.
    /// </summary>
    /// <param name="images">The preprocessed image tensors.</param>
    /// <returns>A collection of normalized embedding vectors.</returns>
    /// <remarks>
    /// <para>
    /// This method uses true batched inference by stacking all images into a single
    /// tensor with shape [batch_size, channels, height, width] for efficient GPU processing.
    /// </para>
    /// </remarks>
    public IEnumerable<Vector<T>> GetImageEmbeddings(IEnumerable<Tensor<T>> images)
    {
        if (images == null)
            throw new ArgumentNullException(nameof(images));

        var imageList = images.ToList();
        if (imageList.Count == 0)
            return Enumerable.Empty<Vector<T>>();

        // Validate all images
        foreach (var image in imageList)
        {
            ValidateImageShape(image);
        }

        if (_useNativeMode)
        {
            // Native mode: process each image individually
            return imageList.Select(img => GetImageEmbeddingNative(img)).ToList();
        }

        return GetImageEmbeddingsOnnx(imageList);
    }

    /// <summary>
    /// Gets image embeddings using ONNX model with batching support.
    /// </summary>
    private IEnumerable<Vector<T>> GetImageEmbeddingsOnnx(List<Tensor<T>> imageList)
    {
        if (_imageEncoder is null)
            throw new InvalidOperationException("Image encoder is not initialized.");

        // Create batched tensor: [batch_size, channels, height, width]
        int batchSize = imageList.Count;
        int channels = imageList[0].Shape[0];
        int height = imageList[0].Shape[1];
        int width = imageList[0].Shape[2];
        int imageSize = channels * height * width;

        var batchedData = new float[batchSize * imageSize];
        for (int b = 0; b < batchSize; b++)
        {
            var imageData = imageList[b].Data;
            for (int i = 0; i < imageSize && i < imageData.Length; i++)
            {
                batchedData[b * imageSize + i] = NumOps.ToFloat(imageData[i]);
            }
        }

        var batchedTensor = new OnnxTensors.DenseTensor<float>(batchedData, new[] { batchSize, channels, height, width });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", batchedTensor)
        };

        using var results = _imageEncoder.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        return ExtractBatchEmbeddings(outputTensor, batchSize);
    }

    /// <summary>
    /// Computes the similarity between a text embedding and an image embedding.
    /// </summary>
    /// <param name="textEmbedding">The text embedding vector.</param>
    /// <param name="imageEmbedding">The image embedding vector.</param>
    /// <returns>A similarity score (cosine similarity for normalized vectors).</returns>
    /// <remarks>
    /// <para>
    /// For L2-normalized embeddings, the dot product equals the cosine similarity.
    /// Values range from -1 (opposite) to 1 (identical), with 0 indicating orthogonality.
    /// </para>
    /// </remarks>
    public T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding)
    {
        if (textEmbedding == null)
            throw new ArgumentNullException(nameof(textEmbedding));
        if (imageEmbedding == null)
            throw new ArgumentNullException(nameof(imageEmbedding));
        if (textEmbedding.Length != imageEmbedding.Length)
            throw new ArgumentException("Embedding vectors must have the same dimension.");

        // Use Engine for vectorized dot product
        return Engine.DotProduct(textEmbedding, imageEmbedding);
    }

    /// <summary>
    /// Performs zero-shot image classification by comparing an image to a set of text labels.
    /// </summary>
    /// <param name="image">The preprocessed image tensor to classify.</param>
    /// <param name="classLabels">The candidate class labels.</param>
    /// <returns>A dictionary mapping each label to its probability score.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the similarity between the image and each label,
    /// then applies softmax to convert similarities into a probability distribution.
    /// </para>
    /// <para><b>For Beginners:</b> Zero-shot means we can classify images into
    /// categories we've never trained on! Just provide text descriptions of the
    /// categories and CLIP will figure out which one matches best.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, IEnumerable<string> classLabels)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (classLabels == null)
            throw new ArgumentNullException(nameof(classLabels));

        var labels = classLabels.ToList();
        if (labels.Count == 0)
            throw new ArgumentException("At least one class label is required.", nameof(classLabels));

        // Get image embedding
        var imageEmbedding = GetImageEmbedding(image);

        // Get text embeddings for all labels (with prompt template)
        var promptedLabels = labels.Select(label => $"a photo of a {label}").ToList();
        var textEmbeddings = GetTextEmbeddings(promptedLabels).ToList();

        // Compute similarities
        var similarities = new Vector<T>(labels.Count);
        for (int i = 0; i < labels.Count; i++)
        {
            similarities[i] = ComputeSimilarity(textEmbeddings[i], imageEmbedding);
        }

        // Apply softmax using Engine
        var probabilities = Engine.Softmax(similarities);

        // Create result dictionary
        var result = new Dictionary<string, T>();
        for (int i = 0; i < labels.Count; i++)
        {
            result[labels[i]] = probabilities[i];
        }

        return result;
    }

    /// <summary>
    /// Makes a prediction using the CLIP network for the given input tensor.
    /// </summary>
    /// <param name="input">The input tensor (image data).</param>
    /// <returns>The predicted embedding tensor.</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        IsTrainingMode = false;

        var embedding = GetImageEmbedding(input);
        var result = Tensor<T>.FromVector(embedding);

        IsTrainingMode = true;
        return result;
    }

    /// <summary>
    /// Trains the CLIP network using contrastive learning.
    /// </summary>
    /// <param name="input">The input tensor (image data).</param>
    /// <param name="expectedOutput">The expected output tensor (text embedding or paired text).</param>
    /// <remarks>
    /// <para>
    /// In native mode, full training with contrastive loss is supported using backpropagation
    /// through all layers. In ONNX mode, only projection layer fine-tuning is available.
    /// </para>
    /// <para><b>For Beginners:</b> CLIP training uses contrastive learning:
    /// - For each image-text pair, maximize similarity between matching pairs
    /// - Minimize similarity between non-matching pairs in the batch
    /// - This teaches the model to understand image-text relationships
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException(
                "Full training is not supported for CLIP with ONNX models. " +
                "Use native mode constructor for full training capability.");
        }

        SetTrainingMode(true);

        // Forward pass through vision encoder
        var imageEmbedding = GetImageEmbeddingNative(input);

        // The expectedOutput contains the text embedding or can be used to generate one
        var textEmbedding = expectedOutput.ToVector();

        // Normalize embeddings
        imageEmbedding = NormalizeVector(imageEmbedding);
        textEmbedding = NormalizeVector(textEmbedding);

        // Calculate contrastive loss
        var imageTensor = Tensor<T>.FromVector(imageEmbedding);
        var textTensor = Tensor<T>.FromVector(textEmbedding);

        LastLoss = LossFunction.CalculateLoss(imageTensor.ToVector(), textTensor.ToVector());

        // Calculate gradients
        var lossGradient = LossFunction.CalculateDerivative(imageTensor.ToVector(), textTensor.ToVector());

        // Backpropagate through vision encoder layers
        Backpropagate(Tensor<T>.FromVector(lossGradient));

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates the parameters of all layers in the network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters.</param>
    /// <remarks>
    /// <para>
    /// In native mode, this includes the CLS tokens and positional embeddings
    /// in addition to all layer parameters.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int expectedCount = ParameterCount;
        if (parameters.Length != expectedCount)
        {
            throw new ArgumentException(
                $"Expected {expectedCount} parameters, but got {parameters.Length}",
                nameof(parameters));
        }

        int index = 0;

        // In native mode, update CLS tokens and positional embeddings first
        if (_useNativeMode)
        {
            // Vision CLS token
            if (_visionClsToken is not null)
            {
                for (int i = 0; i < _visionClsToken.Length; i++)
                {
                    _visionClsToken[i] = parameters[index++];
                }
            }

            // Text CLS token
            if (_textClsToken is not null)
            {
                for (int i = 0; i < _textClsToken.Length; i++)
                {
                    _textClsToken[i] = parameters[index++];
                }
            }

            // Vision positional embeddings
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
        }

        // Update layer parameters
        foreach (var layer in Layers)
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
    /// Gets all parameters of the network as a single vector.
    /// </summary>
    /// <returns>A vector containing all network parameters.</returns>
    /// <remarks>
    /// <para>
    /// In native mode, this includes the CLS tokens and positional embeddings
    /// in addition to all layer parameters.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(ParameterCount);
        int index = 0;

        // In native mode, include CLS tokens and positional embeddings first
        if (_useNativeMode)
        {
            // Vision CLS token
            if (_visionClsToken is not null)
            {
                for (int i = 0; i < _visionClsToken.Length; i++)
                {
                    parameters[index++] = _visionClsToken[i];
                }
            }

            // Text CLS token
            if (_textClsToken is not null)
            {
                for (int i = 0; i < _textClsToken.Length; i++)
                {
                    parameters[index++] = _textClsToken[i];
                }
            }

            // Vision positional embeddings
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
        }

        // Get layer parameters
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                parameters[index++] = layerParams[i];
            }
        }

        return parameters;
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the network.
    /// </summary>
    /// <returns>The total parameter count.</returns>
    /// <remarks>
    /// <para>
    /// In native mode, this includes the CLS tokens and positional embeddings
    /// in addition to all layer parameters.
    /// </para>
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            int count = 0;

            // In native mode, count CLS tokens and positional embeddings
            if (_useNativeMode)
            {
                if (_visionClsToken is not null)
                    count += _visionClsToken.Length;

                if (_textClsToken is not null)
                    count += _textClsToken.Length;

                if (_visionPositionalEmbeddings is not null)
                    count += _visionPositionalEmbeddings.Rows * _visionPositionalEmbeddings.Columns;
            }

            // Add layer parameters
            foreach (var layer in Layers)
            {
                count += layer.ParameterCount;
            }

            return count;
        }
    }

    /// <summary>
    /// Retrieves metadata about the CLIP neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.Clip,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "EmbeddingDimension", _embeddingDimension },
                { "MaxSequenceLength", _maxSequenceLength },
                { "ImageSize", _imageSize },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() },
                { "TaskType", Architecture.TaskType.ToString() },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Indicates whether this network supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In native mode, full training with contrastive loss is supported.
    /// With ONNX models, only projection layer fine-tuning is available.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Serializes CLIP-specific data to a binary writer.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embeddingDimension);
        writer.Write(_maxSequenceLength);
        writer.Write(_imageSize);
        writer.Write(_optimizer.GetType().FullName ?? "AdamOptimizer");
        writer.Write(_lossFunction.GetType().FullName ?? "ContrastiveLoss");
    }

    /// <summary>
    /// Deserializes CLIP-specific data from a binary reader.
    /// </summary>
    /// <remarks>
    /// The readonly fields (_embeddingDimension, _maxSequenceLength, _imageSize) are set
    /// in the constructor and cannot be modified. This method validates that the deserialized
    /// values match the current instance configuration.
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int embeddingDim = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        int imgSize = reader.ReadInt32();

        // Read optimizer and loss function types to maintain binary format compatibility.
        // The stream must be read to advance past these values, even though they're not used
        // during deserialization (the current instance already has these configured).
        _ = reader.ReadString(); // optimizerType - read and discard
        _ = reader.ReadString(); // lossFunctionType - read and discard

        // Validate that loaded values match current instance
        if (embeddingDim != _embeddingDimension)
        {
            throw new InvalidOperationException(
                $"Loaded embedding dimension ({embeddingDim}) doesn't match current ({_embeddingDimension}).");
        }

        if (maxSeqLen != _maxSequenceLength)
        {
            throw new InvalidOperationException(
                $"Loaded max sequence length ({maxSeqLen}) doesn't match current ({_maxSequenceLength}).");
        }

        if (imgSize != _imageSize)
        {
            throw new InvalidOperationException(
                $"Loaded image size ({imgSize}) doesn't match current ({_imageSize}).");
        }
    }

    /// <summary>
    /// Creates a new instance of ClipNeuralNetwork with the same configuration.
    /// </summary>
    /// <returns>A new ClipNeuralNetwork instance with the same configuration.</returns>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new ClipNeuralNetwork<T>(
                Architecture,
                _imageSize,
                _channels,
                _patchSize,
                _vocabularySize,
                _maxSequenceLength,
                _embeddingDimension,
                _hiddenDim,
                _numLayers,
                _numHeads,
                _mlpDim,
                _tokenizer,
                _optimizer,
                _lossFunction
            );
        }

        // ONNX mode requires valid paths
        string imageEncoderPath = _imageEncoderPath ?? string.Empty;
        string textEncoderPath = _textEncoderPath ?? string.Empty;

        if (string.IsNullOrEmpty(imageEncoderPath) || string.IsNullOrEmpty(textEncoderPath))
        {
            throw new InvalidOperationException(
                "Cannot create new instance: ONNX encoder paths are not available.");
        }

        return new ClipNeuralNetwork<T>(
            Architecture,
            imageEncoderPath,
            textEncoderPath,
            _tokenizer,
            _optimizer,
            _lossFunction,
            _embeddingDimension,
            _maxSequenceLength,
            _imageSize
        );
    }

    /// <summary>
    /// Validates that the image tensor has the expected shape.
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
    /// Converts the internal Tensor to an ONNX DenseTensor.
    /// </summary>
    private OnnxTensors.DenseTensor<float> ConvertToOnnxTensor(Tensor<T> tensor)
    {
        var data = tensor.Data.Select(v => NumOps.ToFloat(v)).ToArray();

        // Add batch dimension if needed
        var shape = tensor.Shape.Length == 3
            ? new[] { 1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2] }
            : tensor.Shape;

        return new OnnxTensors.DenseTensor<float>(data, shape);
    }

    /// <summary>
    /// Extracts embeddings from a batched ONNX output tensor.
    /// </summary>
    /// <param name="onnxTensor">The batched output tensor.</param>
    /// <param name="batchSize">The number of items in the batch.</param>
    /// <returns>A collection of normalized embedding vectors.</returns>
    private IEnumerable<Vector<T>> ExtractBatchEmbeddings(OnnxTensors.Tensor<float> onnxTensor, int batchSize)
    {
        var dims = onnxTensor.Dimensions.ToArray();
        var embeddings = new List<Vector<T>>();

        if (dims.Length == 2)
        {
            // Shape: [batch, hidden]
            int hiddenDim = dims[1];
            for (int b = 0; b < batchSize; b++)
            {
                var result = new Vector<T>(_embeddingDimension);
                for (int i = 0; i < _embeddingDimension && i < hiddenDim; i++)
                {
                    result[i] = NumOps.FromDouble(onnxTensor[b, i]);
                }
                embeddings.Add(NormalizeVector(result));
            }
        }
        else if (dims.Length == 3)
        {
            // Shape: [batch, seq_len, hidden] - take CLS token (position 0) from each batch
            int hiddenDim = dims[2];
            for (int b = 0; b < batchSize; b++)
            {
                var result = new Vector<T>(_embeddingDimension);
                for (int i = 0; i < _embeddingDimension && i < hiddenDim; i++)
                {
                    result[i] = NumOps.FromDouble(onnxTensor[b, 0, i]);
                }
                embeddings.Add(NormalizeVector(result));
            }
        }
        else if (dims.Length == 1)
        {
            // Shape: [hidden] - single embedding, no batch dimension
            // This should only happen when batchSize == 1
            var result = new Vector<T>(_embeddingDimension);
            for (int i = 0; i < _embeddingDimension && i < dims[0]; i++)
            {
                result[i] = NumOps.FromDouble(onnxTensor[i]);
            }
            embeddings.Add(NormalizeVector(result));
        }
        else
        {
            // Unexpected tensor shape - throw to avoid silent incorrect results
            throw new InvalidOperationException(
                $"Unexpected ONNX output tensor shape: [{string.Join(", ", dims)}]. " +
                $"Expected 1D [hidden], 2D [batch, hidden], or 3D [batch, seq_len, hidden].");
        }

        return embeddings;
    }

    /// <summary>
    /// Converts an ONNX output tensor to a Vector.
    /// </summary>
    private Vector<T> ConvertToVector(OnnxTensors.Tensor<float> onnxTensor)
    {
        var result = new Vector<T>(_embeddingDimension);
        var dims = onnxTensor.Dimensions.ToArray();

        if (dims.Length == 2)
        {
            // Shape: [batch, hidden] - take first batch embedding
            int hiddenDim = dims[1];
            for (int i = 0; i < _embeddingDimension && i < hiddenDim; i++)
            {
                result[i] = NumOps.FromDouble(onnxTensor[0, i]);
            }
        }
        else if (dims.Length == 3)
        {
            // Shape: [batch, seq_len, hidden] - take CLS token (position 0) embedding
            int hiddenDim = dims[2];
            for (int i = 0; i < _embeddingDimension && i < hiddenDim; i++)
            {
                result[i] = NumOps.FromDouble(onnxTensor[0, 0, i]);
            }
        }
        else if (dims.Length == 1)
        {
            // Shape: [hidden] - flat embedding
            for (int i = 0; i < _embeddingDimension && i < dims[0]; i++)
            {
                result[i] = NumOps.FromDouble(onnxTensor.GetValue(i));
            }
        }
        else
        {
            // Fallback for unexpected shapes - use flat indexing
            for (int i = 0; i < _embeddingDimension && i < onnxTensor.Length; i++)
            {
                result[i] = NumOps.FromDouble(onnxTensor.GetValue(i));
            }
        }

        return result;
    }

    /// <summary>
    /// L2-normalizes a vector using the Engine.
    /// </summary>
    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        return vector.Normalize();
    }

    /// <summary>
    /// Disposes of the ONNX inference sessions.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _imageEncoder?.Dispose();
            _textEncoder?.Dispose();
        }
        base.Dispose(disposing);
    }
}
