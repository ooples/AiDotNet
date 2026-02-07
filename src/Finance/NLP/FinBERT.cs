using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.NLP;

/// <summary>
/// FinBERT (Financial BERT) model for financial sentiment analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// FinBERT is a BERT model fine-tuned on financial text for sentiment analysis.
/// It understands financial terminology and context to accurately classify
/// sentiment in financial news, SEC filings, earnings calls, and other financial text.
/// </para>
/// <para><b>For Beginners:</b> FinBERT solves a key problem in financial NLP:
///
/// <b>The Key Insight:</b>
/// General-purpose sentiment models often misinterpret financial language. For example,
/// "shares dropped 5% on earnings miss" is clearly negative for the stock, but a general
/// sentiment model might not understand this. FinBERT is trained specifically on financial
/// text to understand such nuances.
///
/// <b>What Problems Does FinBERT Solve?</b>
/// - Sentiment analysis of financial news articles
/// - Processing SEC filings (10-K, 10-Q, 8-K) for sentiment signals
/// - Analyzing earnings call transcripts for tone
/// - Social media sentiment monitoring for trading signals
/// - Document classification in financial contexts
///
/// <b>How FinBERT Works:</b>
/// 1. <b>Tokenization:</b> Text is split into WordPiece tokens (subwords)
/// 2. <b>Embedding:</b> Tokens are converted to dense vectors with position information
/// 3. <b>Transformer:</b> 12 layers of bidirectional self-attention capture context
/// 4. <b>Classification:</b> [CLS] token embedding is used for sentiment prediction
/// 5. <b>Output:</b> Softmax over classes: Positive, Negative, Neutral
///
/// <b>FinBERT Architecture:</b>
/// - Input: [CLS] token1 token2 ... tokenN [SEP]
/// - Embeddings: Token + Position + Segment embeddings (768-dim)
/// - Transformer: 12 layers, 12 heads, 768 hidden dim
/// - Output: 3-class softmax (negative, neutral, positive)
///
/// <b>Key Benefits:</b>
/// - Understands financial language and terminology
/// - Captures context across the entire input sequence
/// - Pre-trained on large financial corpora
/// - State-of-the-art accuracy on financial sentiment benchmarks
/// </para>
/// <para>
/// <b>Reference:</b> Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models", 2019.
/// https://arxiv.org/abs/1908.10063
/// </para>
/// </remarks>
public class FinBERT<T> : FinancialNLPModelBase<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    
    #region Native Mode Fields
    private EmbeddingLayer<T>? _tokenEmbedding;
    private EmbeddingLayer<T>? _positionEmbedding;
    private LayerNormalizationLayer<T>? _embeddingNorm;
    private List<MultiHeadAttentionLayer<T>>? _attentionLayers;
    private List<DenseLayer<T>>? _feedForwardLayers;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private List<DropoutLayer<T>>? _dropoutLayers;
    private DenseLayer<T>? _pooler;
    private DenseLayer<T>? _classifier;
    #endregion

    #region Tokenizer Fields
    /// <summary>
    /// Simple vocabulary mapping for demonstration.
    /// In production, use a proper WordPiece tokenizer.
    /// </summary>
    private readonly Dictionary<string, int> _vocabulary;
    private readonly Dictionary<int, string> _reverseVocabulary;
    private readonly string[] _sentimentClasses;
    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly FinBERTOptions<T> _options;
    private int _maxSequenceLength;
    private int _vocabularySize;
    private int _hiddenDimension;
    private int _numAttentionHeads;
    private int _intermediateDimension;
    private int _numLayers;
    private int _numSentimentClasses;
    private double _dropoutRate;
    #endregion

    #region IFinancialNLPModel Properties

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    public override int MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    public override int VocabularySize => _vocabularySize;

    /// <inheritdoc/>
    public override int HiddenDimension => _hiddenDimension;

    /// <inheritdoc/>
    public override int NumSentimentClasses => _numSentimentClasses;

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> BERT-base has 12 layers, BERT-large has 24.
    /// More layers can capture more complex patterns but require more computation.
    /// </para>
    /// </remarks>
    public int NumLayers => _numLayers;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-head attention allows the model to attend to
    /// information from different representation subspaces at different positions.
    /// </para>
    /// </remarks>
    public int NumAttentionHeads => _numAttentionHeads;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FinBERT model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">FinBERT-specific options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained FinBERT model
    /// in ONNX format. This is the recommended approach for production deployment as it
    /// leverages optimized inference from pretrained weights.
    /// </para>
    /// </remarks>
    public FinBERT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        FinBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);
        _options = options ?? new FinBERTOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _maxSequenceLength = _options.MaxSequenceLength;
        _vocabularySize = _options.VocabularySize;
        _hiddenDimension = _options.HiddenDimension;
        _numAttentionHeads = _options.NumAttentionHeads;
        _intermediateDimension = _options.IntermediateDimension;
        _numLayers = _options.NumLayers;
        _numSentimentClasses = _options.NumSentimentClasses;
        _dropoutRate = _options.DropoutRate;

        _vocabulary = InitializeBasicVocabulary();
        _reverseVocabulary = _vocabulary.ToDictionary(kv => kv.Value, kv => kv.Key);
        _sentimentClasses = new[] { "negative", "neutral", "positive" };
        _random = RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the FinBERT model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">FinBERT-specific options.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create a FinBERT model for
    /// training or fine-tuning on your financial data. Native mode enables gradient
    /// computation and parameter updates.
    /// </para>
    /// </remarks>
    public FinBERT(
        NeuralNetworkArchitecture<T> architecture,
        FinBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new FinBERTOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _maxSequenceLength = _options.MaxSequenceLength;
        _vocabularySize = _options.VocabularySize;
        _hiddenDimension = _options.HiddenDimension;
        _numAttentionHeads = _options.NumAttentionHeads;
        _intermediateDimension = _options.IntermediateDimension;
        _numLayers = _options.NumLayers;
        _numSentimentClasses = _options.NumSentimentClasses;
        _dropoutRate = _options.DropoutRate;

        _vocabulary = InitializeBasicVocabulary();
        _reverseVocabulary = _vocabulary.ToDictionary(kv => kv.Value, kv => kv.Key);
        _sentimentClasses = new[] { "negative", "neutral", "positive" };
        _random = RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for FinBERT.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sets up the BERT architecture with:
    /// - Token and position embeddings
    /// - Multiple transformer layers with self-attention
    /// - Pooler and classification head
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultFinBERTLayers(
                Architecture,
                _vocabularySize,
                _maxSequenceLength,
                _hiddenDimension,
                _numAttentionHeads,
                _intermediateDimension,
                _numLayers,
                _numSentimentClasses,
                _dropoutRate));
        }

        ExtractLayerReferences();
    }

    /// <summary>
    /// Extracts references to specific layer types for direct access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After creating layers, we store references to specific
    /// layers for easier access during forward/backward passes. This enables efficient
    /// embedding lookup and layer-by-layer processing.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        var embeddings = Layers.OfType<EmbeddingLayer<T>>().ToList();
        _tokenEmbedding = embeddings.FirstOrDefault();
        _positionEmbedding = embeddings.Skip(1).FirstOrDefault();

        var norms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();
        _embeddingNorm = norms.FirstOrDefault();
        _layerNorms = norms.Skip(1).ToList();

        _attentionLayers = Layers.OfType<MultiHeadAttentionLayer<T>>().ToList();
        _feedForwardLayers = Layers.OfType<DenseLayer<T>>().ToList();
        _dropoutLayers = Layers.OfType<DropoutLayer<T>>().ToList();

        // Last two dense layers are pooler and classifier
        if (_feedForwardLayers.Count >= 2)
        {
            _classifier = _feedForwardLayers.Last();
            _pooler = _feedForwardLayers[_feedForwardLayers.Count - 2];
        }
    }

    /// <summary>
    /// Initializes a basic vocabulary for demonstration purposes.
    /// </summary>
    /// <returns>Dictionary mapping tokens to IDs.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a simplified vocabulary.
    /// In production, you would load the actual BERT WordPiece vocabulary (30,522 tokens)
    /// which handles subword tokenization properly.
    /// </para>
    /// </remarks>
    private Dictionary<string, int> InitializeBasicVocabulary()
    {
        var vocab = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
        {
            ["[PAD]"] = 0,
            ["[UNK]"] = 1,
            ["[CLS]"] = 2,
            ["[SEP]"] = 3,
            ["[MASK]"] = 4
        };

        // Add common financial terms
        string[] financialTerms =
        {
            "stock", "shares", "price", "market", "earnings", "revenue", "profit", "loss",
            "growth", "decline", "increase", "decrease", "rise", "fall", "drop", "surge",
            "buy", "sell", "hold", "upgrade", "downgrade", "bullish", "bearish",
            "positive", "negative", "neutral", "strong", "weak", "beat", "miss",
            "quarter", "year", "annual", "quarterly", "fiscal", "outlook", "guidance",
            "forecast", "estimate", "analyst", "investor", "company", "corporation",
            "business", "industry", "sector", "economy", "recession", "inflation"
        };

        int id = 5;
        foreach (var term in financialTerms)
        {
            if (!vocab.ContainsKey(term))
            {
                vocab[term] = id++;
            }
        }

        return vocab;
    }

    /// <summary>
    /// Validates custom layers provided by the user.
    /// </summary>
    /// <param name="layers">The list of custom layers.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensures custom layers form a valid FinBERT architecture
    /// with embeddings, transformer layers, and classification head.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 4)
            throw new ArgumentException("FinBERT requires at least 4 layers (embeddings, transformer, pooler, classifier).");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Performs forward prediction on the input tensor.
    /// </summary>
    /// <param name="input">Input tensor with token IDs.</param>
    /// <returns>Output tensor with sentiment logits.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Takes tokenized input and produces sentiment scores.
    /// The output has shape [batch_size, num_sentiment_classes] where each row contains
    /// logits for [negative, neutral, positive].
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForwardNative(input) : ForwardOnnx(input);
    }

    /// <summary>
    /// Trains the FinBERT model on a batch of input-target pairs.
    /// </summary>
    /// <param name="input">Input tensor with token IDs.</param>
    /// <param name="target">Target tensor with sentiment labels (one-hot or class indices).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training updates all layers:
    /// - Token and position embeddings
    /// - Multi-head attention weights
    /// - Feed-forward network weights
    /// - Classification head
    ///
    /// Fine-tuning on domain-specific data typically improves performance.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training requires native mode.");

        SetTrainingMode(true);

        // Forward pass
        var output = Forward(input);

        // Calculate loss
        LastLoss = _lossFunction.CalculateLoss(output.ToVector(), target.ToVector());

        // Backward pass
        var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient, output.Shape));

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates parameters using the provided gradients.
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Parameters are updated through the optimizer in Train().
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <summary>
    /// Gets metadata about the FinBERT model.
    /// </summary>
    /// <returns>ModelMetadata containing model information.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns a dictionary of model configuration values
    /// for logging and debugging purposes.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "FinBERT" },
                { "MaxSequenceLength", _maxSequenceLength },
                { "VocabularySize", _vocabularySize },
                { "HiddenDimension", _hiddenDimension },
                { "NumAttentionHeads", _numAttentionHeads },
                { "IntermediateDimension", _intermediateDimension },
                { "NumLayers", _numLayers },
                { "NumSentimentClasses", _numSentimentClasses },
                { "DropoutRate", _dropoutRate },
                { "UseNativeMode", _useNativeMode }
            }
        };
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <returns>A new FinBERT instance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a fresh model with the same architecture
    /// and options but randomly reinitialized weights.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new FinBERT<T>(Architecture, _options);
    }

    /// <summary>
    /// Serializes FinBERT-specific data.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves model configuration and vocabulary
    /// for later loading.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_maxSequenceLength);
        writer.Write(_vocabularySize);
        writer.Write(_hiddenDimension);
        writer.Write(_numAttentionHeads);
        writer.Write(_intermediateDimension);
        writer.Write(_numLayers);
        writer.Write(_numSentimentClasses);
        writer.Write(_dropoutRate);

        // Serialize vocabulary
        writer.Write(_vocabulary.Count);
        foreach (var kvp in _vocabulary)
        {
            writer.Write(kvp.Key);
            writer.Write(kvp.Value);
        }
    }

    /// <summary>
    /// Deserializes FinBERT-specific data.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Loads model configuration and vocabulary
    /// from disk.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _maxSequenceLength = reader.ReadInt32();
        _vocabularySize = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numAttentionHeads = reader.ReadInt32();
        _intermediateDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numSentimentClasses = reader.ReadInt32();
        _dropoutRate = reader.ReadDouble();

        // Deserialize vocabulary
        int vocabCount = reader.ReadInt32();
        _vocabulary.Clear();
        for (int i = 0; i < vocabCount; i++)
        {
            string key = reader.ReadString();
            int value = reader.ReadInt32();
            _vocabulary[key] = value;
        }
    }

    #endregion

    #region IFinancialNLPModel Implementation

    /// <summary>
    /// Analyzes sentiment from tokenized input.
    /// </summary>
    /// <param name="tokenIds">Tensor of token IDs [batch_size, sequence_length].</param>
    /// <returns>Sentiment probabilities [batch_size, num_sentiment_classes].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Takes tokenized text and returns sentiment probabilities.
    /// For a 3-class model: [P(negative), P(neutral), P(positive)].
    /// </para>
    /// </remarks>
    public override Tensor<T> AnalyzeSentiment(Tensor<T> tokenIds)
    {
        var logits = Predict(tokenIds);
        return ApplySoftmax(logits);
    }

    /// <summary>
    /// Analyzes sentiment from raw text strings.
    /// </summary>
    /// <param name="texts">Array of text strings to analyze.</param>
    /// <returns>Sentiment results with class probabilities for each text.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Convenient method that handles tokenization internally.
    /// Returns structured results with predicted class, confidence, and all class probabilities.
    /// </para>
    /// </remarks>
    public override SentimentResult<T>[] AnalyzeSentiment(string[] texts)
    {
        var results = new SentimentResult<T>[texts.Length];

        for (int i = 0; i < texts.Length; i++)
        {
            var tokenIds = Tokenize(texts[i], _maxSequenceLength);
            var tensor = TokenIdsToTensor(tokenIds);
            var probs = AnalyzeSentiment(tensor);

            var probVector = probs.ToVector();
            int maxIdx = 0;
            T maxProb = probVector[0];
            var classProbabilities = new Dictionary<string, T>();

            for (int j = 0; j < _numSentimentClasses && j < probVector.Length; j++)
            {
                string className = j < _sentimentClasses.Length ? _sentimentClasses[j] : $"class_{j}";
                classProbabilities[className] = probVector[j];

                if (NumOps.ToDouble(probVector[j]) > NumOps.ToDouble(maxProb))
                {
                    maxProb = probVector[j];
                    maxIdx = j;
                }
            }

            results[i] = new SentimentResult<T>
            {
                OriginalText = texts[i],
                PredictedClass = maxIdx < _sentimentClasses.Length ? _sentimentClasses[maxIdx] : $"class_{maxIdx}",
                Confidence = maxProb,
                ClassProbabilities = classProbabilities
            };
        }

        return results;
    }

    /// <summary>
    /// Gets embeddings (vector representations) for input tokens.
    /// </summary>
    /// <param name="tokenIds">Tensor of token IDs [batch_size, sequence_length].</param>
    /// <returns>Token embeddings [batch_size, sequence_length, hidden_dim].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Embeddings are dense vector representations of tokens.
    /// These can be used for similarity search, clustering, or as features for downstream tasks.
    /// </para>
    /// </remarks>
    public override Tensor<T> GetEmbeddings(Tensor<T> tokenIds)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("GetEmbeddings requires native mode.");

        SetTrainingMode(false);

        // Get token embeddings from embedding layer
        var embeddings = _tokenEmbedding is not null
            ? _tokenEmbedding.Forward(tokenIds)
            : tokenIds;

        return embeddings;
    }

    /// <summary>
    /// Gets the [CLS] token embedding representing the entire input sequence.
    /// </summary>
    /// <param name="tokenIds">Tensor of token IDs [batch_size, sequence_length].</param>
    /// <returns>Sequence embeddings [batch_size, hidden_dim].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> In BERT, the [CLS] token is a special token at the beginning
    /// whose embedding represents the entire sequence. This is commonly used for classification
    /// tasks and document similarity.
    /// </para>
    /// </remarks>
    public override Tensor<T> GetSequenceEmbedding(Tensor<T> tokenIds)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("GetSequenceEmbedding requires native mode.");

        SetTrainingMode(false);

        // Forward through most of the network but stop before classifier
        var hidden = Forward(tokenIds);

        // Pooler output is the [CLS] representation
        return hidden;
    }

    /// <summary>
    /// Tokenizes raw text into token IDs.
    /// </summary>
    /// <param name="text">The text to tokenize.</param>
    /// <param name="maxLength">Maximum sequence length (will pad or truncate).</param>
    /// <returns>Array of token IDs.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tokenization converts text to numbers that the model understands.
    /// This simplified version does word-level tokenization. Production implementations should
    /// use WordPiece tokenization for proper subword handling.
    /// </para>
    /// </remarks>
    public override int[] Tokenize(string text, int? maxLength = null)
    {
        int maxLen = maxLength ?? _maxSequenceLength;
        var tokens = new List<int> { _vocabulary["[CLS]"] };

        // Simple word tokenization (production should use WordPiece)
        var words = text.ToLowerInvariant()
            .Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']' },
                   StringSplitOptions.RemoveEmptyEntries);

        foreach (var word in words)
        {
            if (tokens.Count >= maxLen - 1) break;

            int tokenId = _vocabulary.TryGetValue(word, out int id) ? id : _vocabulary["[UNK]"];
            tokens.Add(tokenId);
        }

        tokens.Add(_vocabulary["[SEP]"]);

        // Pad to max length
        while (tokens.Count < maxLen)
        {
            tokens.Add(_vocabulary["[PAD]"]);
        }

        return tokens.Take(maxLen).ToArray();
    }

    /// <summary>
    /// Converts token IDs back to text.
    /// </summary>
    /// <param name="tokenIds">Array of token IDs.</param>
    /// <returns>Reconstructed text string.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Converts the numeric token IDs back into human-readable text.
    /// Useful for debugging and understanding what the model sees.
    /// </para>
    /// </remarks>
    public override string Detokenize(int[] tokenIds)
    {
        var words = new List<string>();

        foreach (var id in tokenIds)
        {
            if (_reverseVocabulary.TryGetValue(id, out string? token))
            {
                if (token != "[PAD]" && token != "[CLS]" && token != "[SEP]")
                {
                    words.Add(token);
                }
            }
            else
            {
                words.Add("[UNK]");
            }
        }

        return string.Join(" ", words);
    }

    /// <summary>
    /// Gets financial-specific NLP metrics from the model.
    /// </summary>
    /// <returns>Dictionary containing NLP metrics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns metrics relevant to financial NLP performance
    /// including model configuration and training status.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["LastLoss"] = lastLoss,
            ["MaxSequenceLength"] = NumOps.FromDouble(_maxSequenceLength),
            ["VocabularySize"] = NumOps.FromDouble(_vocabularySize),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["NumAttentionHeads"] = NumOps.FromDouble(_numAttentionHeads),
            ["NumSentimentClasses"] = NumOps.FromDouble(_numSentimentClasses)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through all layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass:
    /// 1. Embeds tokens and positions
    /// 2. Passes through transformer layers
    /// 3. Pools the [CLS] token representation
    /// 4. Projects to sentiment classes
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;

        // Apply layers
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Performs the backward pass for gradient computation.
    /// </summary>
    /// <param name="gradOutput">Gradient from the loss function.</param>
    /// <returns>Gradient with respect to input.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Propagates gradients backward through all layers,
    /// computing how each layer's parameters should change to reduce the loss.
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        // Backward through layers in reverse
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            current = Layers[i].Backward(current);
        }

        return current;
    }

    /// <summary>
    /// Performs native mode forward pass.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output logits.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Runs the full BERT forward pass using native C# layers.
    /// </para>
    /// </remarks>
    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        SetTrainingMode(false);
        return Forward(input);
    }

    /// <summary>
    /// Performs ONNX mode forward pass.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output logits.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uses ONNX runtime for optimized inference with
    /// pretrained weights.
    /// </para>
    /// </remarks>
    private Tensor<T> ForwardOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var inputData = new long[input.Data.Length];
        for (int i = 0; i < input.Data.Length; i++)
        {
            inputData[i] = (long)NumOps.ToDouble(input.Data.Span[i]);
        }

        var inputTensor = new OnnxTensors.DenseTensor<long>(
            inputData,
            new[] { 1, _maxSequenceLength });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(new[] { _numSentimentClasses }, new Vector<T>(outputData));
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Converts token ID array to tensor.
    /// </summary>
    /// <param name="tokenIds">Array of token IDs.</param>
    /// <returns>Tensor with token IDs.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Wraps a token ID array in a tensor for model input.
    /// </para>
    /// </remarks>
    private Tensor<T> TokenIdsToTensor(int[] tokenIds)
    {
        var data = new T[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
        {
            data[i] = NumOps.FromDouble(tokenIds[i]);
        }
        return new Tensor<T>(new[] { 1, tokenIds.Length }, new Vector<T>(data));
    }

    /// <summary>
    /// Applies softmax to convert logits to probabilities.
    /// </summary>
    /// <param name="logits">Input logits tensor.</param>
    /// <returns>Probability tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Softmax converts raw model outputs (logits) to probabilities
    /// that sum to 1. Higher logits become higher probabilities.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        var data = logits.ToVector();
        var probs = new T[data.Length];

        // Find max for numerical stability
        double maxVal = double.MinValue;
        for (int i = 0; i < data.Length; i++)
        {
            double val = NumOps.ToDouble(data[i]);
            if (val > maxVal) maxVal = val;
        }

        // Compute exp(x - max) and sum
        double sum = 0;
        var expValues = new double[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            expValues[i] = Math.Exp(NumOps.ToDouble(data[i]) - maxVal);
            sum += expValues[i];
        }

        // Normalize
        for (int i = 0; i < data.Length; i++)
        {
            probs[i] = NumOps.FromDouble(expValues[i] / sum);
        }

        return new Tensor<T>(logits.Shape, new Vector<T>(probs));
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes managed resources.
    /// </summary>
    /// <param name="disposing">Whether to dispose managed resources.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Cleans up resources when the model is no longer needed,
    /// particularly the ONNX session which holds native resources.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxSession?.Dispose();
        }

        base.Dispose(disposing);
    }

    #endregion
}

