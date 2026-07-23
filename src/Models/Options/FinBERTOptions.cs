using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for FinBERT (Financial BERT) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// FinBERT is a BERT model fine-tuned on financial text for sentiment analysis
/// and understanding of financial language.
/// </para>
/// <para><b>For Beginners:</b> FinBERT is designed specifically for financial text:
///
/// <b>The Key Insight:</b>
/// General-purpose sentiment models often misinterpret financial language. For example,
/// "shares fell 3%" is negative for the stock, but a general sentiment model might not
/// understand this. FinBERT is trained on financial text to understand such nuances.
///
/// <b>What Problems Does FinBERT Solve?</b>
/// - Sentiment analysis of financial news articles
/// - Analyzing SEC filings (10-K, 10-Q, 8-K)
/// - Processing earnings call transcripts
/// - Social media sentiment for stock prediction
/// - Document classification in financial contexts
///
/// <b>How FinBERT Works:</b>
/// 1. <b>Pre-training:</b> BERT architecture trained on large text corpus
/// 2. <b>Fine-tuning:</b> Further trained on financial-specific text and labels
/// 3. <b>Tokenization:</b> Text is split into WordPiece tokens (subwords)
/// 4. <b>Embedding:</b> Tokens are converted to dense vectors
/// 5. <b>Transformer:</b> Self-attention captures context across the sequence
/// 6. <b>Classification:</b> [CLS] token is used for sentiment prediction
///
/// <b>FinBERT Architecture:</b>
/// - Input: [CLS] token1 token2 ... tokenN [SEP]
/// - Embeddings: Token + Position + Segment embeddings
/// - Transformer: 12 layers of multi-head self-attention
/// - Output: Softmax over sentiment classes
///
/// <b>Key Benefits:</b>
/// - Understands financial language and terminology
/// - Captures context and nuance in financial text
/// - Pre-trained on large financial corpora
/// - State-of-the-art for financial sentiment analysis
/// </para>
/// <para>
/// <b>Reference:</b> Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models", 2019.
/// https://arxiv.org/abs/1908.10063
/// </para>
/// </remarks>
public class FinBERTOptions<T> : ModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FinBERTOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default FinBERT configuration using BERT-base
    /// architecture with 3-class sentiment classification (positive, negative, neutral).
    /// </para>
    /// </remarks>
    public FinBERTOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FinBERTOptions(FinBERTOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        MaxSequenceLength = other.MaxSequenceLength;
        VocabularySize = other.VocabularySize;
        HiddenDimension = other.HiddenDimension;
        NumAttentionHeads = other.NumAttentionHeads;
        IntermediateDimension = other.IntermediateDimension;
        NumLayers = other.NumLayers;
        NumSentimentClasses = other.NumSentimentClasses;
        DropoutRate = other.DropoutRate;
        AttentionDropoutRate = other.AttentionDropoutRate;
        HiddenDropoutRate = other.HiddenDropoutRate;
        TypeVocabSize = other.TypeVocabSize;
        MaxPositionEmbeddings = other.MaxPositionEmbeddings;
        UsePretrainedWeights = other.UsePretrainedWeights;
        PretrainedModelPath = other.PretrainedModelPath;
        FreezeBaseModel = other.FreezeBaseModel;
        NumFineTuneLayers = other.NumFineTuneLayers;
    }

    /// <summary>
    /// Gets or sets the maximum sequence length in tokens.
    /// </summary>
    /// <value>The maximum sequence length, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The maximum number of tokens (roughly words/subwords)
    /// the model can process. BERT supports up to 512 tokens. Longer texts need to be
    /// chunked or truncated.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the vocabulary size.
    /// </summary>
    /// <value>The vocabulary size, defaulting to 30522 (BERT-base).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The number of unique tokens in the tokenizer's vocabulary.
    /// BERT uses WordPiece tokenization with approximately 30,000 tokens.
    /// </para>
    /// </remarks>
    public int VocabularySize { get; set; } = 30522;

    /// <summary>
    /// Gets or sets the hidden dimension of transformer layers.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 768 (BERT-base).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the hidden representations in each layer.
    /// BERT-base uses 768, BERT-large uses 1024. Larger dimensions capture more information
    /// but require more memory and computation.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>The number of attention heads, defaulting to 12 (BERT-base).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-head attention lets the model attend to information
    /// from different representation subspaces at different positions. BERT-base uses 12 heads,
    /// each with dimension 768/12 = 64.
    /// </para>
    /// </remarks>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>
    /// Gets or sets the intermediate (feed-forward) dimension.
    /// </summary>
    /// <value>The intermediate dimension, defaulting to 3072.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The hidden dimension of the feed-forward network in each
    /// transformer layer. Typically 4x the hidden dimension (768 * 4 = 3072).
    /// </para>
    /// </remarks>
    public int IntermediateDimension { get; set; } = 3072;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 12 (BERT-base).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many transformer blocks are stacked. BERT-base has 12 layers,
    /// BERT-large has 24 layers. More layers can capture more complex patterns but require
    /// more computation.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of sentiment classes.
    /// </summary>
    /// <value>The number of sentiment classes, defaulting to 3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> FinBERT typically uses 3 classes:
    /// - Negative (bearish sentiment, bad news)
    /// - Neutral (factual, no sentiment)
    /// - Positive (bullish sentiment, good news)
    ///
    /// Some variants use 5 classes including "very positive" and "very negative".
    /// </para>
    /// </remarks>
    public int NumSentimentClasses { get; set; } = 3;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Randomly drops connections during training to prevent overfitting.
    /// BERT uses 0.1 dropout throughout the model.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the attention-specific dropout rate.
    /// </summary>
    /// <value>The attention dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout applied to attention weights.
    /// Helps prevent the model from over-relying on specific attention patterns.
    /// </para>
    /// </remarks>
    public double AttentionDropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the hidden layer dropout rate.
    /// </summary>
    /// <value>The hidden dropout rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Dropout applied to hidden layer outputs.
    /// </para>
    /// </remarks>
    public double HiddenDropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the type vocabulary size (for sentence pairs).
    /// </summary>
    /// <value>The type vocabulary size, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> BERT can process sentence pairs with segment embeddings.
    /// Type vocab size of 2 means there are two segment types (sentence A and B).
    /// For single-sentence classification, only segment 0 is used.
    /// </para>
    /// </remarks>
    public int TypeVocabSize { get; set; } = 2;

    /// <summary>
    /// Gets or sets the maximum position embeddings.
    /// </summary>
    /// <value>The maximum positions, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Positional embeddings encode where each token appears
    /// in the sequence. This must be at least MaxSequenceLength.
    /// </para>
    /// </remarks>
    public int MaxPositionEmbeddings { get; set; } = 512;

    /// <summary>
    /// Gets or sets whether to use pretrained weights.
    /// </summary>
    /// <value>True to use pretrained weights; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Pretrained weights from large-scale pretraining significantly
    /// improve performance. Set to false only if you want to train from scratch (not recommended).
    /// </para>
    /// </remarks>
    public bool UsePretrainedWeights { get; set; } = true;

    /// <summary>
    /// Gets or sets the path to pretrained model weights.
    /// </summary>
    /// <value>The pretrained model path, defaulting to empty string.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Path to ONNX file or checkpoint with pretrained FinBERT weights.
    /// Leave empty to train from scratch or use built-in initialization.
    /// </para>
    /// </remarks>
    public string PretrainedModelPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether to freeze the base model during fine-tuning.
    /// </summary>
    /// <value>True to freeze base model; false otherwise. Default: false.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Freezing prevents the base BERT weights from updating
    /// during training, only training the classification head. This is faster but may
    /// result in lower performance than fine-tuning all layers.
    /// </para>
    /// </remarks>
    public bool FreezeBaseModel { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of layers to fine-tune (from the top).
    /// </summary>
    /// <value>The number of layers to fine-tune, defaulting to -1 (all layers).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of fine-tuning all layers or freezing all,
    /// you can fine-tune only the top N layers. -1 means fine-tune all layers.
    /// Setting to 2 means only the last 2 transformer layers plus the classification head
    /// will be updated during training.
    /// </para>
    /// </remarks>
    public int NumFineTuneLayers { get; set; } = -1;
}
