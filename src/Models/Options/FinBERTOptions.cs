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
public class FinBERTOptions<T> : FinancialBertEncoderOptions<T>
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
        : base(other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        MaxSequenceLength = other.MaxSequenceLength;
        VocabularySize = other.VocabularySize;
        HiddenDimension = other.HiddenDimension;
        NumAttentionHeads = other.NumAttentionHeads;
        IntermediateDimension = other.IntermediateDimension;
        NumLayers = other.NumLayers;
        NumSentimentClasses = other.NumSentimentClasses;
        DropoutRate = other.DropoutRate;
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
    /// Validates the FinBERT options.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when a setting is out of range or the settings contradict each
    /// other.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This checks the sizes and rates before the model is built, so a typo fails here
    /// with a clear message instead of deep inside a layer.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        ValidateBertEncoderOptions(MaxSequenceLength);
        if (MaxSequenceLength < 1)
            throw new ArgumentException("MaxSequenceLength must be at least 1.", nameof(MaxSequenceLength));
        if (VocabularySize < 1)
            throw new ArgumentException("VocabularySize must be at least 1.", nameof(VocabularySize));
        if (HiddenDimension < 1)
            throw new ArgumentException("HiddenDimension must be at least 1.", nameof(HiddenDimension));
        if (NumAttentionHeads < 1)
            throw new ArgumentException("NumAttentionHeads must be at least 1.", nameof(NumAttentionHeads));
        if (HiddenDimension % NumAttentionHeads != 0)
            throw new ArgumentException("HiddenDimension must be divisible by NumAttentionHeads.", nameof(NumAttentionHeads));
        if (IntermediateDimension < 1)
            throw new ArgumentException("IntermediateDimension must be at least 1.", nameof(IntermediateDimension));
        if (NumLayers < 1)
            throw new ArgumentException("NumLayers must be at least 1.", nameof(NumLayers));
        if (NumSentimentClasses < 1)
            throw new ArgumentException("NumSentimentClasses must be at least 1.", nameof(NumSentimentClasses));
        if (DropoutRate < 0 || DropoutRate >= 1)
            throw new ArgumentException("DropoutRate must be between 0 (inclusive) and 1 (exclusive).", nameof(DropoutRate));
    }
}
