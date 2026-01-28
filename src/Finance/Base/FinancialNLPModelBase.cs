using AiDotNet.Finance.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.LossFunctions;
using System.IO;

namespace AiDotNet.Finance.Base;

/// <summary>
/// Base class for all financial NLP models, implementing the dual-mode pattern.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class FinancialNLPModelBase<T> : FinancialModelBase<T>, IFinancialNLPModel<T>
{
    protected readonly int _maxSequenceLength;
    protected readonly int _vocabularySize;
    protected readonly int _hiddenDimension;
    protected readonly int _numSentimentClasses;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The maximum number of tokens the model can read at once.
    /// Longer texts are truncated or split to fit this length.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The size of the vocabulary (how many unique tokens the model knows).
    /// Larger vocabularies allow more precise language understanding but require more memory.
    /// </para>
    /// </remarks>
    public int VocabularySize => _vocabularySize;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The embedding size used to represent each token.
    /// Think of this as the number of features describing each word.
    /// </para>
    /// </remarks>
    public int HiddenDimension => _hiddenDimension;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many sentiment categories the model can output
    /// (e.g., negative/neutral/positive).
    /// </para>
    /// </remarks>
    public int NumSentimentClasses => _numSentimentClasses;

    /// <summary>
    /// Initializes a new NLP model base for training (native mode).
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="maxSequenceLength">Maximum token sequence length.</param>
    /// <param name="vocabularySize">Vocabulary size.</param>
    /// <param name="hiddenDimension">Embedding dimension.</param>
    /// <param name="numSentimentClasses">Number of sentiment classes.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up a trainable NLP model.
    /// It defines how long text inputs can be, how big the vocabulary is,
    /// and how large the internal embeddings are.
    /// </para>
    /// </remarks>
    protected FinancialNLPModelBase(
        NeuralNetworkArchitecture<T> architecture,
        int maxSequenceLength,
        int vocabularySize,
        int hiddenDimension,
        int numSentimentClasses = 3,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, maxSequenceLength, 1, architecture.InputSize, lossFunction)
    {
        _maxSequenceLength = maxSequenceLength;
        _vocabularySize = vocabularySize;
        _hiddenDimension = hiddenDimension;
        _numSentimentClasses = numSentimentClasses;
    }

    /// <summary>
    /// Initializes a new NLP model base from a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="maxSequenceLength">Maximum token sequence length.</param>
    /// <param name="vocabularySize">Vocabulary size.</param>
    /// <param name="hiddenDimension">Embedding dimension.</param>
    /// <param name="numSentimentClasses">Number of sentiment classes.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you have a pretrained language model
    /// and want fast inference without training.
    /// </para>
    /// </remarks>
    protected FinancialNLPModelBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int maxSequenceLength,
        int vocabularySize,
        int hiddenDimension,
        int numSentimentClasses = 3)
        : base(architecture, onnxModelPath, maxSequenceLength, 1, architecture.InputSize)
    {
        _maxSequenceLength = maxSequenceLength;
        _vocabularySize = vocabularySize;
        _hiddenDimension = hiddenDimension;
        _numSentimentClasses = numSentimentClasses;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the simplest sentiment interface:
    /// provide token IDs and the model returns a sentiment prediction.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> AnalyzeSentiment(Tensor<T> tokenIds) => Predict(tokenIds);

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper accepts raw text strings, tokenizes them,
    /// runs the model, and returns a readable sentiment result for each input.
    /// </para>
    /// </remarks>
    public virtual SentimentResult<T>[] AnalyzeSentiment(string[] texts)
    {
        var results = new SentimentResult<T>[texts.Length];
        for (int i = 0; i < texts.Length; i++)
        {
            var tokenIds = Tokenize(texts[i]);
            var prediction = AnalyzeSentiment(Tensor<T>.FromVector(new Vector<T>(tokenIds.Select(id => NumOps.FromDouble(id)).ToArray())));
            
            results[i] = new SentimentResult<T>
            {
                OriginalText = texts[i],
                PredictedClass = "neutral", // Simplified placeholder
                Confidence = prediction.Data.Span[0], // Simplified
                ClassProbabilities = new Dictionary<string, T> { ["neutral"] = prediction.Data.Span[0] }
            };
        }

        return results;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Embeddings are dense numeric representations of text
    /// that capture meaning. This method returns those embeddings for tokens.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> GetEmbeddings(Tensor<T> tokenIds) => Predict(tokenIds);

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This summarizes an entire sequence into a single embedding,
    /// which can be used for classification or similarity search.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> GetSequenceEmbedding(Tensor<T> tokenIds) => Predict(tokenIds);

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tokenization turns text into integers the model can read.
    /// This default implementation is a placeholder and should be overridden by real tokenizers.
    /// </para>
    /// </remarks>
    public virtual int[] Tokenize(string text, int? maxLength = null) => new int[maxLength ?? _maxSequenceLength];

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Detokenization converts token IDs back into text.
    /// This default implementation returns an empty string as a placeholder.
    /// </para>
    /// </remarks>
    public virtual string Detokenize(int[] tokenIds) => string.Empty;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adds NLP-specific configuration to the standard finance metrics,
    /// such as vocabulary size and maximum sequence length.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        var metrics = base.GetFinancialMetrics();
        metrics["MaxSequenceLength"] = NumOps.FromDouble(_maxSequenceLength);
        metrics["VocabularySize"] = NumOps.FromDouble(_vocabularySize);
        metrics["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension);
        
        return metrics;
    }
}
