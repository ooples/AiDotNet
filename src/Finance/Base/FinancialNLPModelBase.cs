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
    protected readonly int _baseMaxSequenceLength;
    protected readonly int _baseVocabularySize;
    protected readonly int _baseHiddenDimension;
    protected readonly int _baseNumSentimentClasses;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The maximum number of tokens the model can read at once.
    /// Longer texts are truncated or split to fit this length.
    /// </para>
    /// </remarks>
    public virtual int MaxSequenceLength => _baseMaxSequenceLength;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The size of the vocabulary (how many unique tokens the model knows).
    /// Larger vocabularies allow more precise language understanding but require more memory.
    /// </para>
    /// </remarks>
    public virtual int VocabularySize => _baseVocabularySize;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The embedding size used to represent each token.
    /// Think of this as the number of features describing each word.
    /// </para>
    /// </remarks>
    public virtual int HiddenDimension => _baseHiddenDimension;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many sentiment categories the model can output
    /// (e.g., negative/neutral/positive).
    /// </para>
    /// </remarks>
    public virtual int NumSentimentClasses => _baseNumSentimentClasses;

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
        _baseMaxSequenceLength = maxSequenceLength;
        _baseVocabularySize = vocabularySize;
        _baseHiddenDimension = hiddenDimension;
        _baseNumSentimentClasses = numSentimentClasses;
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
        _baseMaxSequenceLength = maxSequenceLength;
        _baseVocabularySize = vocabularySize;
        _baseHiddenDimension = hiddenDimension;
        _baseNumSentimentClasses = numSentimentClasses;
    }

    /// <summary>
    /// Initializes a new instance with deferred NLP configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor keeps the legacy pattern where
    /// derived NLP models configure vocabulary size and sequence length later.
    /// </para>
    /// </remarks>
    protected FinancialNLPModelBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction, maxGradNorm)
    {
        _baseMaxSequenceLength = 0;
        _baseVocabularySize = 0;
        _baseHiddenDimension = 0;
        _baseNumSentimentClasses = 0;
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

            // Guard against empty tokenIds - if Tokenize returns empty, short-circuit
            // with an "unknown" result instead of creating a [1,0] Tensor
            if (tokenIds.Length == 0)
            {
                results[i] = new SentimentResult<T>
                {
                    OriginalText = texts[i],
                    PredictedClass = "unknown",
                    Confidence = NumOps.Zero,
                    ClassProbabilities = new Dictionary<string, T>()
                };
                continue;
            }

            // Create a 2D tensor [1, sequence_length] since NLP models expect batch dimension
            var tokenVector = new Vector<T>(tokenIds.Select(id => NumOps.FromDouble(id)).ToArray());
            var inputTensor = new Tensor<T>(new[] { 1, tokenIds.Length }, tokenVector);
            var prediction = AnalyzeSentiment(inputTensor);

            if (prediction.Length == 0)
            {
                results[i] = new SentimentResult<T>
                {
                    OriginalText = texts[i],
                    PredictedClass = "unknown",
                    Confidence = NumOps.Zero,
                    ClassProbabilities = new Dictionary<string, T>()
                };
                continue;
            }

            int classCount = Math.Max(1, NumSentimentClasses);
            int count = Math.Min(classCount, prediction.Length);
            int bestIdx = 0;
            T bestVal = prediction.Data.Span[0];
            var probabilities = new Dictionary<string, T>();

            for (int c = 0; c < count; c++)
            {
                var value = prediction.Data.Span[c];
                if (NumOps.GreaterThan(value, bestVal))
                {
                    bestVal = value;
                    bestIdx = c;
                }

                probabilities[GetSentimentLabel(c, classCount)] = value;
            }

            results[i] = new SentimentResult<T>
            {
                OriginalText = texts[i],
                PredictedClass = GetSentimentLabel(bestIdx, classCount),
                Confidence = bestVal,
                ClassProbabilities = probabilities
            };
        }

        return results;
    }

    private static string GetSentimentLabel(int index, int classCount)
    {
        if (classCount == 3)
        {
            return index switch
            {
                0 => "negative",
                1 => "neutral",
                2 => "positive",
                _ => $"class_{index}"
            };
        }

        return $"class_{index}";
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
    public virtual int[] Tokenize(string text, int? maxLength = null) => new int[maxLength ?? MaxSequenceLength];

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
        metrics["MaxSequenceLength"] = NumOps.FromDouble(MaxSequenceLength);
        metrics["VocabularySize"] = NumOps.FromDouble(VocabularySize);
        metrics["HiddenDimension"] = NumOps.FromDouble(HiddenDimension);
        metrics["NumSentimentClasses"] = NumOps.FromDouble(NumSentimentClasses);

        return metrics;
    }
}
