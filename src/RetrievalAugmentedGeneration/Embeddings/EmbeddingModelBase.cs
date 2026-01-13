
using System.Threading;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.Embeddings;

/// <summary>
/// Provides a base implementation for embedding models with common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements the IEmbeddingModel interface and provides common functionality
/// for text embedding models. It handles validation, batching, and normalization while allowing
/// derived classes to focus on implementing the core embedding algorithm.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all embedding models build upon.
/// 
/// Think of it like a template for creating embedding models:
/// - It handles common tasks (checking inputs, batching text, normalizing vectors)
/// - Specific embedding models (BERT, GPT, etc.) just fill in how they convert text to numbers
/// - This keeps code consistent and reduces duplication
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector calculations (typically float or double).</typeparam>
public abstract class EmbeddingModelBase<T> : IEmbeddingModel<T>, IDisposable
{
    /// <summary>
    /// Gets the numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the dimensionality of the embedding vectors produced by this model.
    /// </summary>
    public abstract int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the maximum length of text (in tokens) that this model can process.
    /// </summary>
    public abstract int MaxTokens { get; }

    /// <summary>
    /// Embeds a single text string into a vector representation.
    /// </summary>
    /// <param name="text">The text to embed.</param>
    /// <returns>A vector representing the semantic meaning of the input text.</returns>
    public Vector<T> Embed(string text)
    {
        ValidateText(text);
        return EmbedCoreAsync(text, CancellationToken.None).ConfigureAwait(false).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Asynchronously embeds a single text string into a vector representation.
    /// </summary>
    /// <param name="text">The text to embed.</param>
    /// <returns>A task representing the async operation, with the resulting vector.</returns>
    public virtual async Task<Vector<T>> EmbedAsync(string text, CancellationToken cancellationToken = default)
    {
        ValidateText(text);
        return await EmbedCoreAsync(text, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Embeds multiple text strings into vector representations in a single batch operation.
    /// </summary>
    /// <param name="texts">The collection of texts to embed.</param>
    /// <returns>A matrix where each row represents the embedding of the corresponding input text.</returns>
    public Matrix<T> EmbedBatch(IEnumerable<string> texts)
    {
        if (texts == null)
            throw new ArgumentNullException(nameof(texts));

        var textList = texts.ToList();
        if (textList.Count == 0)
            throw new ArgumentException("Text collection cannot be empty", nameof(texts));

        foreach (var text in textList)
        {
            ValidateText(text);
        }

        return EmbedBatchCoreAsync(textList, CancellationToken.None).ConfigureAwait(false).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Asynchronously embeds multiple text strings into vector representations in a single batch operation.
    /// </summary>
    /// <param name="texts">The collection of texts to embed.</param>
    /// <returns>A task representing the async operation, with the resulting matrix.</returns>
    public virtual async Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default)
    {
        if (texts == null)
            throw new ArgumentNullException(nameof(texts));

        var textList = texts.ToList();
        if (textList.Count == 0)
            throw new ArgumentException("Text collection cannot be empty", nameof(texts));

        foreach (var text in textList)
        {
            ValidateText(text);
        }

        return await EmbedBatchCoreAsync(textList, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Core embedding logic to be implemented by derived classes for a single text.
    /// </summary>
    /// <param name="text">The validated text to embed.</param>
    /// <returns>A vector representing the semantic meaning of the input text.</returns>
    /// <remarks>
    /// <para>
    /// Implementers should focus on the core embedding algorithm without worrying about
    /// input validation, which is handled by the base class. The returned vector should
    /// ideally be normalized for consistent similarity calculations.
    /// </para>
    /// <para><b>For Implementers:</b> This is where you put your specific embedding algorithm.
    /// 
    /// You don't need to:
    /// - Check if text is null or empty (already done)
    /// - Validate text length (already done)
    /// - Handle errors for invalid input (already done)
    /// 
    /// Just focus on: Converting the text into a Vector<T> of size EmbeddingDimension.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> EmbedCore(string text)
    {
        throw new NotSupportedException("Override EmbedCore or EmbedCoreAsync to provide embedding logic.");
    }

    /// <summary>
    /// Asynchronous core embedding logic to be implemented by derived classes.
    /// </summary>
    /// <param name="text">The validated text to embed.</param>
    /// <returns>A task representing the async operation, with the resulting vector.</returns>
    protected virtual Task<Vector<T>> EmbedCoreAsync(string text, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(EmbedCore(text));
    }

    /// <summary>
    /// Core batch embedding logic to be implemented by derived classes.
    /// </summary>
    /// <param name="texts">The validated collection of texts to embed.</param>
    /// <returns>A matrix where each row represents the embedding of the corresponding input text.</returns>
    /// <remarks>
    /// <para>
    /// The default implementation calls EmbedCore for each text and combines results into a matrix.
    /// Derived classes can override this to provide more efficient batch processing if their
    /// underlying model supports true batching.
    /// </para>
    /// <para><b>For Implementers:</b> Override this if your model can batch embed more efficiently.
    /// 
    /// For example:
    /// - Neural network models can process multiple texts in one forward pass
    /// - API-based models often have batch endpoints
    /// 
    /// If you don't override, each text is embedded individually (slower but works).
    /// </para>
    /// </remarks>
    protected virtual Matrix<T> EmbedBatchCore(IList<string> texts, CancellationToken cancellationToken = default)
    {
        var embeddings = new List<Vector<T>>();
        foreach (var text in texts)
        {
            cancellationToken.ThrowIfCancellationRequested();
            embeddings.Add(EmbedCoreAsync(text, cancellationToken).ConfigureAwait(false).GetAwaiter().GetResult());
        }

        return CreateMatrixFromVectors(embeddings);
    }

    /// <summary>
    /// Asynchronous core batch embedding logic to be implemented by derived classes.
    /// </summary>
    /// <param name="texts">The validated collection of texts to embed.</param>
    /// <returns>A task representing the async operation, with the resulting matrix.</returns>
    protected virtual async Task<Matrix<T>> EmbedBatchCoreAsync(IList<string> texts, CancellationToken cancellationToken = default)
    {
        var embeddings = new List<Vector<T>>();
        foreach (var text in texts)
        {
            cancellationToken.ThrowIfCancellationRequested();
            embeddings.Add(await EmbedCoreAsync(text, cancellationToken).ConfigureAwait(false));
        }

        return CreateMatrixFromVectors(embeddings);
    }

    /// <summary>
    /// Validates the input text.
    /// </summary>
    /// <param name="text">The text to validate.</param>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this if you need custom validation rules.
    /// For example, checking for specific characters, length limits, or encoding.
    /// </para>
    /// </remarks>
    protected virtual void ValidateText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Text cannot be null or empty", nameof(text));
    }

    /// <summary>
    /// Creates a matrix from a collection of vectors.
    /// </summary>
    /// <param name="vectors">The vectors to combine into a matrix.</param>
    /// <returns>A matrix where each row is one of the input vectors.</returns>
    protected Matrix<T> CreateMatrixFromVectors(IList<Vector<T>> vectors)
    {
        if (vectors == null || vectors.Count == 0)
            throw new ArgumentException("Vector collection cannot be null or empty");

        var rows = vectors.Count;
        var cols = vectors[0].Length;

        // Validate all vectors have same dimension
        for (int i = 1; i < vectors.Count; i++)
        {
            if (vectors[i].Length != cols)
                throw new ArgumentException($"All vectors must have the same dimension. Expected {cols}, got {vectors[i].Length} at index {i}");
        }

        var data = new T[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                data[i, j] = vectors[i][j];
            }
        }

        return new Matrix<T>(data);
    }

    /// <summary>
    /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases unmanaged and - optionally - managed resources.
    /// </summary>
    /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        // Base implementation does nothing, intended for overrides
    }
}
