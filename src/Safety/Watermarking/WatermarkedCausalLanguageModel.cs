using AiDotNet.Agentic.Models.Local;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// A language model whose next-token logits carry a generation-time watermark: the inner model's logits plus the
/// watermarker's green-list bias for the current context.
/// </summary>
/// <remarks>
/// <para>
/// Created by <see cref="IGenerationWatermarker{T}.WatermarkModel"/>. The inner model's logits are copied before the
/// bias is added, so a model that reuses its output buffer is not modified. Use the wrapped model anywhere an
/// <see cref="ICausalLanguageModel{T}"/> is accepted, for example in <see cref="LocalEngineChatClient{T}"/>; every
/// token it samples is then drawn from the watermarked distribution.
/// </para>
/// <para>
/// <b>For Beginners:</b> this wraps your model so everything it writes carries your watermark, without changing the
/// model itself.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type of the logits.</typeparam>
public class WatermarkedCausalLanguageModel<T> : ICausalLanguageModel<T>
{
    private readonly ICausalLanguageModel<T> _inner;

    /// <summary>
    /// Wraps <paramref name="inner"/> with <paramref name="watermarker"/>'s logit bias.
    /// </summary>
    /// <param name="inner">The language model to watermark.</param>
    /// <param name="watermarker">The generation watermarker that biases the logits.</param>
    public WatermarkedCausalLanguageModel(ICausalLanguageModel<T> inner, IGenerationWatermarker<T> watermarker)
    {
        Guard.NotNull(inner);
        Guard.NotNull(watermarker);
        _inner = inner;
        Watermarker = watermarker;
    }

    /// <summary>The watermarker whose bias is applied.</summary>
    public IGenerationWatermarker<T> Watermarker { get; }

    /// <inheritdoc />
    public int VocabularySize => _inner.VocabularySize;

    /// <inheritdoc />
    public Vector<T> NextTokenLogits(IReadOnlyList<int> tokenIds) => Biased(_inner.NextTokenLogits(tokenIds), tokenIds);

    /// <summary>A copy of <paramref name="logits"/> with the watermark bias for <paramref name="context"/>.</summary>
    protected Vector<T> Biased(Vector<T> logits, IReadOnlyList<int> context)
    {
        if (logits is null) throw new InvalidOperationException("The inner model returned null logits.");
        var copy = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++) copy[i] = logits[i];
        Watermarker.BiasLogits(copy, context);
        return copy;
    }
}

/// <summary>
/// The incremental (KV-cached) form of <see cref="WatermarkedCausalLanguageModel{T}"/>: it tracks the context the
/// inner model has seen, so each appended token's logits are biased for the right green list.
/// </summary>
/// <remarks>
/// <para>
/// The tracked context starts at <see cref="StartSequence"/> and grows with every <see cref="AppendToken"/>; it is not
/// shared between threads, matching the inner model's own cache. <b>For Beginners:</b> this keeps a fast, cached model
/// fast while still watermarking it.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type of the logits.</typeparam>
public class WatermarkedIncrementalCausalLanguageModel<T> : WatermarkedCausalLanguageModel<T>, IIncrementalCausalLanguageModel<T>
{
    private readonly IIncrementalCausalLanguageModel<T> _incremental;
    private readonly List<int> _context = new();

    /// <summary>
    /// Wraps <paramref name="inner"/> with <paramref name="watermarker"/>'s logit bias.
    /// </summary>
    /// <param name="inner">The incremental language model to watermark.</param>
    /// <param name="watermarker">The generation watermarker that biases the logits.</param>
    public WatermarkedIncrementalCausalLanguageModel(IIncrementalCausalLanguageModel<T> inner, IGenerationWatermarker<T> watermarker)
        : base(inner, watermarker)
    {
        _incremental = inner;
    }

    /// <inheritdoc />
    public void ResetCache()
    {
        _incremental.ResetCache();
        _context.Clear();
    }

    /// <inheritdoc />
    public Vector<T> StartSequence(IReadOnlyList<int> promptTokenIds)
    {
        if (promptTokenIds is null) throw new ArgumentNullException(nameof(promptTokenIds));
        var logits = _incremental.StartSequence(promptTokenIds);
        _context.Clear();
        _context.AddRange(promptTokenIds);
        return Biased(logits, _context);
    }

    /// <inheritdoc />
    public Vector<T> AppendToken(int tokenId)
    {
        var logits = _incremental.AppendToken(tokenId);
        _context.Add(tokenId);
        return Biased(logits, _context);
    }
}
