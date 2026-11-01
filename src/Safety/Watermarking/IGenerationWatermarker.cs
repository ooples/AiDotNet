using AiDotNet.Agentic.Models.Local;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// A text watermarker that embeds its keyed watermark while a language model generates, by biasing the model's
/// next-token logits.
/// </summary>
/// <remarks>
/// <para>
/// This is the watermark of Kirchenbauer et al. (2023): at every step the preceding token(s) and the key select a
/// "green list" holding a fraction gamma of the vocabulary, and delta is added to the logits of the green tokens before
/// sampling. Detection counts the green tokens of a text and applies a one-proportion z-test.
/// </para>
/// <para>
/// Embedding needs a hook into generation: wrap the model with <see cref="WatermarkModel"/> (for example before
/// handing it to <see cref="LocalEngineChatClient{T}"/>) or call <see cref="BiasLogits"/> on each step's logits in your
/// own decode loop. Detection needs the token ids, either directly (<see cref="DetectTokens"/>) or by re-tokenizing the
/// text with the generator's tokenizer.
/// </para>
/// <para>
/// <b>For Beginners:</b> while the AI writes, this watermarker nudges it toward words from a secret, constantly
/// changing list. The text reads normally, but someone with the key can count how often the AI picked listed
/// words, which happens far more often than chance.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IGenerationWatermarker<T> : ITextWatermarker<T>, IKeyedWatermarker
{
    /// <summary>
    /// Adds the watermark bias to <paramref name="logits"/> in place: delta is added to every token on the green list
    /// selected by the last tokens of <paramref name="context"/>.
    /// </summary>
    /// <param name="logits">The next-token logits, one per vocabulary entry.</param>
    /// <param name="context">The token ids so far (prompt included). A context shorter than the hashing width is left unbiased.</param>
    void BiasLogits(Vector<T> logits, IReadOnlyList<int> context);

    /// <summary>
    /// Wraps <paramref name="model"/> so every next-token logits vector it returns carries the watermark bias. An
    /// incremental (KV-cached) model is wrapped as an incremental model.
    /// </summary>
    /// <param name="model">The language model to watermark.</param>
    /// <returns>A model whose generations carry this watermark.</returns>
    ICausalLanguageModel<T> WatermarkModel(ICausalLanguageModel<T> model);

    /// <summary>
    /// Detects the watermark in a sequence of token ids.
    /// </summary>
    /// <param name="tokenIds">The token ids of the text to test, in order.</param>
    /// <returns>The detection result.</returns>
    WatermarkResult DetectTokens(IReadOnlyList<int> tokenIds);
}
