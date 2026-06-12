namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// The minimal contract an in-process language model exposes to the local generation engine: given the
/// tokens seen so far, produce the logits for the next token. This is the seam between AiDotNet's own
/// Transformer (or any other model) and <see cref="LocalEngineChatClient{T}"/>.
/// </summary>
/// <typeparam name="T">The tensor element type (e.g., <see cref="float"/> or <see cref="double"/>).</typeparam>
/// <remarks>
/// <para>
/// Keeping the contract this small means the generation loop, sampling, and chat templating are written
/// once and tested independently of any particular model, and the real network is wired in behind this
/// interface. An implementation is free to maintain an internal KV-cache keyed on the growing context so
/// repeated calls stay efficient; callers only ever ask for "the next-token logits given this context".
/// </para>
/// <para><b>For Beginners:</b> A language model, at its heart, answers one question over and over: "given
/// everything so far, how likely is each possible next word-piece?" Those likelihoods (before turning them
/// into probabilities) are called <em>logits</em>. This interface is exactly that one question, so the rest
/// of the engine can focus on <em>choosing</em> the next token and stitching the words back together.
/// </para>
/// </remarks>
public interface ICausalLanguageModel<T>
{
    /// <summary>
    /// Gets the size of the model's vocabulary (the length of the logits vector returned by
    /// <see cref="NextTokenLogits"/>).
    /// </summary>
    int VocabularySize { get; }

    /// <summary>
    /// Computes the next-token logits for the given context.
    /// </summary>
    /// <param name="tokenIds">The token ids of the context so far (prompt plus any tokens already generated). Must be non-empty.</param>
    /// <returns>A vector of length <see cref="VocabularySize"/> giving the unnormalized score for each candidate next token.</returns>
    Vector<T> NextTokenLogits(IReadOnlyList<int> tokenIds);
}
