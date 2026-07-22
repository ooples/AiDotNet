namespace AiDotNet.Interfaces;

/// <summary>
/// A minimal, numeric-type-agnostic text generator: turns a prompt into generated text.
/// </summary>
/// <remarks>
/// <para>
/// Several RAG helpers (query expansion, multi-query generation, LLM reranking, context compression)
/// only need "prompt in, text out" and do not care about the numeric type <c>T</c> used for scoring.
/// This non-generic contract lets them accept any real generator — including any
/// <see cref="IGenerator{T}"/> (which extends this) backed by a live chat model — instead of running
/// lexical heuristics with unused API credentials.
/// </para>
/// <para><b>For Beginners:</b> it's the "give it a prompt, get back text" part of a language model,
/// with no other baggage — so the string-only RAG helpers can use a real LLM.</para>
/// </remarks>
public interface ITextGenerator
{
    /// <summary>
    /// Generates text for the given prompt.
    /// </summary>
    /// <param name="prompt">The input prompt.</param>
    /// <returns>The generated text.</returns>
    string Generate(string prompt);
}
