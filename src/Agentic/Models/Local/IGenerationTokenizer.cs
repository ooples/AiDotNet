namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// The minimal tokenizer contract the local generation engine needs: turn text into token ids, turn token
/// ids back into text, and know which token marks end-of-sequence.
/// </summary>
/// <remarks>
/// <para>
/// This is intentionally narrower than the full <see cref="AiDotNet.Tokenization.Interfaces.ITokenizer"/> so
/// the engine stays decoupled and trivially testable. <see cref="TokenizerGenerationAdapter"/> bridges a
/// real repo tokenizer to this seam.
/// </para>
/// <para><b>For Beginners:</b> Models don't read text directly — they read numbers (token ids). This turns
/// your prompt into those numbers, turns the model's numbers back into readable text, and tells the engine
/// the special "stop here" token so it knows when the model is done.
/// </para>
/// </remarks>
public interface IGenerationTokenizer
{
    /// <summary>
    /// Gets the id of the end-of-sequence token. Generation stops when the model produces it. A negative
    /// value means "no EOS token" (generation then stops only at the token limit).
    /// </summary>
    int EosTokenId { get; }

    /// <summary>
    /// Encodes text into token ids.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <returns>The token ids.</returns>
    IReadOnlyList<int> Encode(string text);

    /// <summary>
    /// Decodes token ids back into text.
    /// </summary>
    /// <param name="tokenIds">The token ids to decode.</param>
    /// <returns>The decoded text.</returns>
    string Decode(IReadOnlyList<int> tokenIds);
}
