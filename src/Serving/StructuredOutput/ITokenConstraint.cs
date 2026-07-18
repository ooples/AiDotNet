namespace AiDotNet.Serving.StructuredOutput;

/// <summary>
/// A per-sequence decoding constraint that restricts which vocabulary tokens may be emitted next,
/// enabling structured / guided output (JSON, regex, grammar-constrained generation).
/// </summary>
/// <remarks>
/// <para>
/// A constraint is a small state machine over generated tokens. Before each sampling step the engine
/// calls <see cref="ApplyMask"/> to forbid tokens that would violate the target language; after a token
/// is chosen it calls <see cref="Accept"/> to advance the state. The constraint is stateful and belongs
/// to exactly one sequence — never share an instance across concurrent sequences.
/// </para>
/// <para><b>For Beginners:</b> This is how "give me valid JSON" or "answer only yes/no" is enforced.
/// At every step the model still produces scores for all tokens; the constraint blanks out the ones that
/// would break the required format so the model is forced to stay on the rails, while still choosing
/// freely among the tokens that keep the output valid.</para>
/// </remarks>
public interface ITokenConstraint
{
    /// <summary>
    /// Masks tokens that are not permitted as the next token in-place: sets
    /// <c>logits[i] = <see cref="float.NegativeInfinity"/></c> for every token id <c>i</c> that would
    /// violate the constraint given the tokens accepted so far. At least one token (a real token or the
    /// end-of-sequence token) is always left unmasked, so sampling can never dead-end.
    /// </summary>
    /// <param name="logits">The per-token logit scores for the current step, over the full vocabulary.</param>
    void ApplyMask(Span<float> logits);

    /// <summary>
    /// Advances the constraint state after <paramref name="tokenId"/> has been committed to the sequence.
    /// </summary>
    /// <param name="tokenId">The token id that was emitted for this step.</param>
    void Accept(int tokenId);

    /// <summary>
    /// Gets a value indicating whether the constraint has reached an accepting/terminal state, i.e. the
    /// generated text so far is a complete valid instance of the target language and generation may stop.
    /// </summary>
    bool IsComplete { get; }
}
