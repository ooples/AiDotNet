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
/// <para><b>Implementers:</b> this interface is new in the serving layer and defines both
/// <see cref="IsComplete"/> (the output so far is a complete valid instance) and <see cref="IsTerminal"/>
/// (complete AND no valid non-EOS continuation remains — where the engine stops). A custom constraint must
/// implement both; a common pattern is <c>IsTerminal =&gt; IsComplete &amp;&amp; !HasNonEosContinuation</c>.
/// The built-in constraints (JSON grammar/schema, regex, token FSM) are the reference implementations.</para>
/// </remarks>
public interface ITokenConstraint
{
    /// <summary>
    /// Masks tokens that are not permitted as the next token in-place: sets
    /// <c>logits[i] = <see cref="float.NegativeInfinity"/></c> for every token id <c>i</c> that would
    /// violate the constraint given the tokens accepted so far. In every reachable state at least one token
    /// (a real token or, when the output is accepting, the end-of-sequence token) is left unmasked.
    /// </summary>
    /// <remarks>
    /// If generation reaches a state from which the required format can no longer be satisfied — no valid
    /// non-EOS token remains and the output so far is not a complete valid instance (a non-accepting
    /// dead-end) — the implementation throws <see cref="StructuredOutputConstraintException"/> rather than
    /// leaving the whole vocabulary masked or silently permitting EOS. The engine fails that sequence closed.
    /// A well-formed constraint driven only through its own mask does not reach such a state.
    /// </remarks>
    /// <param name="logits">The per-token logit scores for the current step, over the full vocabulary.</param>
    /// <exception cref="StructuredOutputConstraintException">The constraint cannot be satisfied from the
    /// current (non-accepting dead-end) state.</exception>
    void ApplyMask(Span<float> logits);

    /// <summary>
    /// Advances the constraint state after <paramref name="tokenId"/> has been committed to the sequence.
    /// </summary>
    /// <param name="tokenId">The token id that was emitted for this step.</param>
    void Accept(int tokenId);

    /// <summary>
    /// Gets a value indicating whether the generated text so far is <b>accepting</b> — a complete valid
    /// instance of the target language. Unlike <see cref="IsTerminal"/>, an accepting state may still have
    /// valid continuations (e.g. after one digit of <c>\d+</c>, or a complete JSON value that could take
    /// more whitespace). The engine does NOT stop here; it uses this only to decide that the end-of-sequence
    /// token is permitted (so the model may choose to stop) and to distinguish a valid completion from a
    /// dead-end. Callers/tests use it to ask "is the output a complete valid instance right now?".
    /// </summary>
    bool IsComplete { get; }

    /// <summary>
    /// Gets a value indicating whether the constraint is <b>terminal</b>: the generated text so far is a
    /// complete valid instance of the target language AND no valid non-EOS continuation remains, so the
    /// engine may stop the sequence.
    /// </summary>
    /// <remarks>
    /// This is deliberately distinct from merely being in an <i>accepting</i> state. An accepting state that
    /// still has valid continuations (e.g. after the first digit of <c>\d+</c>, or a complete JSON value that
    /// could still be followed by insignificant whitespace) is NOT terminal — stopping there would truncate
    /// output to its shortest valid prefix. For such variable-length matches the model itself ends generation
    /// by emitting the end-of-sequence token, which <see cref="ApplyMask"/> leaves permitted whenever the
    /// current state is accepting. Terminality is reached only when the grammar is exhausted (e.g. a closed
    /// fixed literal, or a state from which no vocabulary token can extend the match).
    /// </remarks>
    bool IsTerminal { get; }
}
