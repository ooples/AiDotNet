using AiDotNet.Helpers;

namespace AiDotNet.Serving.StructuredOutput;

/// <summary>
/// An <see cref="ITokenConstraint"/> backed by a compiled deterministic finite automaton (DFA) over
/// token ids. Higher-level compilers (regex, JSON-schema, grammar) lower their language to a token-level
/// DFA and hand it to this class, which does the per-step masking and state advance.
/// </summary>
/// <remarks>
/// <para>
/// Each state has a set of outgoing transitions <c>tokenId -&gt; nextState</c>. In an accepting state the
/// end-of-sequence token is additionally permitted, which is how the engine learns it may stop. The DFA
/// is tokenizer-agnostic: it speaks only token ids, so it composes with any tokenizer the serving layer
/// registers.
/// </para>
/// <para><b>For Beginners:</b> think of this as a board game where the current square lists which moves
/// (tokens) are legal. The model picks among the legal moves; illegal ones are hidden. When you land on a
/// "finish" square you're allowed to stop.</para>
/// </remarks>
public sealed class TokenFsmConstraint : ITokenConstraint
{
    private readonly IReadOnlyDictionary<int, int>[] _transitions;
    private readonly bool[] _accepting;
    private readonly int _eosTokenId;
    private int _current;
    private bool _finished;

    /// <summary>
    /// Creates a token-level DFA constraint.
    /// </summary>
    /// <param name="transitions">Per-state outgoing transitions (<c>tokenId -&gt; next state</c>), indexed
    /// by state id. Length is the number of states.</param>
    /// <param name="acceptingStates">The state ids in which the generated text is a complete valid
    /// instance (end-of-sequence becomes permitted there).</param>
    /// <param name="eosTokenId">The end-of-sequence token id permitted in accepting states.</param>
    /// <param name="startState">The initial state id (default 0).</param>
    public TokenFsmConstraint(
        IReadOnlyList<IReadOnlyDictionary<int, int>> transitions,
        IEnumerable<int> acceptingStates,
        int eosTokenId,
        int startState = 0)
    {
        Guard.NotNull(transitions);
        Guard.NotNull(acceptingStates);
        if (transitions.Count == 0)
        {
            throw new ArgumentException("A DFA constraint must have at least one state.", nameof(transitions));
        }
        if (startState < 0 || startState >= transitions.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(startState),
                $"Start state {startState} is outside the state range [0, {transitions.Count - 1}].");
        }

        _transitions = new IReadOnlyDictionary<int, int>[transitions.Count];
        _accepting = new bool[transitions.Count];
        for (int s = 0; s < transitions.Count; s++)
        {
            var row = transitions[s] ?? throw new ArgumentException(
                $"Transition row for state {s} is null.", nameof(transitions));
            foreach (var target in row.Values)
            {
                if (target < 0 || target >= transitions.Count)
                {
                    throw new ArgumentException(
                        $"State {s} has a transition to out-of-range state {target}.", nameof(transitions));
                }
            }
            _transitions[s] = row;
        }

        foreach (int a in acceptingStates)
        {
            if (a < 0 || a >= _accepting.Length)
            {
                throw new ArgumentException(
                    $"Accepting state {a} is outside the state range [0, {_accepting.Length - 1}].",
                    nameof(acceptingStates));
            }
            _accepting[a] = true;
        }

        _eosTokenId = eosTokenId;
        _current = startState;
    }

    /// <inheritdoc/>
    public bool IsComplete => _finished || _accepting[_current];

    /// <inheritdoc/>
    public void ApplyMask(Span<float> logits)
    {
        // Once finished (EOS emitted) nothing further should be generated; forbid everything except EOS
        // so any additional step degenerates to another EOS rather than free-running.
        var allowed = _finished ? EmptyRow : _transitions[_current];
        bool eosAllowed = _finished || _accepting[_current];

        for (int i = 0; i < logits.Length; i++)
        {
            bool ok = (eosAllowed && i == _eosTokenId) || allowed.ContainsKey(i);
            if (!ok)
            {
                logits[i] = float.NegativeInfinity;
            }
        }

        // Safety net: a well-formed DFA never dead-ends, but if a state has no outgoing transitions and is
        // not accepting (or EOS is out of the vocab range), leave EOS open so sampling can still terminate.
        if (!eosAllowed && allowed.Count == 0 && _eosTokenId >= 0 && _eosTokenId < logits.Length)
        {
            logits[_eosTokenId] = 0f;
        }
    }

    /// <inheritdoc/>
    public void Accept(int tokenId)
    {
        if (_finished)
        {
            return;
        }
        if (tokenId == _eosTokenId)
        {
            _finished = true;
            return;
        }
        if (_transitions[_current].TryGetValue(tokenId, out int next))
        {
            _current = next;
        }
        // A token that isn't in the transition table can only occur if the caller sampled without applying
        // the mask; leaving the state unchanged keeps the constraint conservative rather than crashing.
    }

    private static readonly IReadOnlyDictionary<int, int> EmptyRow = new Dictionary<int, int>();

    /// <summary>
    /// Builds a constraint that forces an exact token sequence, then end-of-sequence. Useful as a building
    /// block and for tests (e.g. "the model must emit these tokens").
    /// </summary>
    /// <param name="tokenIds">The exact token sequence the output must equal.</param>
    /// <param name="eosTokenId">The end-of-sequence token id permitted once the sequence is complete.</param>
    public static TokenFsmConstraint FromSequence(IReadOnlyList<int> tokenIds, int eosTokenId)
    {
        Guard.NotNull(tokenIds);
        int n = tokenIds.Count;
        var rows = new Dictionary<int, int>[n + 1];
        for (int i = 0; i < n; i++)
        {
            rows[i] = new Dictionary<int, int> { [tokenIds[i]] = i + 1 };
        }
        rows[n] = new Dictionary<int, int>();
        return new TokenFsmConstraint(rows, new[] { n }, eosTokenId);
    }

    /// <summary>
    /// Builds a constraint that restricts the output to exactly one of a fixed set of token sequences
    /// (a choice / enum constraint). Sequences sharing a prefix share DFA states, so the model may pick
    /// freely among the still-viable choices at each step.
    /// </summary>
    /// <param name="choices">The allowed token sequences; each must be non-empty.</param>
    /// <param name="eosTokenId">The end-of-sequence token id permitted at the end of any choice.</param>
    public static TokenFsmConstraint FromChoices(IEnumerable<IReadOnlyList<int>> choices, int eosTokenId)
    {
        Guard.NotNull(choices);

        // Build a trie of the choices; each node is a DFA state. Node 0 is the root (start).
        var rows = new List<Dictionary<int, int>> { new() };
        var accepting = new HashSet<int>();
        bool any = false;

        foreach (var choice in choices)
        {
            if (choice is null || choice.Count == 0)
            {
                throw new ArgumentException("Each choice must be a non-empty token sequence.", nameof(choices));
            }
            any = true;
            int state = 0;
            foreach (int tok in choice)
            {
                if (!rows[state].TryGetValue(tok, out int next))
                {
                    next = rows.Count;
                    rows.Add(new Dictionary<int, int>());
                    rows[state][tok] = next;
                }
                state = next;
            }
            accepting.Add(state);
        }

        if (!any)
        {
            throw new ArgumentException("At least one choice is required.", nameof(choices));
        }

        return new TokenFsmConstraint(rows, accepting, eosTokenId);
    }
}
