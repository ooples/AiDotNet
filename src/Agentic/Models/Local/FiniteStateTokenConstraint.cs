namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// A constraint defined by a finite-state grammar over token ids: the set of allowed next tokens depends on
/// the most recently generated token (the current state). This expresses exact sequences, branching choices,
/// and loops — the general mechanism a JSON-schema or regular grammar compiles down to.
/// </summary>
/// <remarks>
/// <para>
/// State is the last generated token. Before any token is generated, the <c>start</c> set applies. From a
/// state with no outgoing transitions, the empty set is returned, which tells the engine to stop — so a
/// chain of single-token transitions forces an exact output, and terminal states end generation cleanly.
/// </para>
/// <para><b>For Beginners:</b> Picture a flowchart where each box says "from here, you may only go to these
/// tokens next". Generation walks the flowchart; it can never step off it. Give each box exactly one exit and
/// you force a precise output; give it several and you allow choices. Boxes with no exit end the answer.
/// </para>
/// </remarks>
public sealed class FiniteStateTokenConstraint : ITokenConstraint
{
    private readonly IReadOnlyCollection<int> _start;
    private readonly IReadOnlyDictionary<int, IReadOnlyCollection<int>> _transitions;

    /// <summary>
    /// Initializes a new finite-state constraint.
    /// </summary>
    /// <param name="start">The tokens allowed as the very first generated token. Must be non-empty.</param>
    /// <param name="transitions">
    /// A map from a just-generated token id to the tokens allowed after it. A token absent from the map (or
    /// mapped to an empty set) is a terminal state at which generation stops.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="start"/> or <paramref name="transitions"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="start"/> is empty.</exception>
    public FiniteStateTokenConstraint(
        IEnumerable<int> start,
        IReadOnlyDictionary<int, IReadOnlyCollection<int>> transitions)
    {
        Guard.NotNull(start);
        Guard.NotNull(transitions);
        var startCopy = new List<int>(start);
        if (startCopy.Count == 0)
        {
            throw new ArgumentException("At least one start token id is required.", nameof(start));
        }

        _start = startCopy.AsReadOnly();

        // Freeze the grammar: the caller may still hold mutable collections
        // behind these read-only interfaces, and a constraint whose allowed
        // sets change mid-generation would silently alter decoding behavior.
        var transitionsCopy = new Dictionary<int, IReadOnlyCollection<int>>(transitions.Count);
        foreach (var pair in transitions)
        {
            transitionsCopy[pair.Key] = pair.Value is null
                ? Array.Empty<int>()
                : new List<int>(pair.Value).AsReadOnly();
        }

        _transitions = transitionsCopy;
    }

    /// <inheritdoc/>
    public IReadOnlyCollection<int>? AllowedNextTokens(IReadOnlyList<int> generatedTokenIds)
    {
        Guard.NotNull(generatedTokenIds);
        if (generatedTokenIds.Count == 0)
        {
            return _start;
        }

        var lastToken = generatedTokenIds[generatedTokenIds.Count - 1];
        return _transitions.TryGetValue(lastToken, out var allowed) ? allowed : Array.Empty<int>();
    }
}
