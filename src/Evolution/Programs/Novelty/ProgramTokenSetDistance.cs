using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Novelty;

/// <summary>A Jaccard distance over the lexical tokens of two programs; no provider, no model, no network.</summary>
/// <remarks>
/// <para>
/// The distance is one minus the Jaccard index of the two token sets: identical token sets give zero, disjoint sets
/// give one, and everything in between scales with how much vocabulary the two programs share. Tokenization keeps
/// identifier and number runs whole and treats every other non-space character as its own token, so a rename
/// registers as a difference while reindentation and line-ending changes do not — the sources are compared after
/// <see cref="ProgramGenome"/> normalization, which already removed those.
/// </para>
/// <para>
/// Cost is a single pass over each source plus two hash-set operations, so one comparison is linear in program
/// length with no allocation per comparison beyond the two token sets, and the sets for a fixed incumbent could be
/// reused by a caller that keeps them. This is the whole point of shipping it: the reference implementation has no
/// distance metric at all, so its cheapest possible novelty decision is an embedding request over the network,
/// while the cheapest decision here is arithmetic on text already in memory.
/// </para>
/// <para>
/// The metric is symmetric, deterministic across processes and target frameworks, and returns zero for two genomes
/// with the same normalized source — the properties <see cref="IGenomeDistance{TGenome}"/> requires. Note that it
/// is a set distance, not a sequence distance: reordering a program's lines without changing its vocabulary gives
/// distance zero. Pair it with <see cref="ProgramLineEditDistance"/> when order matters.
/// </para>
/// <para><b>For Beginners:</b> This asks a simple question — of all the words, numbers and symbols that appear in
/// these two programs, what fraction appears in both? If nearly all of them do, the programs are near-twins and the
/// number is close to zero. If they barely overlap, it is close to one. It costs nothing but a little arithmetic,
/// which is why it is the first check a novelty gate makes.</para>
/// </remarks>
public sealed class ProgramTokenSetDistance : IGenomeDistance<ProgramGenome>
{
    /// <summary>The identifier this metric reports.</summary>
    public const string MetricId = "program-token-set";

    /// <summary>The default number of genome token sets retained.</summary>
    public const int DefaultMemoCapacity = 1_024;

    private readonly Dictionary<string, HashSet<string>> _memo = new(StringComparer.Ordinal);
    private readonly Queue<string> _memoOrder = new();
    private readonly object _gate = new();

    /// <summary>Initializes a token-set distance.</summary>
    /// <param name="memoCapacity">
    /// How many genome token sets are retained, or zero to tokenize on every comparison; 0 to 1,000,000.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="memoCapacity"/> is outside its permitted range.</exception>
    /// <remarks>
    /// Memoization is keyed by <see cref="ProgramGenome.Id"/>, the SHA-256 of the normalized source of an immutable
    /// genome, so a remembered token set can only ever belong to the text it was computed from and the metric stays
    /// a pure function of its arguments. It matters because a novelty decision compares one candidate against a
    /// whole set of stable incumbents: without it every decision re-tokenizes every incumbent, and the incumbents
    /// do not change between decisions.
    /// </remarks>
    public ProgramTokenSetDistance(int memoCapacity = DefaultMemoCapacity)
    {
        if (memoCapacity < 0 || memoCapacity > 1_000_000)
        {
            throw new ArgumentOutOfRangeException(nameof(memoCapacity), memoCapacity,
                "Value must be between 0 and 1000000.");
        }

        MemoCapacity = memoCapacity;
    }

    /// <summary>Gets how many genome token sets this instance retains; zero disables memoization.</summary>
    public int MemoCapacity { get; }

    /// <inheritdoc/>
    public string Id => MetricId;

    /// <inheritdoc/>
    public string VersionHash => MetricId + "-v1";

    /// <inheritdoc/>
    public double Distance(ProgramGenome first, ProgramGenome second)
    {
        Guard.NotNull(first);
        Guard.NotNull(second);
        if (string.Equals(first.Id, second.Id, StringComparison.Ordinal)) return 0.0;
        return Jaccard(TokensFor(first), TokensFor(second));
    }

    /// <summary>Discards every memoized token set.</summary>
    public void ClearMemo()
    {
        lock (_gate)
        {
            _memo.Clear();
            _memoOrder.Clear();
        }
    }

    private HashSet<string> TokensFor(ProgramGenome genome)
    {
        if (MemoCapacity == 0) return ProgramTokenizer.Tokenize(genome.NormalizedSource);

        lock (_gate)
        {
            if (_memo.TryGetValue(genome.Id, out HashSet<string>? remembered)) return remembered;
        }

        // Tokenizing outside the lock keeps a long program from blocking every other comparison; a duplicate
        // computation under contention is harmless because the result is a pure function of the genome.
        HashSet<string> tokens = ProgramTokenizer.Tokenize(genome.NormalizedSource);
        lock (_gate)
        {
            if (_memo.TryGetValue(genome.Id, out HashSet<string>? raced)) return raced;
            while (_memo.Count >= MemoCapacity && _memoOrder.Count > 0)
            {
                _memo.Remove(_memoOrder.Dequeue());
            }

            if (_memo.Count < MemoCapacity)
            {
                _memo[genome.Id] = tokens;
                _memoOrder.Enqueue(genome.Id);
            }
        }

        return tokens;
    }

    /// <summary>Computes the token-set distance between two program sources.</summary>
    /// <param name="first">The first source; normalized before comparison.</param>
    /// <param name="second">The second source; normalized before comparison.</param>
    /// <returns>Zero when the token sets match, one when they are disjoint.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="first"/> or <paramref name="second"/> is <c>null</c>.</exception>
    public static double ComputeDistance(string first, string second)
    {
        Guard.NotNull(first);
        Guard.NotNull(second);
        return Compute(ProgramText.Normalize(first), ProgramText.Normalize(second));
    }

    private static double Compute(string first, string second)
    {
        if (string.Equals(first, second, StringComparison.Ordinal)) return 0.0;
        return Jaccard(ProgramTokenizer.Tokenize(first), ProgramTokenizer.Tokenize(second));
    }

    private static double Jaccard(HashSet<string> firstTokens, HashSet<string> secondTokens)
    {
        if (firstTokens.Count == 0 && secondTokens.Count == 0) return 0.0;

        HashSet<string> smaller = firstTokens.Count <= secondTokens.Count ? firstTokens : secondTokens;
        HashSet<string> larger = ReferenceEquals(smaller, firstTokens) ? secondTokens : firstTokens;

        int shared = 0;
        foreach (string token in smaller)
        {
            if (larger.Contains(token)) shared++;
        }

        int union = firstTokens.Count + secondTokens.Count - shared;
        if (union == 0) return 0.0;

        double distance = 1.0 - ((double)shared / union);
        return distance < 0.0 ? 0.0 : distance > 1.0 ? 1.0 : distance;
    }
}
