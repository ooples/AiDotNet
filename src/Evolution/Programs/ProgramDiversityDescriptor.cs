using System.Collections.ObjectModel;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>Measures how far a candidate program sits from a fixed reference set of programs.</summary>
/// <remarks>
/// <para>
/// The distance between two programs is the reference implementation's fast approximation — the absolute
/// difference in character length weighted 0.1, the absolute difference in line count weighted 10, and the size of
/// the symmetric difference of their character sets weighted 0.5 — and the descriptor value is the mean distance
/// to every reference program. What differs is where the reference set comes from. OpenEvolve rebuilds it from the
/// live database with a greedy selection seeded by the global random module, so the same candidate can be assigned
/// different diversity values in two runs of the same configuration, and the value drifts as the database grows.
/// Here the reference set is supplied once at construction, deduplicated and sorted ordinally, so the descriptor
/// is a pure function of the genome and survives a checkpoint resume unchanged.
/// </para>
/// <para>
/// Reference sources are normalized the same way genomes are, so a reference that differs only in line endings
/// contributes the same distance as one that does not. A reference identical to the candidate is skipped, and an
/// empty reference set yields zero.
/// </para>
/// <para><b>For Beginners:</b> This descriptor answers "how unusual is this program compared with a fixed set of
/// examples?". You give it a handful of reference programs when you create it — typically the seed programs of the
/// run — and it reports the average distance from the candidate to those references. Using it as an archive axis
/// keeps genuinely different approaches alive instead of letting the search fill up with variations on one idea.
/// The reference set never changes during a run, which is what makes the numbers comparable from start to
/// finish.</para>
/// </remarks>
public sealed class ProgramDiversityDescriptor : IRebasableProgramDescriptor
{
    /// <summary>The descriptor name used when none is supplied.</summary>
    public const string DefaultName = "diversity";

    private readonly ReadOnlyCollection<string> _references;

    /// <summary>Initializes a diversity descriptor against a fixed reference set.</summary>
    /// <param name="referenceSources">
    /// The programs the candidate is compared with. Entries are normalized, deduplicated, and sorted ordinally, so
    /// the order they are supplied in does not affect the result.
    /// </param>
    /// <param name="name">The archive dimension name this descriptor fills.</param>
    /// <exception cref="ArgumentNullException"><paramref name="referenceSources"/> is <c>null</c>, or an entry is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="name"/> is empty or white space.</exception>
    public ProgramDiversityDescriptor(IEnumerable<string> referenceSources, string name = DefaultName)
    {
        Guard.NotNull(referenceSources);
        Guard.NotNullOrWhiteSpace(name);

        var unique = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string reference in referenceSources)
        {
            if (reference is null) throw new ArgumentNullException(nameof(referenceSources), "Reference sources cannot be null.");
            string normalized = ProgramText.Normalize(reference);
            if (normalized.Length == 0) continue;
            unique.Add(normalized);
        }

        var ordered = new List<string>(unique);
        _references = new ReadOnlyCollection<string>(ordered);
        Name = name.Trim();

        var components = new List<string>(ordered.Count + 2) { "program-diversity-descriptor-v1", Name };
        components.AddRange(ordered);
        VersionHash = EvolutionHash.Combine(components);
    }

    /// <summary>Initializes a diversity descriptor against a fixed set of reference genomes.</summary>
    /// <param name="referenceGenomes">The genomes the candidate is compared with.</param>
    /// <param name="name">The archive dimension name this descriptor fills.</param>
    /// <exception cref="ArgumentNullException"><paramref name="referenceGenomes"/> is <c>null</c>, or an entry is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="name"/> is empty or white space.</exception>
    public ProgramDiversityDescriptor(IEnumerable<ProgramGenome> referenceGenomes, string name = DefaultName)
        : this(ToSources(referenceGenomes), name)
    {
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <summary>Gets the normalized, deduplicated, ordinally sorted reference programs.</summary>
    public IReadOnlyList<string> ReferenceSources => _references;

    /// <inheritdoc/>
    /// <remarks>
    /// Covers the dimension name and every normalized reference program, so swapping the reference set changes the
    /// containing <see cref="ProgramDescriptorSet.VersionHash"/> and a checkpoint binned against the old set is
    /// refused instead of being silently re-binned against distances it never produced.
    /// </remarks>
    public string VersionHash { get; }

    /// <inheritdoc/>
    /// <remarks>
    /// The result is a new descriptor, not a changed one: the old reading may still be in use while this is being
    /// prepared, and <see cref="VersionHash"/> is supposed to identify one exact reading. Rebasing alone leaves the
    /// archive holding coordinates taken against the old references, so pair it with a re-measurement of what is
    /// already archived — otherwise the map mixes readings from two different rulers.
    /// </remarks>
    public IRebasableProgramDescriptor Rebase(IReadOnlyList<ProgramGenome> references)
    {
        Guard.NotNull(references);
        return new ProgramDiversityDescriptor(references, Name);
    }

    /// <inheritdoc/>
    public double Compute(ProgramGenome genome)
    {
        Guard.NotNull(genome);
        if (_references.Count == 0) return 0;

        double total = 0;
        int compared = 0;
        string candidate = genome.NormalizedSource;
        foreach (string reference in _references)
        {
            if (string.Equals(candidate, reference, StringComparison.Ordinal)) continue;
            total += Distance(candidate, reference);
            compared++;
        }

        return compared == 0 ? 0 : total / compared;
    }

    /// <summary>Computes the fast structural distance between two program sources.</summary>
    /// <param name="first">The first source; it is normalized before comparison.</param>
    /// <param name="second">The second source; it is normalized before comparison.</param>
    /// <returns>
    /// A non-negative distance combining length difference, line-count difference, and character-set difference;
    /// zero when the normalized sources are identical.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="first"/> or <paramref name="second"/> is <c>null</c>.</exception>
    public static double ComputeDistance(string first, string second)
    {
        if (first is null) throw new ArgumentNullException(nameof(first));
        if (second is null) throw new ArgumentNullException(nameof(second));
        return Distance(ProgramText.Normalize(first), ProgramText.Normalize(second));
    }

    private static double Distance(string first, string second)
    {
        if (string.Equals(first, second, StringComparison.Ordinal)) return 0;

        double lengthDifference = Math.Abs(first.Length - second.Length);
        double lineDifference = Math.Abs(CountLineFeeds(first) - CountLineFeeds(second));

        var firstSet = new HashSet<char>(first);
        var secondSet = new HashSet<char>(second);
        int shared = 0;
        foreach (char character in firstSet)
        {
            if (secondSet.Contains(character)) shared++;
        }

        double symmetricDifference = firstSet.Count + secondSet.Count - (2.0 * shared);
        return (lengthDifference * 0.1) + (lineDifference * 10.0) + (symmetricDifference * 0.5);
    }

    private static int CountLineFeeds(string text)
    {
        int count = 0;
        foreach (char character in text)
        {
            if (character == '\n') count++;
        }

        return count;
    }

    private static IEnumerable<string> ToSources(IEnumerable<ProgramGenome> genomes)
    {
        Guard.NotNull(genomes);
        var sources = new List<string>();
        foreach (ProgramGenome genome in genomes)
        {
            if (genome is null) throw new ArgumentNullException(nameof(genomes), "Reference genomes cannot be null.");
            sources.Add(genome.NormalizedSource);
        }

        return sources;
    }
}
