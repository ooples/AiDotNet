using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Novelty;

/// <summary>A normalized Levenshtein distance over whole lines; order sensitive, and still entirely local.</summary>
/// <remarks>
/// <para>
/// The unit of edit is a line, not a character. That is the right granularity for evolved source: a model that
/// rewrites one loop body changes a handful of lines out of dozens, which this reports as a small distance, whereas
/// character-level edit distance over the same change is dominated by the unchanged bulk and is quadratic in
/// program length rather than in line count. The result is divided by the longer line count, so it lands in 0 to 1
/// and is comparable against the same threshold as <see cref="ProgramTokenSetDistance"/>.
/// </para>
/// <para>
/// Unlike the token-set metric this one is order sensitive: moving a block without changing any vocabulary is a
/// real distance here and zero there. It costs O(L1 x L2) time and O(min(L1, L2)) memory, so it is cheap for
/// ordinary programs and bounded for pathological ones — a source longer than <see cref="MaxComparedLines"/> lines
/// is compared on its first <see cref="MaxComparedLines"/> lines, which keeps a single comparison bounded no matter
/// what a model emits. Neither variant makes a network request or loads a model.
/// </para>
/// <para><b>For Beginners:</b> This counts how many whole lines you would have to add, delete or change to turn one
/// program into the other, then divides by the length of the longer one. Nought means identical, one means nothing
/// in common. Use it instead of the token metric when the order of the code matters to you, and alongside it when
/// you want both a vocabulary view and a structure view.</para>
/// </remarks>
public sealed class ProgramLineEditDistance : IGenomeDistance<ProgramGenome>
{
    /// <summary>The identifier this metric reports.</summary>
    public const string MetricId = "program-line-edit";

    /// <summary>The default line bound applied to each source before comparison.</summary>
    public const int DefaultMaxComparedLines = 2_000;

    /// <summary>Initializes a line edit distance.</summary>
    /// <param name="maxComparedLines">The per-source line bound; 1 to 100,000.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxComparedLines"/> is outside its permitted range.</exception>
    public ProgramLineEditDistance(int maxComparedLines = DefaultMaxComparedLines)
    {
        if (maxComparedLines < 1 || maxComparedLines > 100_000)
        {
            throw new ArgumentOutOfRangeException(nameof(maxComparedLines), maxComparedLines,
                "Value must be between 1 and 100000.");
        }

        MaxComparedLines = maxComparedLines;
    }

    /// <summary>Gets the per-source line bound applied before comparison.</summary>
    public int MaxComparedLines { get; }

    /// <inheritdoc/>
    public string Id => MetricId;

    /// <inheritdoc/>
    public string VersionHash =>
        MetricId + "-v1-" + MaxComparedLines.ToString(System.Globalization.CultureInfo.InvariantCulture);

    /// <inheritdoc/>
    public double Distance(ProgramGenome first, ProgramGenome second)
    {
        Guard.NotNull(first);
        Guard.NotNull(second);
        return Compute(first.NormalizedSource, second.NormalizedSource, MaxComparedLines);
    }

    /// <summary>Computes the normalized line edit distance between two program sources.</summary>
    /// <param name="first">The first source; normalized before comparison.</param>
    /// <param name="second">The second source; normalized before comparison.</param>
    /// <param name="maxComparedLines">The per-source line bound; 1 to 100,000.</param>
    /// <returns>Zero for identical line sequences, one when no line survives.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="first"/> or <paramref name="second"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxComparedLines"/> is outside its permitted range.</exception>
    public static double ComputeDistance(string first, string second, int maxComparedLines = DefaultMaxComparedLines)
    {
        Guard.NotNull(first);
        Guard.NotNull(second);
        if (maxComparedLines < 1 || maxComparedLines > 100_000)
        {
            throw new ArgumentOutOfRangeException(nameof(maxComparedLines), maxComparedLines,
                "Value must be between 1 and 100000.");
        }

        return Compute(ProgramText.Normalize(first), ProgramText.Normalize(second), maxComparedLines);
    }

    private static double Compute(string first, string second, int maxComparedLines)
    {
        if (string.Equals(first, second, StringComparison.Ordinal)) return 0.0;

        IReadOnlyList<string> firstLines = Bound(ProgramText.SplitLines(first), maxComparedLines);
        IReadOnlyList<string> secondLines = Bound(ProgramText.SplitLines(second), maxComparedLines);

        int longer = Math.Max(firstLines.Count, secondLines.Count);
        if (longer == 0) return 0.0;

        int edits = Levenshtein(firstLines, secondLines);
        double distance = (double)edits / longer;
        return distance < 0.0 ? 0.0 : distance > 1.0 ? 1.0 : distance;
    }

    private static IReadOnlyList<string> Bound(List<string> lines, int maxComparedLines) =>
        lines.Count <= maxComparedLines ? lines : lines.GetRange(0, maxComparedLines);

    private static int Levenshtein(IReadOnlyList<string> first, IReadOnlyList<string> second)
    {
        // Keep the shorter sequence on the row axis so the two working rows stay as small as possible.
        IReadOnlyList<string> rows = first.Count <= second.Count ? first : second;
        IReadOnlyList<string> columns = ReferenceEquals(rows, first) ? second : first;

        var previous = new int[rows.Count + 1];
        var current = new int[rows.Count + 1];
        for (int index = 0; index <= rows.Count; index++) previous[index] = index;

        for (int column = 1; column <= columns.Count; column++)
        {
            current[0] = column;
            string columnLine = columns[column - 1];
            for (int row = 1; row <= rows.Count; row++)
            {
                int substitution = previous[row - 1] +
                    (string.Equals(rows[row - 1], columnLine, StringComparison.Ordinal) ? 0 : 1);
                int deletion = previous[row] + 1;
                int insertion = current[row - 1] + 1;
                int best = substitution < deletion ? substitution : deletion;
                current[row] = best < insertion ? best : insertion;
            }

            int[] swap = previous;
            previous = current;
            current = swap;
        }

        return previous[rows.Count];
    }
}
