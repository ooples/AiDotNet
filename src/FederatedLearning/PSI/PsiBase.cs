using System.Diagnostics;
using System.Text.RegularExpressions;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Base class providing shared functionality for PSI protocol implementations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This base class handles common tasks like input validation,
/// building alignment mappings from intersection results, and timing execution.
/// Each concrete PSI protocol only needs to implement the core intersection algorithm.</para>
/// </remarks>
public abstract class PsiBase : IPrivateSetIntersection
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    /// <inheritdoc/>
    public abstract string ProtocolName { get; }

    /// <inheritdoc/>
    public PsiResult ComputeIntersection(IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        ValidateInputs(localIds, remoteIds, options);

        var stopwatch = Stopwatch.StartNew();

        PsiResult result;
        if (options.FuzzyMatch is not null && options.FuzzyMatch.Strategy != FuzzyMatchStrategy.Exact)
        {
            result = ComputeFuzzyIntersection(localIds, remoteIds, options);
        }
        else
        {
            result = ComputeExactIntersection(localIds, remoteIds, options);
        }

        stopwatch.Stop();
        result.ExecutionTime = stopwatch.Elapsed;
        result.ProtocolUsed = options.Protocol;
        result.LocalOverlapRatio = localIds.Count > 0 ? (double)result.IntersectionSize / localIds.Count : 0.0;
        result.RemoteOverlapRatio = remoteIds.Count > 0 ? (double)result.IntersectionSize / remoteIds.Count : 0.0;

        return result;
    }

    /// <inheritdoc/>
    public int ComputeCardinality(IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        ValidateInputs(localIds, remoteIds, options);

        if (options.FuzzyMatch is not null && options.FuzzyMatch.Strategy != FuzzyMatchStrategy.Exact)
        {
            var result = ComputeFuzzyIntersection(localIds, remoteIds, options);
            return result.IntersectionSize;
        }

        return ComputeExactCardinality(localIds, remoteIds, options);
    }

    /// <summary>
    /// Computes the exact intersection using the protocol-specific algorithm.
    /// </summary>
    /// <param name="localIds">The local party's IDs.</param>
    /// <param name="remoteIds">The remote party's IDs.</param>
    /// <param name="options">Protocol options.</param>
    /// <returns>The PSI result with intersection and alignment mappings.</returns>
    protected abstract PsiResult ComputeExactIntersection(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options);

    /// <summary>
    /// Computes the exact cardinality using the protocol-specific algorithm.
    /// Override for protocols that can compute cardinality more efficiently than full intersection.
    /// </summary>
    protected virtual int ComputeExactCardinality(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        var result = ComputeExactIntersection(localIds, remoteIds, options);
        return result.IntersectionSize;
    }

    /// <summary>
    /// Computes fuzzy intersection by delegating to the appropriate fuzzy matcher.
    /// </summary>
    protected PsiResult ComputeFuzzyIntersection(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        var fuzzyOptions = options.FuzzyMatch ?? new FuzzyMatchOptions();
        var matcher = CreateFuzzyMatcher(fuzzyOptions.Strategy);

        var normalizedLocal = new string[localIds.Count];
        for (int i = 0; i < localIds.Count; i++)
        {
            normalizedLocal[i] = matcher.Normalize(localIds[i], fuzzyOptions);
        }

        var normalizedRemote = new string[remoteIds.Count];
        for (int i = 0; i < remoteIds.Count; i++)
        {
            normalizedRemote[i] = matcher.Normalize(remoteIds[i], fuzzyOptions);
        }

        var intersectionIds = new List<string>();
        var localToShared = new Dictionary<int, int>();
        var remoteToShared = new Dictionary<int, int>();
        var confidences = new Dictionary<int, double>();
        var usedRemoteIndices = new HashSet<int>();
        int sharedIndex = 0;

        for (int i = 0; i < normalizedLocal.Length; i++)
        {
            var matches = matcher.FindMatches(normalizedLocal[i], normalizedRemote, fuzzyOptions);
            foreach (var (candidateIndex, similarity) in matches)
            {
                if (usedRemoteIndices.Contains(candidateIndex))
                {
                    continue;
                }

                intersectionIds.Add(localIds[i]);
                localToShared[i] = sharedIndex;
                remoteToShared[candidateIndex] = sharedIndex;
                confidences[sharedIndex] = similarity;
                usedRemoteIndices.Add(candidateIndex);
                sharedIndex++;
                break;
            }
        }

        return new PsiResult
        {
            IntersectionIds = intersectionIds,
            IntersectionSize = intersectionIds.Count,
            LocalToSharedIndexMap = localToShared,
            RemoteToSharedIndexMap = remoteToShared,
            IsFuzzyMatch = true,
            FuzzyMatchConfidences = confidences
        };
    }

    /// <summary>
    /// Builds alignment mappings from a set of intersection IDs and the original ID lists.
    /// </summary>
    protected static PsiResult BuildAlignmentResult(
        IReadOnlyList<string> localIds,
        IReadOnlyList<string> remoteIds,
        IReadOnlyList<string> intersectionIds)
    {
        var localIndex = new Dictionary<string, int>(localIds.Count, StringComparer.Ordinal);
        for (int i = 0; i < localIds.Count; i++)
        {
            if (!localIndex.ContainsKey(localIds[i]))
            {
                localIndex[localIds[i]] = i;
            }
        }

        var remoteIndex = new Dictionary<string, int>(remoteIds.Count, StringComparer.Ordinal);
        for (int i = 0; i < remoteIds.Count; i++)
        {
            if (!remoteIndex.ContainsKey(remoteIds[i]))
            {
                remoteIndex[remoteIds[i]] = i;
            }
        }

        var localToShared = new Dictionary<int, int>(intersectionIds.Count);
        var remoteToShared = new Dictionary<int, int>(intersectionIds.Count);

        for (int sharedIdx = 0; sharedIdx < intersectionIds.Count; sharedIdx++)
        {
            string id = intersectionIds[sharedIdx];
            if (localIndex.TryGetValue(id, out int localIdx))
            {
                localToShared[localIdx] = sharedIdx;
            }
            if (remoteIndex.TryGetValue(id, out int remoteIdx))
            {
                remoteToShared[remoteIdx] = sharedIdx;
            }
        }

        return new PsiResult
        {
            IntersectionIds = intersectionIds,
            IntersectionSize = intersectionIds.Count,
            LocalToSharedIndexMap = localToShared,
            RemoteToSharedIndexMap = remoteToShared
        };
    }

    /// <summary>
    /// Creates a fuzzy matcher for the given strategy.
    /// </summary>
    protected static IFuzzyMatcher CreateFuzzyMatcher(FuzzyMatchStrategy strategy)
    {
        return strategy switch
        {
            FuzzyMatchStrategy.Exact => new ExactMatcher(),
            FuzzyMatchStrategy.EditDistance => new EditDistanceMatcher(),
            FuzzyMatchStrategy.Phonetic => new PhoneticMatcher(),
            FuzzyMatchStrategy.NGram => new NGramMatcher(),
            FuzzyMatchStrategy.Jaccard => new JaccardMatcher(),
            _ => new ExactMatcher()
        };
    }

    /// <summary>
    /// Normalizes whitespace in a string by collapsing runs of whitespace to single spaces and trimming.
    /// </summary>
    internal static string NormalizeWhitespace(string input)
    {
        if (string.IsNullOrEmpty(input))
        {
            return input;
        }

        return Regex.Replace(input.Trim(), @"\s+", " ", RegexOptions.None, RegexTimeout);
    }

    private static void ValidateInputs(IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        if (localIds is null)
        {
            throw new ArgumentNullException(nameof(localIds));
        }

        if (remoteIds is null)
        {
            throw new ArgumentNullException(nameof(remoteIds));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (options.SecurityParameter != 128 && options.SecurityParameter != 256)
        {
            throw new ArgumentException(
                "Security parameter must be 128 or 256 bits.", nameof(options));
        }
    }
}
