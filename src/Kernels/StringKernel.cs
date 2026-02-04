namespace AiDotNet.Kernels;

/// <summary>
/// Implements various string kernels for comparing text/sequence data in Gaussian Processes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> String kernels allow Gaussian Processes to work directly with
/// text or sequence data without needing to convert them to fixed-length feature vectors.
///
/// The key insight: We can define a kernel (similarity measure) between strings that
/// captures meaningful notions of similarity:
/// - Spectrum kernel: Counts shared substrings
/// - Subsequence kernel: Counts shared (possibly non-contiguous) subsequences
/// - Edit distance kernel: Based on how many edits to transform one string to another
///
/// Applications:
/// - Text classification (spam detection, sentiment analysis)
/// - Bioinformatics (DNA/protein sequence comparison)
/// - Natural language processing
/// - Any domain with sequential/string data
/// </para>
/// <para>
/// <b>Note:</b> This class does NOT implement IKernelFunction&lt;T&gt; by design. Unlike numeric
/// kernels that operate on Vector&lt;T&gt; feature vectors, string kernels operate directly on
/// text data. To use string kernels with standard GP models, use this class to compute a
/// kernel matrix from your text data, then use that matrix with a custom kernel implementation.
/// </para>
/// </remarks>
public class StringKernel<T>
{
    /// <summary>
    /// The type of string kernel to use.
    /// </summary>
    public enum KernelType
    {
        /// <summary>
        /// Spectrum kernel: counts shared k-mers (substrings of length k).
        /// </summary>
        Spectrum,

        /// <summary>
        /// Subsequence kernel: counts shared subsequences with gap penalty.
        /// </summary>
        Subsequence,

        /// <summary>
        /// Normalized edit distance kernel: exp(-d(s,t)/scale).
        /// </summary>
        EditDistance,

        /// <summary>
        /// Bag of words kernel: treats strings as bags of words.
        /// </summary>
        BagOfWords
    }

    private readonly KernelType _kernelType;
    private readonly int _substringLength;
    private readonly double _lambda;
    private readonly double _scale;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new string kernel.
    /// </summary>
    /// <param name="kernelType">The type of string kernel to use.</param>
    /// <param name="substringLength">Length of substrings for Spectrum kernel (k). Default is 3.</param>
    /// <param name="lambda">Decay parameter for Subsequence kernel. Default is 0.5.</param>
    /// <param name="scale">Scale parameter for EditDistance kernel. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a string kernel with specified parameters.
    ///
    /// For Spectrum kernel:
    /// - substringLength (k) determines what size patterns to look for
    /// - k=3 looks for triplets like "the", "he ", "e w", etc.
    /// - Larger k = more specific matching
    ///
    /// For Subsequence kernel:
    /// - lambda controls gap penalty (how much non-contiguous matches are penalized)
    /// - lambda close to 1 = gaps not penalized much
    /// - lambda close to 0 = prefer contiguous matches
    ///
    /// For EditDistance kernel:
    /// - scale controls how quickly similarity decreases with edit distance
    /// - Larger scale = more tolerant of differences
    /// </para>
    /// </remarks>
    public StringKernel(
        KernelType kernelType = KernelType.Spectrum,
        int substringLength = 3,
        double lambda = 0.5,
        double scale = 1.0)
    {
        if (substringLength < 1)
            throw new ArgumentException("Substring length must be at least 1.", nameof(substringLength));
        if (lambda <= 0 || lambda > 1)
            throw new ArgumentException("Lambda must be in (0, 1].", nameof(lambda));
        if (scale <= 0)
            throw new ArgumentException("Scale must be positive.", nameof(scale));

        _kernelType = kernelType;
        _substringLength = substringLength;
        _lambda = lambda;
        _scale = scale;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the kernel type.
    /// </summary>
    public KernelType Type => _kernelType;

    /// <summary>
    /// Calculates the string kernel value between two strings.
    /// </summary>
    /// <param name="s1">The first string.</param>
    /// <param name="s2">The second string.</param>
    /// <returns>The kernel value (similarity measure).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes a similarity score between two strings.
    ///
    /// Higher values mean more similar strings.
    /// The exact meaning depends on the kernel type:
    /// - Spectrum: More shared substrings = higher value
    /// - Subsequence: More shared patterns (with gaps) = higher value
    /// - EditDistance: Fewer edits needed = higher value
    /// - BagOfWords: More shared words = higher value
    /// </para>
    /// </remarks>
    public T Calculate(string s1, string s2)
    {
        if (s1 is null) throw new ArgumentNullException(nameof(s1));
        if (s2 is null) throw new ArgumentNullException(nameof(s2));

        double result = _kernelType switch
        {
            KernelType.Spectrum => CalculateSpectrumKernel(s1, s2),
            KernelType.Subsequence => CalculateSubsequenceKernel(s1, s2),
            KernelType.EditDistance => CalculateEditDistanceKernel(s1, s2),
            KernelType.BagOfWords => CalculateBagOfWordsKernel(s1, s2),
            _ => throw new InvalidOperationException($"Unknown kernel type: {_kernelType}")
        };

        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Calculates the spectrum (k-mer) kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The spectrum kernel works by:
    /// 1. Extracting all substrings of length k from each string
    /// 2. Counting how many times each substring appears
    /// 3. Computing dot product of these count vectors
    ///
    /// Example with k=2:
    /// "hello" → {"he":1, "el":1, "ll":1, "lo":1}
    /// "help"  → {"he":1, "el":1, "lp":1}
    /// Kernel = 1×1 + 1×1 = 2 (for "he" and "el")
    ///
    /// The result is normalized to give values in [0, 1].
    /// </para>
    /// </remarks>
    private double CalculateSpectrumKernel(string s1, string s2)
    {
        if (s1.Length < _substringLength || s2.Length < _substringLength)
        {
            return s1 == s2 ? 1.0 : 0.0;
        }

        // Count k-mers in each string
        var counts1 = GetKmerCounts(s1);
        var counts2 = GetKmerCounts(s2);

        // Compute dot product
        double dotProduct = 0;
        foreach (var kvp in counts1)
        {
            if (counts2.TryGetValue(kvp.Key, out int count2))
            {
                dotProduct += kvp.Value * count2;
            }
        }

        // Compute norms for normalization
        double norm1 = 0, norm2 = 0;
        foreach (var count in counts1.Values)
        {
            norm1 += count * count;
        }
        foreach (var count in counts2.Values)
        {
            norm2 += count * count;
        }

        norm1 = Math.Sqrt(norm1);
        norm2 = Math.Sqrt(norm2);

        if (norm1 < 1e-10 || norm2 < 1e-10)
        {
            return 0.0;
        }

        return dotProduct / (norm1 * norm2);
    }

    /// <summary>
    /// Gets k-mer counts for a string.
    /// </summary>
    private Dictionary<string, int> GetKmerCounts(string s)
    {
        var counts = new Dictionary<string, int>();

        for (int i = 0; i <= s.Length - _substringLength; i++)
        {
            string kmer = s.Substring(i, _substringLength);
            if (counts.TryGetValue(kmer, out int existing))
            {
                counts[kmer] = existing + 1;
            }
            else
            {
                counts[kmer] = 1;
            }
        }

        return counts;
    }

    /// <summary>
    /// Calculates the subsequence kernel with gap penalty.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The subsequence kernel counts shared subsequences,
    /// where subsequences don't have to be contiguous.
    ///
    /// Example: "cat" and "cart" share subsequence "cat" (c-a-t with gap at 'r')
    ///
    /// The gap penalty λ controls how much non-contiguous matches count:
    /// - Contiguous match contributes λ^k (where k is length)
    /// - Gap of g characters contributes extra λ^g penalty
    ///
    /// This kernel is more flexible than spectrum kernel because it allows
    /// for insertions/deletions in the middle of patterns.
    /// </para>
    /// </remarks>
    private double CalculateSubsequenceKernel(string s1, string s2)
    {
        int n = s1.Length;
        int m = s2.Length;
        int k = _substringLength;

        if (n == 0 || m == 0)
        {
            return 0.0;
        }

        // Use dynamic programming
        // K[i,j,l] = sum over all common subsequences of length l ending at s1[i] and s2[j]
        // Simplified: compute for fixed k

        // K'[i,j] = partial computation
        var Kp = new double[n + 1, m + 1];
        var K = new double[n + 1, m + 1];

        // Initialize
        for (int i = 0; i <= n; i++)
        {
            for (int j = 0; j <= m; j++)
            {
                Kp[i, j] = 1.0; // Empty subsequence
            }
        }

        // Dynamic programming for each length
        for (int l = 1; l <= k; l++)
        {
            var newK = new double[n + 1, m + 1];

            for (int i = l; i <= n; i++)
            {
                double Kpp = 0;
                for (int j = l; j <= m; j++)
                {
                    Kpp = _lambda * (Kpp + _lambda * (s1[i - 1] == s2[j - 1] ? 1 : 0) * Kp[i - 1, j - 1]);
                    newK[i, j] = newK[i - 1, j] * _lambda + Kpp;
                }
            }

            // Update Kp for next iteration
            for (int i = 0; i <= n; i++)
            {
                for (int j = 0; j <= m; j++)
                {
                    Kp[i, j] = newK[i, j];
                }
            }
            K = newK;
        }

        double result = K[n, m];

        // Normalize
        double k11 = CalculateSubsequenceKernelSelf(s1);
        double k22 = CalculateSubsequenceKernelSelf(s2);

        if (k11 < 1e-10 || k22 < 1e-10)
        {
            return 0.0;
        }

        return result / Math.Sqrt(k11 * k22);
    }

    /// <summary>
    /// Calculates subsequence kernel of a string with itself (for normalization).
    /// </summary>
    private double CalculateSubsequenceKernelSelf(string s)
    {
        int n = s.Length;
        int k = _substringLength;

        if (n < k)
        {
            return 0.0;
        }

        // Simplified: count k-length contiguous subsequences
        double result = 0;
        for (int i = 0; i <= n - k; i++)
        {
            result += Math.Pow(_lambda, 2 * k);
        }

        return Math.Max(result, 1e-10);
    }

    /// <summary>
    /// Calculates the edit distance (Levenshtein) kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The edit distance kernel is based on how many
    /// character operations (insert, delete, substitute) are needed to
    /// transform one string into another.
    ///
    /// k(s1, s2) = exp(-d(s1, s2) / scale)
    ///
    /// Where d is the edit distance:
    /// - d("cat", "cat") = 0 → k = 1.0 (identical)
    /// - d("cat", "car") = 1 → k = exp(-1/scale)
    /// - d("cat", "dog") = 3 → k = exp(-3/scale)
    ///
    /// This kernel is intuitive: strings that require fewer edits are more similar.
    /// </para>
    /// </remarks>
    private double CalculateEditDistanceKernel(string s1, string s2)
    {
        int editDist = ComputeEditDistance(s1, s2);
        return Math.Exp(-editDist / _scale);
    }

    /// <summary>
    /// Computes the Levenshtein edit distance between two strings.
    /// </summary>
    private static int ComputeEditDistance(string s1, string s2)
    {
        int n = s1.Length;
        int m = s2.Length;

        if (n == 0) return m;
        if (m == 0) return n;

        // Use two rows instead of full matrix for space efficiency
        var prev = new int[m + 1];
        var curr = new int[m + 1];

        for (int j = 0; j <= m; j++)
        {
            prev[j] = j;
        }

        for (int i = 1; i <= n; i++)
        {
            curr[0] = i;

            for (int j = 1; j <= m; j++)
            {
                int cost = s1[i - 1] == s2[j - 1] ? 0 : 1;
                curr[j] = Math.Min(Math.Min(
                    prev[j] + 1,      // deletion
                    curr[j - 1] + 1), // insertion
                    prev[j - 1] + cost); // substitution
            }

            // Swap rows
            (prev, curr) = (curr, prev);
        }

        return prev[m];
    }

    /// <summary>
    /// Calculates the bag of words kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The bag of words kernel treats each string as a
    /// collection of words (ignoring order) and computes similarity based on
    /// shared words.
    ///
    /// Steps:
    /// 1. Split each string into words
    /// 2. Count word frequencies in each string
    /// 3. Compute normalized dot product of frequency vectors
    ///
    /// Example:
    /// "the cat sat" → {the:1, cat:1, sat:1}
    /// "the cat ran" → {the:1, cat:1, ran:1}
    /// Shared words: "the", "cat" → similarity = 2/3
    ///
    /// This is a simple but effective kernel for text classification.
    /// </para>
    /// </remarks>
    private double CalculateBagOfWordsKernel(string s1, string s2)
    {
        var words1 = GetWordCounts(s1);
        var words2 = GetWordCounts(s2);

        if (words1.Count == 0 || words2.Count == 0)
        {
            return 0.0;
        }

        // Compute dot product
        double dotProduct = 0;
        foreach (var kvp in words1)
        {
            if (words2.TryGetValue(kvp.Key, out int count2))
            {
                dotProduct += kvp.Value * count2;
            }
        }

        // Compute norms
        double norm1 = 0, norm2 = 0;
        foreach (var count in words1.Values)
        {
            norm1 += count * count;
        }
        foreach (var count in words2.Values)
        {
            norm2 += count * count;
        }

        norm1 = Math.Sqrt(norm1);
        norm2 = Math.Sqrt(norm2);

        if (norm1 < 1e-10 || norm2 < 1e-10)
        {
            return 0.0;
        }

        return dotProduct / (norm1 * norm2);
    }

    /// <summary>
    /// Gets word counts for a string.
    /// </summary>
    private static Dictionary<string, int> GetWordCounts(string s)
    {
        var counts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        // Simple word splitting (split on whitespace and punctuation)
        var words = s.Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}' },
            StringSplitOptions.RemoveEmptyEntries);

        foreach (var word in words)
        {
            string normalized = word.ToLowerInvariant();
            if (normalized.Length > 0)
            {
                if (counts.TryGetValue(normalized, out int existing))
                {
                    counts[normalized] = existing + 1;
                }
                else
                {
                    counts[normalized] = 1;
                }
            }
        }

        return counts;
    }

    /// <summary>
    /// Creates a kernel matrix for a collection of strings.
    /// </summary>
    /// <param name="strings">The collection of strings.</param>
    /// <returns>The kernel matrix (N x N).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes pairwise similarities between all strings.
    ///
    /// The result is a symmetric matrix where K[i,j] is the kernel value
    /// between strings[i] and strings[j].
    ///
    /// This matrix can be used directly with kernel methods like kernel PCA,
    /// kernel k-means, or support vector machines.
    /// </para>
    /// </remarks>
    public Matrix<T> ComputeKernelMatrix(string[] strings)
    {
        if (strings is null) throw new ArgumentNullException(nameof(strings));

        int n = strings.Length;
        var K = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            K[i, i] = _numOps.One; // Normalized kernel has k(x,x) = 1

            for (int j = i + 1; j < n; j++)
            {
                T value = Calculate(strings[i], strings[j]);
                K[i, j] = value;
                K[j, i] = value;
            }
        }

        return K;
    }
}
