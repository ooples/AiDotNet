using AiDotNet.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>How a self-sorting memory bank decides what to keep.</summary>
public enum MemoryBankUpdateRule
{
    /// <summary>
    /// Medical SAM 2's rule (Zhu et al. 2024, eqs. 2-3): a new memory whose predicted IoU reaches the confidence threshold
    /// joins the bank's memories as candidates, and the K candidates with the highest total dissimilarity (the sum of one
    /// minus the cosine similarity to every other candidate) are kept.
    /// </summary>
    TopKDissimilarity,

    /// <summary>
    /// The official code's rule: the bank fills to K memories; after that, the new memory replaces the bank memory most
    /// similar to the one least similar to it, when it is less similar to that one than that pair and its predicted IoU is
    /// above the replaced memory's minus 0.1.
    /// </summary>
    ReplaceRedundant,
}

/// <summary>How a self-sorting memory bank draws the memories a frame attends to.</summary>
public enum MemoryBankResampling
{
    /// <summary>
    /// Medical SAM 2's eq. 4: K draws with replacement, each memory with probability its cosine similarity to the frame over
    /// the sum of all of them. A negative similarity makes that undefined, and it is reported as an error.
    /// </summary>
    NormalizedSimilarity,

    /// <summary>The official code's: K draws with replacement from the softmax of the cosine similarities.</summary>
    Softmax,
}

/// <summary>A self-sorting memory bank's settings.</summary>
internal sealed record Sam2BankSettings(int Capacity, double? ConfidenceThreshold, MemoryBankUpdateRule Rule, MemoryBankResampling Resampling);

/// <summary>
/// Medical SAM 2's self-sorting memory bank (Zhu et al. 2024, section 3.2): the memories a set of images (a pseudo-video)
/// keeps for the frames after them, chosen for confidence and diversity rather than recency, and resampled for each frame by
/// similarity. The memories are compared by their memory features, and a frame by its image features against the stored
/// memory-conditioned embeddings, as the official code compares them.
/// </summary>
internal sealed class Sam2SortedMemoryBank<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    internal sealed record Entry(Tensor<T> Memory, double[] MemoryKey, double[] EmbeddingKey, double Iou);

    private readonly Sam2BankSettings _settings;

    public Sam2SortedMemoryBank(Sam2BankSettings settings)
    {
        if (settings.Capacity < 1) throw new ArgumentOutOfRangeException(nameof(settings), "The bank holds at least one memory.");
        _settings = settings;
    }

    public List<Entry> Entries { get; } = new();

    public int Count => Entries.Count;

    /// <summary>The unit-length flattening of a tensor (the cosine similarity's operand).</summary>
    public static double[] Key(Tensor<T> tensor)
    {
        var values = tensor.ToArray().Select(NumOps.ToDouble).ToArray();
        double norm = Math.Sqrt(values.Sum(v => v * v));
        // F.normalize divides by max(norm, 1e-12).
        double scale = 1.0 / Math.Max(norm, 1e-12);
        for (int i = 0; i < values.Length; i++) values[i] *= scale;
        return values;
    }

    private static double Cosine(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    /// <summary>Offers a frame's memory <c>[1, MemoryDim, g, g]</c>, its memory-conditioned embedding and its predicted IoU to
    /// the bank.</summary>
    public void Add(Tensor<T> memory, Tensor<T> embedding, double iou)
    {
        var entry = new Entry(memory, Key(memory), Key(embedding), iou);
        int k = _settings.Capacity;
        if (_settings.Rule == MemoryBankUpdateRule.TopKDissimilarity)
        {
            if (_settings.ConfidenceThreshold is { } threshold && iou < threshold) return;
            var candidates = Entries.Concat(new[] { entry }).ToList();
            if (candidates.Count <= k)
            {
                Entries.Add(entry);
                return;
            }
            var dissimilarity = candidates.Select((c, i) => candidates.Where((_, j) => j != i).Sum(o => 1.0 - Cosine(c.MemoryKey, o.MemoryKey))).ToArray();
            var keep = Enumerable.Range(0, candidates.Count).OrderByDescending(i => dissimilarity[i]).ThenBy(i => i).Take(k).OrderBy(i => i).ToList();
            Entries.Clear();
            Entries.AddRange(keep.Select(i => candidates[i]));
            return;
        }
        if (Entries.Count < k)
        {
            Entries.Add(entry);
            return;
        }
        int n = Entries.Count;
        var similarity = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                similarity[i, j] = i == j ? double.NegativeInfinity : Cosine(Entries[i].MemoryKey, Entries[j].MemoryKey);
        var toNew = Entries.Select(e => Cosine(e.MemoryKey, entry.MemoryKey)).ToArray();
        int least = 0;
        for (int i = 1; i < n; i++) if (toNew[i] < toNew[least]) least = i;
        int redundant = 0;
        for (int j = 1; j < n; j++) if (similarity[least, j] > similarity[least, redundant]) redundant = j;
        if (toNew[least] < similarity[least, redundant] && iou > Entries[redundant].Iou - 0.1)
        {
            Entries.RemoveAt(redundant);
            Entries.Add(entry);
        }
    }

    /// <summary>The memories a frame with features <paramref name="features"/> attends to: <c>Capacity</c> draws with
    /// replacement by similarity.</summary>
    public List<Entry> Resample(Tensor<T> features, Random random)
    {
        if (Entries.Count == 0) throw new InvalidOperationException("The memory bank is empty.");
        var key = Key(features);
        var similarities = Entries.Select(e => Cosine(key, e.EmbeddingKey)).ToArray();
        double[] probabilities;
        if (_settings.Resampling == MemoryBankResampling.Softmax)
        {
            double max = similarities.Max();
            var exp = similarities.Select(s => Math.Exp(s - max)).ToArray();
            double total = exp.Sum();
            probabilities = exp.Select(e => e / total).ToArray();
        }
        else
        {
            if (similarities.Any(s => s < 0) || similarities.Sum() <= 0)
                throw new InvalidOperationException(
                    "A memory's cosine similarity to the frame is negative, where Medical SAM 2's resampling (eq. 4, the similarity over " +
                    "their sum) is undefined. Set MemoryResampling to Softmax to draw as the official code does.");
            double total = similarities.Sum();
            probabilities = similarities.Select(s => s / total).ToArray();
        }
        var drawn = new List<Entry>(_settings.Capacity);
        for (int d = 0; d < _settings.Capacity; d++)
        {
            double u = random.NextDouble(), cumulative = 0;
            int pick = probabilities.Length - 1;
            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (u < cumulative) { pick = i; break; }
            }
            drawn.Add(Entries[pick]);
        }
        return drawn;
    }
}
