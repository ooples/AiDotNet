using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// MbPA's episodic memory <c>M = {(h_i, v_i)}</c>: embeddings as keys, observed targets as values,
/// with K-nearest-neighbour retrieval by Euclidean distance.
/// </summary>
/// <remarks>
/// <para>
/// From "Memory-based Parameter Adaptation" (Sprechmann et al., arXiv:1802.10542): "Keys {h_i} are
/// given by the embedding network. The values {v_i} correspond to the desired output y_i", and
/// "Retrieval from the memory M uses K-nearest neighbour search on the keys {h_i} with Euclidean
/// distance to obtain the K most similar keys and associated values."
/// </para>
/// <para>
/// <b>For Beginners:</b> A notebook of "when the network saw something that looked like THIS, the
/// right answer was THAT". Looking something up means finding the few most similar entries.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class MbPAEpisodicMemory<T>
{
    private readonly List<(Vector<T> Key, Vector<T> Value)> _entries;
    private readonly int _capacity;

    /// <summary>Gets the number of stored pairs.</summary>
    public int Count => _entries.Count;

    /// <summary>Gets the maximum number of pairs held before eviction begins.</summary>
    public int Capacity => _capacity;

    /// <summary>
    /// Initializes an episodic memory.
    /// </summary>
    /// <param name="capacity">Maximum stored pairs; the oldest is evicted when full.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="capacity"/> is not positive.</exception>
    public MbPAEpisodicMemory(int capacity)
    {
        if (capacity <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Memory capacity must be positive.");
        }
        _capacity = capacity;
        _entries = new List<(Vector<T>, Vector<T>)>(Math.Min(capacity, 1024));
    }

    /// <summary>
    /// Stores one <c>(embedding, target)</c> pair, evicting the oldest when at capacity.
    /// </summary>
    public void Write(Vector<T> key, Vector<T> value)
    {
        if (_entries.Count >= _capacity) _entries.RemoveAt(0);
        _entries.Add((key, value));
    }

    /// <summary>Removes every stored pair.</summary>
    public void Clear() => _entries.Clear();

    /// <summary>
    /// Returns the K entries whose keys are nearest to <paramref name="query"/> in Euclidean
    /// distance, together with the kernel weights <c>w_k ~ 1 / (eps + ||h_k - q||^2)</c> normalized
    /// to sum to one.
    /// </summary>
    /// <remarks>
    /// The weights are the paper's <c>kern(h, q) = 1 / (eps + ||h - q||_2^2)</c>. A softmax over
    /// similarities would be a different function: this one falls off with the SQUARE of the
    /// distance and has no temperature, so a neighbour twice as far away counts a quarter as much
    /// regardless of the overall scale of the embedding space.
    /// </remarks>
    /// <param name="query">The query embedding q.</param>
    /// <param name="k">How many neighbours to retrieve; clamped to the number stored.</param>
    /// <param name="epsilon">Kernel epsilon, keeping the weight finite at zero distance.</param>
    /// <param name="toDouble">Converts a stored numeric to double for the distance computation.</param>
    public IReadOnlyList<(Vector<T> Key, Vector<T> Value, double Weight)> Retrieve(
        Vector<T> query, int k, double epsilon, Func<T, double> toDouble)
    {
        var empty = Array.Empty<(Vector<T>, Vector<T>, double)>();
        if (_entries.Count == 0 || k <= 0) return empty;

        int take = Math.Min(k, _entries.Count);

        var distances = new (int Index, double DistanceSquared)[_entries.Count];
        for (int i = 0; i < _entries.Count; i++)
        {
            var key = _entries[i].Key;
            double sum = 0.0;
            int len = Math.Min(key.Length, query.Length);
            for (int d = 0; d < len; d++)
            {
                double diff = toDouble(key[d]) - toDouble(query[d]);
                sum += diff * diff;
            }
            distances[i] = (i, sum);
        }

        // PARTIAL SELECTION, NOT A FULL SORT. Retrieve runs once per MbPA prediction per batch item,
        // and LocallyAdapt SUMS over the neighbours -- it never needs them ordered. A full sort cost
        // O(n log n) in the memory size on every prediction to produce an ordering nothing reads.
        //
        // A bounded max-heap of size `take` keeps the running k-smallest in O(n log k): the root is
        // the worst kept candidate, so a new entry is compared against it and either discarded or
        // swapped in. For the usual k << MemorySize that is close to a single linear scan.
        var kept = new List<(int Index, double DistanceSquared)>(take);

        void SiftUp(int child)
        {
            while (child > 0)
            {
                int parent = (child - 1) / 2;
                if (kept[parent].DistanceSquared >= kept[child].DistanceSquared) break;
                (kept[parent], kept[child]) = (kept[child], kept[parent]);
                child = parent;
            }
        }

        void SiftDown(int parent)
        {
            while (true)
            {
                int left = (2 * parent) + 1;
                if (left >= kept.Count) break;
                int worst = left;
                int right = left + 1;
                if (right < kept.Count && kept[right].DistanceSquared > kept[left].DistanceSquared) worst = right;
                if (kept[parent].DistanceSquared >= kept[worst].DistanceSquared) break;
                (kept[parent], kept[worst]) = (kept[worst], kept[parent]);
                parent = worst;
            }
        }

        for (int i = 0; i < distances.Length; i++)
        {
            if (kept.Count < take)
            {
                kept.Add(distances[i]);
                SiftUp(kept.Count - 1);
            }
            else if (distances[i].DistanceSquared < kept[0].DistanceSquared)
            {
                kept[0] = distances[i];
                SiftDown(0);
            }
        }

        for (int i = 0; i < take; i++) distances[i] = kept[i];

        var kernels = new double[take];
        double total = 0.0;
        for (int i = 0; i < take; i++)
        {
            kernels[i] = 1.0 / (epsilon + distances[i].DistanceSquared);
            total += kernels[i];
        }

        var result = new (Vector<T>, Vector<T>, double)[take];
        for (int i = 0; i < take; i++)
        {
            var entry = _entries[distances[i].Index];
            result[i] = (entry.Key, entry.Value, total > 0.0 ? kernels[i] / total : 1.0 / take);
        }
        return result;
    }
}
