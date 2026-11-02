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
    // KeyParameters: the embedding parameters the key was computed with, or null for a pair stored through the public
    // Write, whose key the caller computed (taken to be the embedding the reading model was adapted with).
    private readonly List<(Vector<T> Key, Vector<T> Value, object? Source, Vector<T>? KeyParameters)> _entries;
    private readonly int _capacity;

    /// <summary>
    /// Advances on every write or clear, so a view built from this memory can tell it is out of date.
    /// </summary>
    internal long Version { get; private set; }

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
        _entries = new List<(Vector<T>, Vector<T>, object?, Vector<T>?)>(Math.Min(capacity, 1024));
    }

    /// <summary>
    /// Stores one <c>(embedding, target)</c> pair, evicting the oldest when at capacity.
    /// </summary>
    public void Write(Vector<T> key, Vector<T> value) => Write(key, value, null, null);

    /// <summary>
    /// Stores one pair together with the example its key was embedded from and the embedding parameters that embedded
    /// it, so a reader with a different embedding network can tell the key is not its own and recompute it.
    /// </summary>
    internal void Write(Vector<T> key, Vector<T> value, object? source, Vector<T>? keyParameters)
    {
        if (_entries.Count >= _capacity) _entries.RemoveAt(0);
        _entries.Add((key, value, source, keyParameters));
        Version++;
    }

    /// <summary>Removes every stored pair.</summary>
    public void Clear()
    {
        _entries.Clear();
        Version++;
    }

    /// <summary>
    /// This memory as seen by an embedding network with <paramref name="current"/> parameters: itself when every key
    /// was computed with those parameters, else a copy in which each key computed with other parameters is recomputed
    /// from its source example by <paramref name="embed"/> (matching keys and all values are kept).
    /// </summary>
    /// <remarks>
    /// Keys are only comparable with queries from the embedding network that produced them, and one memory collects
    /// keys from whatever network wrote each entry: the meta-learner keeps training and writing after a model was
    /// adapted from it, and an adapted model's own parameters can be changed. Each entry is checked, not the memory.
    /// </remarks>
    /// <param name="current">The reading embedding network's parameters.</param>
    /// <param name="unstamped">The parameters a pair stored through the public <see cref="Write(Vector{T}, Vector{T})"/>
    /// is taken to have been embedded with.</param>
    /// <param name="sameParameters">Whether two parameter vectors are equal.</param>
    /// <param name="embed">Recomputes a key from its source example with the reading network.</param>
    /// <exception cref="InvalidOperationException">A key to recompute was written without its source example.</exception>
    internal MbPAEpisodicMemory<T> KeyedFor(Vector<T> current, Vector<T> unstamped,
        Func<Vector<T>, Vector<T>, bool> sameParameters, Func<object, Vector<T>> embed)
    {
        // One verdict per distinct parameter vector: entries written by one WriteToMemory call share theirs.
        var verdicts = new List<(Vector<T> Parameters, bool Current)>();
        bool IsCurrent(Vector<T>? keyParameters)
        {
            var parameters = keyParameters ?? unstamped;
            foreach (var (seen, verdict) in verdicts)
            {
                if (ReferenceEquals(seen, parameters)) return verdict;
            }
            bool same = sameParameters(parameters, current);
            verdicts.Add((parameters, same));
            return same;
        }

        bool allCurrent = true;
        foreach (var entry in _entries)
        {
            if (!IsCurrent(entry.KeyParameters))
            {
                allCurrent = false;
                break;
            }
        }
        if (allCurrent) return this;

        var copy = new MbPAEpisodicMemory<T>(_capacity);
        foreach (var (key, value, source, keyParameters) in _entries)
        {
            if (IsCurrent(keyParameters))
            {
                copy.Write(key, value, source, current);
                continue;
            }
            if (source is null)
            {
                throw new InvalidOperationException(
                    "An episodic-memory entry was written without its source example, so its key cannot be recomputed " +
                    "for a model whose embedding network differs from the one that wrote it. Write entries through " +
                    "MbPAAlgorithm.WriteToMemory.");
            }
            copy.Write(embed(source), value, source, current);
        }
        return copy;
    }

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

        // NEAREST FIRST. `kept` is a MAX-heap, so kept[0] is the FARTHEST of the k selected and the
        // remainder sit in heap order rather than distance order. Copying it out as-is made a method
        // documented as returning "the k nearest" return a result that was not fully ordered, with
        // the farthest survivor first. Kernels still matched their entries; only the public ordering
        // was incorrect.
        //
        // The selection above is still the O(n log k) partial scan the comment describes -- ordering
        // only the k SURVIVORS costs O(k log k) on a set that is already tiny (k << MemorySize), so
        // the reason the sort was skipped ("LocallyAdapt sums over the neighbours, it never needs
        // them ordered") holds for that one caller but not for the public contract every other
        // caller reads.
        kept.Sort(static (a, b) => a.DistanceSquared.CompareTo(b.DistanceSquared));
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
