using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Training;

/// <summary>Reusable device-side snapshots of live dense tensors, independent of model inheritance.</summary>
/// <typeparam name="T">The tensor element type.</typeparam>
public static class TensorCheckpoint<T>
{
    /// <summary>Copies tensors without recording gradients; reuses a compatible snapshot's storage.</summary>
    public static Tensor<T>[] Capture(IReadOnlyList<Tensor<T>> tensors, Tensor<T>[]? snapshot = null)
    {
        if (tensors is null) throw new ArgumentNullException(nameof(tensors));
        ValidateLive(tensors);
        if (snapshot is not null) Validate(tensors, snapshot);
        using var noArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        snapshot ??= tensors.Select(t => new Tensor<T>(t.Shape.ToArray())).ToArray();
        for (int i = 0; i < tensors.Count; i++) engine.TensorCopy(tensors[i], snapshot[i]);
        return snapshot;
    }

    /// <summary>Restores values in place, preserving live tensor identities and invalidating cached weights.</summary>
    public static void Restore(IReadOnlyList<Tensor<T>> tensors, Tensor<T>[] snapshot)
    {
        if (tensors is null) throw new ArgumentNullException(nameof(tensors));
        if (snapshot is null) throw new ArgumentNullException(nameof(snapshot));
        Validate(tensors, snapshot);
        for (int i = 0; i < tensors.Count; i++) CopyValues(snapshot[i], tensors[i]);
    }

    /// <summary>Copies an already validated dense chunk, retaining the destination's shape and identity.</summary>
    internal static void CopyValues(Tensor<T> source, Tensor<T> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Checkpoint chunk length differs from its destination.", nameof(source));
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        engine.TensorCopy(SameShape(source, destination) ? source : engine.Reshape(source, destination.Shape.ToArray()), destination);
        AiDotNet.Tensors.Engines.InferenceWeightCache.Invalidate(destination.GetLiveBackingArrayOrNull());
    }

    private static bool SameShape(Tensor<T> left, Tensor<T> right)
    {
        if (left.Rank != right.Rank) return false;
        for (int i = 0; i < left.Rank; i++)
            if (left.Shape[i] != right.Shape[i]) return false;
        return true;
    }

    private static void ValidateLive(IReadOnlyList<Tensor<T>> tensors)
    {
        for (int i = 0; i < tensors.Count; i++)
            if (tensors[i] is null || tensors[i] is SparseTensor<T>)
                throw new ArgumentException("Checkpoints require non-null dense tensors.", nameof(tensors));
    }

    private static void Validate(IReadOnlyList<Tensor<T>> tensors, Tensor<T>[] snapshot)
    {
        if (tensors.Count != snapshot.Length)
            throw new ArgumentException("Checkpoint tensor count differs from the live tensor count.", nameof(snapshot));
        ValidateLive(tensors);
        var live = new HashSet<Tensor<T>>(tensors, TensorReferenceComparer<Tensor<T>>.Instance);
        var saved = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        // Validate the entire layout before copying so a mismatch cannot partially restore a model.
        for (int i = 0; i < tensors.Count; i++)
        {
            if (snapshot[i] is null || snapshot[i] is SparseTensor<T> || !SameShape(tensors[i], snapshot[i]))
                throw new ArgumentException($"Checkpoint shape differs at tensor {i}.", nameof(snapshot));
            if (live.Contains(snapshot[i]) || !saved.Add(snapshot[i])
                || tensors.Any(t => snapshot[i].SharesStorageWith(t))
                || snapshot.Take(i).Any(t => snapshot[i].SharesStorageWith(t)))
                throw new ArgumentException("Checkpoint storage must be independent of live tensors and other snapshot entries.", nameof(snapshot));
        }
    }
}
