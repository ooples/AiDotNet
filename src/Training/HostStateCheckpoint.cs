namespace AiDotNet.Training;

/// <summary>Reusable snapshots for models whose canonical state is host matrices and vectors.</summary>
internal static class HostStateCheckpoint<T>
{
    internal static Matrix<T> Capture(Matrix<T> source, Matrix<T>? snapshot = null)
    {
        if (snapshot is not null && (source.Rows != snapshot.Rows || source.Columns != snapshot.Columns))
            throw new ArgumentException("Checkpoint matrix shape differs from live state.", nameof(snapshot));
        snapshot ??= new Matrix<T>(source.Rows, source.Columns);
        source.AsSpan().CopyTo(snapshot.AsWritableSpan());
        return snapshot;
    }

    internal static Vector<T> Capture(Vector<T> source, Vector<T>? snapshot = null)
    {
        if (snapshot is not null && source.Length != snapshot.Length)
            throw new ArgumentException("Checkpoint vector shape differs from live state.", nameof(snapshot));
        snapshot ??= new Vector<T>(source.Length);
        source.AsSpan().CopyTo(snapshot.AsWritableSpan());
        return snapshot;
    }

    internal static List<Matrix<T>> Capture(IReadOnlyList<Matrix<T>> source, List<Matrix<T>>? snapshot = null)
    {
        if (snapshot is not null && (source.Count != snapshot.Count || source.Where((t, i) =>
            t.Rows != snapshot[i].Rows || t.Columns != snapshot[i].Columns).Any()))
            throw new ArgumentException("Checkpoint matrix layout differs from live state.", nameof(snapshot));
        if (snapshot is null) return source.Select(t => Capture(t)).ToList();
        for (int i = 0; i < source.Count; i++) Capture(source[i], snapshot[i]);
        return snapshot;
    }

    internal static List<Vector<T>> Capture(IReadOnlyList<Vector<T>> source, List<Vector<T>>? snapshot = null)
    {
        if (snapshot is not null && (source.Count != snapshot.Count || source.Where((t, i) => t.Length != snapshot[i].Length).Any()))
            throw new ArgumentException("Checkpoint vector layout differs from live state.", nameof(snapshot));
        if (snapshot is null) return source.Select(t => Capture(t)).ToList();
        for (int i = 0; i < source.Count; i++) Capture(source[i], snapshot[i]);
        return snapshot;
    }
}
