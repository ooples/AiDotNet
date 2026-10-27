namespace AiDotNet.Models.Parameters;

/// <summary>Validated flat fallback for models with custom packed parameter contracts.</summary>
internal static class ParameterChunkBuffer
{
    internal static Vector<T> ToVector<T>(IEnumerable<Tensor<T>> chunks)
    {
        if (chunks is null) throw new ArgumentNullException(nameof(chunks));

        var buffered = new List<Tensor<T>>();
        long total = 0;
        foreach (var chunk in chunks)
        {
            if (chunk is null)
                throw new ArgumentException("Chunk sequence contains a null tensor.", nameof(chunks));
            buffered.Add(chunk);
            total += chunk.Length;
        }

        var flat = new Vector<T>(checked((int)total));
        int offset = 0;
        var destination = flat.AsWritableSpan();
        foreach (var chunk in buffered)
        {
            chunk.AsSpan().CopyTo(destination.Slice(offset, chunk.Length));
            offset += chunk.Length;
        }

        return flat;
    }
}
