namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Reads a class-index target of a fused readout loss (one class id per row, stored in the network's numeric type) as
/// the <c>int</c> ids the fused kernels take. Shared by every <see cref="Interfaces.IDetachableFusedReadoutLossLayer{T}"/>
/// so the readouts agree on what a valid id is.
/// </summary>
internal static class FusedReadoutClassIndices
{
    /// <summary>
    /// The ids of <paramref name="target"/> in row order, or <c>null</c> when any value is not a whole number in
    /// <c>[0, vocabularySize)</c>: fusion is then declined and the caller takes the decomposed loss path, which
    /// may ignore such an id (for instance an ignore index of <c>-1</c>) rather than report it.
    /// </summary>
    internal static Tensor<int>? TryRead<T>(Tensor<T> target, int vocabularySize)
    {
        var values = target.ToArray();
        var ids = new int[values.Length];
        bool ok = values switch
        {
            float[] f => TryReadSingle(f, ids, vocabularySize),
            double[] d => TryReadDouble(d, ids, vocabularySize),
            int[] i => TryReadInt32(i, ids, vocabularySize),
            _ => TryReadGeneric(values, ids, vocabularySize),
        };
        return ok ? new Tensor<int>(ids, new[] { ids.Length }) : null;
    }

    private static bool TryReadSingle(float[] values, int[] ids, int vocabularySize)
    {
        for (int r = 0; r < values.Length; r++)
            if (!TryId(values[r], vocabularySize, out ids[r])) return false;
        return true;
    }

    private static bool TryReadDouble(double[] values, int[] ids, int vocabularySize)
    {
        for (int r = 0; r < values.Length; r++)
            if (!TryId(values[r], vocabularySize, out ids[r])) return false;
        return true;
    }

    private static bool TryReadInt32(int[] values, int[] ids, int vocabularySize)
    {
        for (int r = 0; r < values.Length; r++)
        {
            if ((uint)values[r] >= (uint)vocabularySize) return false;
            ids[r] = values[r];
        }
        return true;
    }

    [AiDotNet.Attributes.ScalarKernel("Element types other than float, double and int have no typed span; those take the loops above.")]
    private static bool TryReadGeneric<T>(T[] values, int[] ids, int vocabularySize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int r = 0; r < values.Length; r++)
            if (!TryId(numOps.ToDouble(values[r]), vocabularySize, out ids[r])) return false;
        return true;
    }

    private static bool TryId(double value, int vocabularySize, out int id)
    {
        id = 0;
        if (!(value >= 0 && value < vocabularySize) || Math.Floor(value) < value) return false;
        id = (int)value;
        return true;
    }
}
