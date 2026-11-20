namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// The NeRF positional encoding (Mildenhall et al., 2020, eq. 4) shared by the 3D implicit fields.
/// </summary>
internal static class NeRFPositionalEncoding
{
    /// <summary>
    /// Encodes <paramref name="values"/> [N, D] as [N, D + 2 D L]: the values followed, for every octave
    /// k &lt; <paramref name="frequencies"/>, by (sin, cos)(2^k pi v) of each coordinate v in turn.
    /// </summary>
    internal static Tensor<T> Encode<T>(IEngine engine, Tensor<T> values, int frequencies)
    {
        if (frequencies <= 0)
            return values;

        int count = values.Shape[0];
        int dims = values.Shape[1];
        var octaves = new double[frequencies];
        for (int k = 0; k < frequencies; k++)
            octaves[k] = Math.Pow(2, k) * Math.PI;
        var scales = new Tensor<double>(new[] { 1, 1, frequencies }, new Vector<double>(octaves)).Cast<T>();

        // [N, D, L] -> (sin, cos) pairs [N, D, L, 2] -> octave-major [N, L, D, 2].
        var scaled = engine.TensorMultiply(engine.Reshape(values, new[] { count, dims, 1 }), scales);
        var pairs = engine.TensorStack(new[] { engine.TensorSin(scaled), engine.TensorCos(scaled) }, 3);
        var ordered = engine.TensorPermute(pairs, new[] { 0, 2, 1, 3 });
        return engine.TensorConcatenate(
            new[] { values, engine.Reshape(ordered.Contiguous(), new[] { count, 2 * dims * frequencies }) }, 1);
    }
}
