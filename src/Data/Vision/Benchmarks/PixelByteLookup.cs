namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Byte-to-<typeparamref name="T"/> pixel table shared by the binary image loaders.
/// </summary>
/// <remarks>
/// Decoding a byte image through a 256-entry table writes each pixel straight into the output, so a loader
/// needs no whole-dataset int or T intermediates. The table is one engine cast of the double values, which
/// gives exactly the scalar <c>NumOps.FromDouble(b / 255.0)</c> (or <c>FromDouble(b)</c>) for every byte.
/// </remarks>
internal static class PixelByteLookup
{
    /// <summary>Creates the table; entry <c>b</c> is <c>b / 255</c> when normalising, otherwise <c>b</c>.</summary>
    internal static T[] Create<T>(bool normalize)
    {
        var values = new double[256];
        for (int b = 0; b < values.Length; b++)
        {
            values[b] = normalize ? b / 255.0 : b;
        }

        return new Tensor<double>(values, new[] { values.Length }).Cast<T>().AsSpan().ToArray();
    }
}
