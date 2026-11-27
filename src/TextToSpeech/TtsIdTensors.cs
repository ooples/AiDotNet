namespace AiDotNet.TextToSpeech;

/// <summary>
/// Token and code ids as tensors of the model's numeric type, for the networks and modules that feed ids to an
/// embedding table.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> text-to-speech models turn text and audio into lists of whole numbers (ids), one per
/// token. The layers that look those ids up expect a tensor, so the list is converted in one step here instead of
/// one number at a time in every model.</para>
/// </remarks>
internal static class TtsIdTensors
{
    /// <summary>
    /// A rank-1 tensor <c>[ids.Count]</c> holding each id as a value of <typeparamref name="T"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="ids">The ids, in order.</param>
    internal static Tensor<T> From<T>(IReadOnlyList<int> ids)
    {
        if (ids is null) throw new ArgumentNullException(nameof(ids));
        if (ids.Count == 0) return new Tensor<T>(new[] { 0 });

        int count = ids.Count;

        // Typed fast paths: the result tensor is the only allocation and is filled in one pass through its span,
        // with no Tensor<int> staging and no Cast copy, so a single id costs what the old element loop did. int ->
        // double is exact, and int -> float rounds once to nearest, the same as the int -> double -> float route of
        // FromDouble.
        if (typeof(T) == typeof(double))
        {
            var widened = new Tensor<double>(new[] { count });
            var span = widened.AsWritableSpan();
            for (int i = 0; i < count; i++)
                span[i] = ids[i];
            return (Tensor<T>)(object)widened;
        }

        if (typeof(T) == typeof(float))
        {
            var narrowed = new Tensor<float>(new[] { count });
            var span = narrowed.AsWritableSpan();
            for (int i = 0; i < count; i++)
                span[i] = ids[i];
            return (Tensor<T>)(object)narrowed;
        }

        var values = ids as int[] ?? ids.ToArray();
        return new Tensor<int>(values, new[] { count }).Cast<T>();
    }
}
