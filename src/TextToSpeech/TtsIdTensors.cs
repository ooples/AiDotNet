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

        // Typed fast paths: one allocation and one widening pass, no Tensor<int> staging and no Cast copy. Short id
        // lists (often a single id) are the common call, where the staging cost dominated. int -> double is exact,
        // and int -> float rounds once to nearest, the same as the int -> double -> float route of FromDouble.
        if (typeof(T) == typeof(double))
        {
            var widened = new double[count];
            for (int i = 0; i < count; i++)
                widened[i] = ids[i];
            return (Tensor<T>)(object)new Tensor<double>(widened, new[] { count });
        }

        if (typeof(T) == typeof(float))
        {
            var narrowed = new float[count];
            for (int i = 0; i < count; i++)
                narrowed[i] = ids[i];
            return (Tensor<T>)(object)new Tensor<float>(narrowed, new[] { count });
        }

        var values = ids as int[] ?? ids.ToArray();
        return new Tensor<int>(values, new[] { count }).Cast<T>();
    }
}
