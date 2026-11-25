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

        var values = ids as int[] ?? ids.ToArray();
        return new Tensor<int>(values, new[] { values.Length }).Cast<T>();
    }
}
