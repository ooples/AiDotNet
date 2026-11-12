namespace AiDotNet.Audio.VoiceActivity;

/// <summary>
/// Whole-tensor helpers shared by the voice activity detectors.
/// </summary>
internal static class VadEngineOps
{
    /// <summary>
    /// 1 for every frame whose probability is at least <paramref name="threshold"/>, else 0 (NaN is 0).
    /// </summary>
    internal static int[] SpeechMask<T>(T[] probabilities, double threshold)
    {
        if (probabilities.Length == 0)
        {
            return Array.Empty<int>();
        }

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var probs = new Tensor<T>((T[])probabilities.Clone(), [probabilities.Length]);
        var limit = new Tensor<T>([probabilities.Length]);
        limit.Fill(numOps.FromDouble(threshold));
        return engine.TensorGreaterEqual(probs, limit).Cast<int>().ToArray();
    }

    /// <summary>
    /// A new <c>[frameSamples]</c> tensor holding <paramref name="audio"/> (read flat) from
    /// <paramref name="start"/>, zero padded past the end of the audio.
    /// </summary>
    internal static Tensor<T> Frame<T>(Tensor<T> audio, int start, int frameSamples)
    {
        var frame = new Tensor<T>([frameSamples]);
        int available = Math.Max(0, Math.Min(frameSamples, audio.Length - start));
        if (available > 0)
        {
            audio.AsSpan().Slice(start, available).CopyTo(frame.AsWritableSpan());
        }

        return frame;
    }
}
