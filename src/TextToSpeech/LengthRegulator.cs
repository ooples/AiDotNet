using AiDotNet.Tensors.Engines;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// FastSpeech's length regulator (Ren et al. 2019, §3.2): repeats each token's hidden state for its duration in
/// frames, so a token-level sequence becomes frame-level.
/// </summary>
/// <remarks>The repetition is an index-select on the gradient tape, so gradients flow back to every token.</remarks>
public static class LengthRegulator
{
    /// <summary>
    /// Repeats row <c>i</c> of <paramref name="tokens"/> (<c>[tokens, hidden]</c>, or <c>[1, tokens, hidden]</c>)
    /// <c>durations[i]</c> times, giving <c>[frames, hidden]</c> (or <c>[1, frames, hidden]</c>).
    /// </summary>
    public static Tensor<T> Expand<T>(Tensor<T> tokens, int[] durations)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (durations is null) throw new ArgumentNullException(nameof(durations));
        bool batched = tokens.Rank == 3;
        if (batched && tokens.Shape[0] != 1)
            throw new ArgumentException("Expand takes one utterance; each utterance expands to its own length.", nameof(tokens));
        var engine = AiDotNetEngine.Current;
        var rows = batched ? engine.Reshape(tokens, new[] { tokens.Shape[1], tokens.Shape[2] }) : tokens;
        if (durations.Length != rows.Shape[0])
            throw new ArgumentException($"Got {durations.Length} durations for {rows.Shape[0]} tokens.", nameof(durations));

        int total = 0;
        foreach (int d in durations)
        {
            if (d < 0) throw new ArgumentOutOfRangeException(nameof(durations), "Durations cannot be negative.");
            total += d;
        }
        if (total == 0) throw new ArgumentException("The durations expand to zero frames.", nameof(durations));

        var indices = new Tensor<int>(new[] { total });
        int k = 0;
        for (int i = 0; i < durations.Length; i++)
            for (int r = 0; r < durations[i]; r++) indices[k++] = i;
        var frames = engine.TensorIndexSelect(rows, indices, 0);
        return batched ? engine.Reshape(frames, new[] { 1, total, rows.Shape[1] }) : frames;
    }
}
