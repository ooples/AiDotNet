using AiDotNet.Tensors.Engines;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// Monotonic text-to-speech alignment over a token-by-frame score matrix: every frame belongs to one token, the
/// first frame to the first token and the last to the last, and the token index stays or advances by one from frame
/// to frame.
/// </summary>
/// <remarks>
/// <para>
/// This is the alignment space of AlignTTS (Zeng et al. 2020, §3.1: the forward variable
/// α<sub>t,s</sub> = (α<sub>t−1,s</sub> + α<sub>t−1,s−1</sub>) · p(y<sub>t</sub> | z<sub>s</sub>)) and of Glow-TTS's
/// monotonic alignment search (Kim et al. 2020). <see cref="MaximumPath"/> is the Viterbi path (the same recursion
/// with a max), from which durations are read; <see cref="LogLikelihood"/> sums over all paths (the Baum-Welch-style
/// alignment loss), computed in log space with engine operations so gradients reach the scores.
/// </para>
/// </remarks>
public static class MonotonicAlignment
{
    /// <summary>
    /// The most likely monotonic alignment of <paramref name="scores"/> (<c>[tokens, frames]</c> log-likelihoods),
    /// returned as the number of frames assigned to each token.
    /// </summary>
    /// <exception cref="ArgumentException">There are fewer frames than tokens, or a score is not finite.</exception>
    public static int[] MaximumPath(double[,] scores)
    {
        if (scores is null) throw new ArgumentNullException(nameof(scores));
        int tokens = scores.GetLength(0), frames = scores.GetLength(1);
        if (tokens == 0 || frames < tokens)
            throw new ArgumentException($"A monotonic alignment of {tokens} tokens needs at least that many frames; got {frames}.", nameof(scores));

        var previous = new double[tokens];
        var current = new double[tokens];
        var advance = new bool[checked(tokens * frames)];
        for (int token = 0; token < tokens; token++) previous[token] = double.NegativeInfinity;
        for (int frame = 0; frame < frames; frame++)
        {
            for (int token = 0; token < tokens; token++) current[token] = double.NegativeInfinity;
            // Tokens reachable at this frame that can still reach the last token by the last frame.
            int first = Math.Max(0, tokens - frames + frame);
            int last = Math.Min(tokens - 1, frame);
            for (int token = first; token <= last; token++)
            {
                double score = scores[token, frame];
                if (double.IsNaN(score) || double.IsInfinity(score))
                    throw new ArgumentException("Alignment scores must be finite.", nameof(scores));
                double stay = previous[token];
                double step = token > 0 ? previous[token - 1] : double.NegativeInfinity;
                bool moves = step > stay;
                current[token] = score + (frame == 0 && token == 0 ? 0 : moves ? step : stay);
                advance[frame * tokens + token] = moves;
            }
            (previous, current) = (current, previous);
        }

        var durations = new int[tokens];
        int activeToken = tokens - 1;
        for (int frame = frames - 1; frame >= 0; frame--)
        {
            durations[activeToken]++;
            if (advance[frame * tokens + activeToken]) activeToken--;
        }
        if (activeToken != 0 || durations.Any(d => d <= 0))
            throw new InvalidOperationException("Monotonic alignment failed to cover every token.");
        return durations;
    }

    /// <summary>
    /// log Σ over all monotonic alignments of Π<sub>t</sub> p(y<sub>t</sub> | z<sub>align(t)</sub>), from
    /// <paramref name="logLikelihood"/> (<c>[tokens, frames]</c>), as a scalar tensor on the gradient tape.
    /// </summary>
    /// <remarks>AlignTTS's forward algorithm (§3.2.1) in log space: α<sub>1</sub> = [log p(y<sub>1</sub>|z<sub>1</sub>),
    /// −∞, …], α<sub>t,s</sub> = log p(y<sub>t</sub>|z<sub>s</sub>) + logsumexp(α<sub>t−1,s</sub>, α<sub>t−1,s−1</sub>),
    /// result α<sub>frames, tokens</sub>. −∞ is represented by a large negative constant so gradients stay finite.</remarks>
    public static Tensor<T> LogLikelihood<T>(Tensor<T> logLikelihood)
    {
        if (logLikelihood is null) throw new ArgumentNullException(nameof(logLikelihood));
        if (logLikelihood.Rank != 2) throw new ArgumentException("Expected [tokens, frames] log-likelihoods.", nameof(logLikelihood));
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int tokens = logLikelihood.Shape[0], frames = logLikelihood.Shape[1];
        if (frames < tokens)
            throw new ArgumentException($"A monotonic alignment of {tokens} tokens needs at least that many frames; got {frames}.", nameof(logLikelihood));

        const double Unreachable = -1e9;
        Tensor<T> Column(int frame) => engine.Reshape(
            engine.TensorSlice(logLikelihood, new[] { 0, frame }, new[] { tokens, 1 }), new[] { tokens });

        // α_1 = [log p(y_1|z_1), unreachable, ...]
        var mask = new Tensor<T>(new[] { tokens });
        for (int s = 1; s < tokens; s++) mask[s] = ops.FromDouble(Unreachable);
        var keepFirst = new Tensor<T>(new[] { tokens });
        keepFirst[0] = ops.One;
        var alpha = engine.TensorAdd(engine.TensorMultiply(Column(0), keepFirst), mask);

        var unreachable = new Tensor<T>(new[] { 1 });
        unreachable[0] = ops.FromDouble(Unreachable);
        for (int frame = 1; frame < frames; frame++)
        {
            // α_{t-1,s-1}: shift down by one token, the first token having no predecessor.
            var shifted = tokens == 1
                ? unreachable
                : engine.TensorConcatenate(new[] { unreachable, engine.TensorSlice(alpha, new[] { 0 }, new[] { tokens - 1 }) }, axis: 0);
            // logsumexp(a, b) = max + log(exp(a - max) + exp(b - max)), elementwise.
            var max = engine.TensorMax(alpha, shifted);
            var sum = engine.TensorAdd(
                engine.TensorExp(engine.TensorSubtract(alpha, max)),
                engine.TensorExp(engine.TensorSubtract(shifted, max)));
            alpha = engine.TensorAdd(Column(frame), engine.TensorAdd(max, engine.TensorLog(sum)));
        }
        return engine.Reshape(engine.TensorSlice(alpha, new[] { tokens - 1 }, new[] { 1 }), new[] { 1 });
    }
}
