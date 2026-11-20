using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// The RNN transducer loss (Graves 2012, "Sequence Transduction with Recurrent Neural Networks", section 2): the negative
/// log-probability of a label sequence summed over every alignment of the joint network's lattice, by the forward
/// (alpha) and backward (beta) recursions, with FastEmit's emission regularization (Yu et al. 2021) as an option.
/// </summary>
internal static class TransducerLoss
{
    /// <summary>
    /// The loss of one utterance. <paramref name="logProbs"/> is the joint network's log-softmax output
    /// <c>[frames, labels + 1, vocabulary]</c>: entry <c>[t, u, k]</c> is log Pr(k | t, u), the probability of emitting
    /// <c>k</c> at frame <c>t</c> after the first <c>u</c> labels. The gradient flows back to <paramref name="logProbs"/>.
    /// </summary>
    /// <param name="engine">The tensor engine.</param>
    /// <param name="logProbs">The lattice's log-probabilities.</param>
    /// <param name="labels">The target labels, none of them the blank.</param>
    /// <param name="blank">The blank symbol.</param>
    /// <param name="fastEmitLambda">FastEmit's λ (Yu et al. 2021, Equation 9): label emissions' gradients are scaled by
    /// <c>1 + λ</c>, which pushes the model to emit earlier; zero is the plain transducer loss.</param>
    /// <returns>The scalar loss <c>[1]</c>.</returns>
    /// <remarks>
    /// <para>The forward recursion runs on the engine, one anti-diagonal of the lattice at a time: every node (t, u)
    /// with t + u = n is reached from diagonal n − 1, by a blank from (t − 1, u) or a label from (t, u − 1), so each
    /// diagonal is one log-add of two shifted vectors. Nothing is read on the host, so a compiled training plan
    /// replays the loss, on the GPU too, and its gradient is the autodiff of those ops: exactly Graves' alpha-beta
    /// gradient. FastEmit scales the label emissions' gradients by 1 + λ through (1 + λ)·x − λ·stop(x), whose value
    /// is x.</para>
    /// <para>Off-lattice lanes hold a large finite negative rather than −∞, so a log-add of two of them is not NaN.
    /// Their weight in a log-add with a lattice node is exp(−10⁶ − ·) = 0. In a log-add of two such lanes the weights
    /// still sum to 1, so their gradient does not grow, and it ends in constants and masked (×0) gathers.</para>
    /// </remarks>
    public static Tensor<T> Compute<T>(IEngine engine, Tensor<T> logProbs, IReadOnlyList<int> labels, int blank, double fastEmitLambda = 0.0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Validate(logProbs, labels, blank);
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], vocabulary = logProbs.Shape[2];
        int diagonals = frames + width - 1;
        var flat = engine.Reshape(logProbs, new[] { frames * width * vocabulary });

        // Per destination diagonal n = 1 .. diagonals - 1 and label position u: the flat index of the transition's
        // log-probability into node (n - u, u), and whether that transition exists.
        int steps = diagonals - 1;
        var blankIndex = new int[Math.Max(1, steps * width)];
        var labelIndex = new int[Math.Max(1, steps * width)];
        var blankValid = new Tensor<T>(new[] { Math.Max(1, steps * width) });
        var labelValid = new Tensor<T>(new[] { Math.Max(1, steps * width) });
        var blankMask = blankValid.AsWritableSpan();
        var labelMask = labelValid.AsWritableSpan();
        for (int n = 1; n <= steps; n++)
            for (int u = 0; u < width; u++)
            {
                int t = n - u, at = (n - 1) * width + u;
                if (t < 0 || t >= frames) continue;
                if (t >= 1) { blankIndex[at] = ((t - 1) * width + u) * vocabulary + blank; blankMask[at] = ops.One; }
                if (u >= 1) { labelIndex[at] = (t * width + u - 1) * vocabulary + labels[u - 1]; labelMask[at] = ops.One; }
            }
        var negative = ops.FromDouble(OffLattice);
        Tensor<T> Masked(Tensor<T> gathered, Tensor<T> valid)
        {
            // gathered * valid + (1 - valid) * OffLattice: an off-lattice lane is the constant, with no gradient path.
            var off = engine.TensorMultiplyScalar(engine.TensorAddScalar(engine.TensorMultiplyScalar(valid, ops.FromDouble(-1.0)), ops.One), negative);
            return engine.TensorAdd(engine.TensorMultiply(gathered, valid), off);
        }

        var alpha = new Tensor<T>(new[] { width });
        var alphaInit = alpha.AsWritableSpan();
        for (int u = 1; u < width; u++) alphaInit[u] = negative;
        if (steps > 0)
        {
            var blanks = Masked(engine.TensorIndexSelect(flat, new Tensor<int>(new[] { blankIndex.Length }, new Vector<int>(blankIndex)), 0), blankValid);
            var emitted = engine.TensorIndexSelect(flat, new Tensor<int>(new[] { labelIndex.Length }, new Vector<int>(labelIndex)), 0);
            if (Math.Abs(fastEmitLambda) > 0.0)
                emitted = engine.TensorSubtract(engine.TensorMultiplyScalar(emitted, ops.FromDouble(1.0 + fastEmitLambda)),
                    engine.TensorMultiplyScalar(engine.StopGradient(emitted), ops.FromDouble(fastEmitLambda)));
            var emissions = Masked(emitted, labelValid);
            blanks = engine.Reshape(blanks, new[] { steps, width });
            emissions = engine.Reshape(emissions, new[] { steps, width });
            var first = new Tensor<T>(new[] { 1 });
            first.AsWritableSpan()[0] = negative;
            for (int n = 1; n <= steps; n++)
            {
                var blankRow = engine.Reshape(engine.TensorSlice(blanks, new[] { n - 1, 0 }, new[] { 1, width }), new[] { width });
                var labelRow = engine.Reshape(engine.TensorSlice(emissions, new[] { n - 1, 0 }, new[] { 1, width }), new[] { width });
                var viaBlank = engine.TensorAdd(alpha, blankRow);
                var shifted = width == 1 ? first
                    : engine.TensorConcatenate(new[] { first, engine.TensorSlice(alpha, new[] { 0 }, new[] { width - 1 }) }, 0);
                var viaLabel = engine.TensorAdd(shifted, labelRow);
                alpha = engine.TensorLogAddExp(viaBlank, viaLabel);
            }
        }
        // log P = alpha(T - 1, U) + log Pr(blank | T - 1, U).
        var closing = engine.TensorIndexSelect(flat, new Tensor<int>(new[] { 1 }, new Vector<int>(new[] { ((frames - 1) * width + width - 1) * vocabulary + blank })), 0);
        var logLikelihood = engine.TensorAdd(engine.TensorSlice(alpha, new[] { width - 1 }, new[] { 1 }), closing);
        return engine.TensorMultiplyScalar(logLikelihood, ops.FromDouble(-1.0));
    }

    // The value of a lane off the lattice: far below any path's log-probability, and finite.
    private const double OffLattice = -1e6;

    private static void Validate<T>(Tensor<T> logProbs, IReadOnlyList<int> labels, int blank)
    {
        if (logProbs.Rank != 3) throw new ArgumentException("The lattice must be [frames, labels + 1, vocabulary].", nameof(logProbs));
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], vocabulary = logProbs.Shape[2], length = labels.Count;
        if (width != length + 1) throw new ArgumentException($"The lattice has {width} label positions for {length} labels.", nameof(logProbs));
        if (frames <= 0) throw new ArgumentException("The lattice has no frames.", nameof(logProbs));
        ValidateTokens(labels, blank, vocabulary);
    }

    // An out-of-range token is not an out-of-range read: the flattened gather lands in a neighbouring cell or channel
    // and the loss is silently wrong.
    private static void ValidateTokens(IReadOnlyList<int> labels, int blank, int vocabulary)
    {
        if (blank < 0 || blank >= vocabulary)
            throw new ArgumentOutOfRangeException(nameof(blank), blank, $"The blank must be a token in [0, {vocabulary}).");
        foreach (int label in labels)
            if (label < 0 || label >= vocabulary || label == blank)
                throw new ArgumentException($"Label {label} is outside the vocabulary or is the blank.", nameof(labels));
    }

    /// <summary>
    /// The same loss by the host alpha-beta recursions with an explicit gradient (Graves 2012, Eq. 16): the reference the
    /// engine form is tested against.
    /// </summary>
    [AiDotNet.Attributes.ScalarKernel("The host reference recursion the engine form is tested against; no model trains through it.")]
    internal static Tensor<T> ComputeOnHost<T>(IEngine engine, Tensor<T> logProbs, IReadOnlyList<int> labels, int blank, double fastEmitLambda = 0.0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Validate(logProbs, labels, blank);
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], vocabulary = logProbs.Shape[2], length = labels.Count;
        var values = logProbs.ToArray();
        double P(int t, int u, int k) => ops.ToDouble(values[(t * width + u) * vocabulary + k]);
        double Add(double a, double b)
        {
            if (double.IsNegativeInfinity(a)) return b;
            if (double.IsNegativeInfinity(b)) return a;
            double m = Math.Max(a, b);
            return m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m));
        }

        // alpha(t, u): log-probability of reaching node (t, u), having emitted the first u labels before frame t ends.
        var alpha = new double[frames, width];
        for (int t = 0; t < frames; t++)
            for (int u = 0; u < width; u++)
            {
                if (t == 0 && u == 0) continue;
                double fromBlank = t > 0 ? alpha[t - 1, u] + P(t - 1, u, blank) : double.NegativeInfinity;
                double fromLabel = u > 0 ? alpha[t, u - 1] + P(t, u - 1, labels[u - 1]) : double.NegativeInfinity;
                alpha[t, u] = Add(fromBlank, fromLabel);
            }
        // beta(t, u): log-probability of completing the sequence from node (t, u), closed by the final blank.
        var beta = new double[frames, width];
        for (int t = frames - 1; t >= 0; t--)
            for (int u = length; u >= 0; u--)
            {
                if (t == frames - 1 && u == length) { beta[t, u] = P(t, u, blank); continue; }
                double viaBlank = t < frames - 1 ? beta[t + 1, u] + P(t, u, blank) : double.NegativeInfinity;
                double viaLabel = u < length ? beta[t, u + 1] + P(t, u, labels[u]) : double.NegativeInfinity;
                beta[t, u] = Add(viaBlank, viaLabel);
            }
        double logLikelihood = beta[0, 0];
        var loss = new Tensor<T>(new[] { 1 });
        loss[0] = ops.FromDouble(-logLikelihood);

        DifferentiableOps.RecordUnary("TransducerLoss", loss, logProbs, (gradOutput, inputs, _, _, eng, grads) =>
        {
            double scale = ops.ToDouble(gradOutput.Data.Span[0]);
            var grad = new Tensor<T>(new[] { frames, width, vocabulary });
            for (int t = 0; t < frames; t++)
                for (int u = 0; u < width; u++)
                {
                    // d(-log P)/d log Pr(k | t, u) = -exp(alpha(t, u) + log Pr(k | t, u) + beta(next) - log P).
                    double blankNext = t < frames - 1 ? beta[t + 1, u] : (u == length ? 0.0 : double.NegativeInfinity);
                    if (!double.IsNegativeInfinity(blankNext))
                        grad[t, u, blank] = ops.FromDouble(-scale * Math.Exp(alpha[t, u] + P(t, u, blank) + blankNext - logLikelihood));
                    if (u < length)
                        grad[t, u, labels[u]] = ops.FromDouble(-scale * (1.0 + fastEmitLambda)
                            * Math.Exp(alpha[t, u] + P(t, u, labels[u]) + beta[t, u + 1] - logLikelihood));
                }
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, eng);
        });
        return loss;
    }

    /// <summary>
    /// The token-and-duration transducer loss (Xu et al. 2023, "Efficient Sequence Transduction by Jointly Predicting
    /// Tokens and Durations", section 3): at node (t, u) the joint network gives a token distribution and, independently, a
    /// distribution over how many frames the emission advances. A token with duration d moves to (t + d, u + 1), a blank
    /// with duration d (at least 1) to (t + d, u); a sequence completes when a blank reaches frame T with every label
    /// emitted. The loss is the negative log of every such path's probability (Equations 3 to 5).
    /// </summary>
    /// <param name="engine">The tensor engine.</param>
    /// <param name="logProbs">Token log-probabilities <c>[T, U + 1, V]</c> then duration log-probabilities
    /// <c>[T, U + 1, durations]</c>, concatenated on the last axis.</param>
    /// <param name="vocabulary">V, the token entries at the start of the last axis.</param>
    /// <param name="durations">The duration each duration entry stands for, such as 0 to 4.</param>
    /// <param name="labels">The target labels.</param>
    /// <param name="blank">The blank token.</param>
    /// <returns>The scalar loss <c>[1]</c>.</returns>
    public static Tensor<T> ComputeTokenAndDuration<T>(IEngine engine, Tensor<T> logProbs, int vocabulary, IReadOnlyList<int> durations,
        IReadOnlyList<int> labels, int blank)
    {
        // The forward recursion on the engine, one anti-diagonal n = t + u at a time (lane u holds node (n - u, u)). A
        // blank of duration d >= 1 reaches (t, u) from (t - d, u), on diagonal n - d; a label of any duration from
        // (t - d, u - 1), on diagonal n - d - 1, one lane down. Each transition's log-probability (token plus duration) is
        // gathered once per kind and duration with a validity mask, as in Compute, and the gradient is the autodiff.
        var ops = MathHelper.GetNumericOperations<T>();
        if (logProbs.Rank != 3) throw new ArgumentException("The lattice must be [frames, labels + 1, tokens + durations].", nameof(logProbs));
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], channels = logProbs.Shape[2], length = labels.Count;
        if (channels != vocabulary + durations.Count) throw new ArgumentException("The last axis must hold the tokens then the durations.", nameof(logProbs));
        if (width != length + 1) throw new ArgumentException($"The lattice has {width} label positions for {length} labels.", nameof(logProbs));
        if (frames <= 0) throw new ArgumentException("The lattice has no frames.", nameof(logProbs));
        if (vocabulary <= 0) throw new ArgumentOutOfRangeException(nameof(vocabulary), vocabulary, "The vocabulary must have at least one token.");
        if (durations.Count == 0) throw new ArgumentException("At least one duration is required.", nameof(durations));
        // A negative duration would step forward in time (t - d > t) and index outside the intended lattice.
        foreach (int duration in durations)
            if (duration < 0) throw new ArgumentException($"Duration {duration} is negative.", nameof(durations));
        ValidateTokens(labels, blank, vocabulary);
        int count = durations.Count, diagonals = frames + width - 1, steps = diagonals - 1;
        var flat = engine.Reshape(logProbs, new[] { frames * width * channels });
        var negative = ops.FromDouble(OffLattice);
        Tensor<T> Masked(Tensor<T> gathered, Tensor<T> valid)
        {
            var off = engine.TensorMultiplyScalar(engine.TensorAddScalar(engine.TensorMultiplyScalar(valid, ops.FromDouble(-1.0)), ops.One), negative);
            return engine.TensorAdd(engine.TensorMultiply(gathered, valid), off);
        }
        int At(int t, int u) => (t * width + u) * channels;
        // Edge log-probabilities [steps, width] of one kind and duration: the token's plus the duration's.
        Tensor<T> Edges(int d, bool isLabel, int slot)
        {
            int size = Math.Max(1, steps * width);
            var tokenIndex = new int[size];
            var durationIndex = new int[size];
            var valid = new Tensor<T>(new[] { size });
            var mask = valid.AsWritableSpan();
            for (int n = 1; n <= steps; n++)
                for (int u = 0; u < width; u++)
                {
                    int t = n - u, at = (n - 1) * width + u, from = t - d;
                    if (t < 0 || t >= frames || from < 0) continue;
                    if (isLabel)
                    {
                        if (u < 1) continue;
                        tokenIndex[at] = At(from, u - 1) + labels[u - 1];
                        durationIndex[at] = At(from, u - 1) + slot;
                    }
                    else
                    {
                        tokenIndex[at] = At(from, u) + blank;
                        durationIndex[at] = At(from, u) + slot;
                    }
                    mask[at] = ops.One;
                }
            var token = engine.TensorIndexSelect(flat, new Tensor<int>(new[] { size }, new Vector<int>(tokenIndex)), 0);
            var duration = engine.TensorIndexSelect(flat, new Tensor<int>(new[] { size }, new Vector<int>(durationIndex)), 0);
            return engine.Reshape(Masked(engine.TensorAdd(token, duration), valid), new[] { Math.Max(1, steps), width });
        }
        var blankEdges = new Tensor<T>?[count];
        var labelEdges = new Tensor<T>[count];
        for (int i = 0; i < count; i++)
        {
            int d = durations[i];
            if (d > 0) blankEdges[i] = Edges(d, isLabel: false, vocabulary + i);
            labelEdges[i] = Edges(d, isLabel: true, vocabulary + i);
        }

        var offDiagonal = new Tensor<T>(new[] { width });
        offDiagonal.AsWritableSpan().Fill(negative);
        var first = new Tensor<T>(new[] { 1 });
        first.AsWritableSpan()[0] = negative;
        Tensor<T> Shifted(Tensor<T> diagonal) => width == 1 ? first
            : engine.TensorConcatenate(new[] { first, engine.TensorSlice(diagonal, new[] { 0 }, new[] { width - 1 }) }, 0);
        Tensor<T> Row(Tensor<T> edges, int n) => engine.Reshape(engine.TensorSlice(edges, new[] { n - 1, 0 }, new[] { 1, width }), new[] { width });

        var alpha = new Tensor<T>[diagonals];
        var start = new Tensor<T>(new[] { width });
        var startSpan = start.AsWritableSpan();
        for (int u = 1; u < width; u++) startSpan[u] = negative;
        alpha[0] = start;
        // log(exp(total) + exp(term)), with an absent total as the empty sum.
        Tensor<T> Join(Tensor<T>? total, Tensor<T> term) => total is null ? term : engine.TensorLogAddExp(total, term);
        // Diagonal n's lanes reached over the transitions of one duration: a blank from diagonal n - d, a label from
        // diagonal n - d - 1 one lane down.
        Tensor<T> Reached(int n, int i)
        {
            int d = durations[i];
            var viaLabel = engine.TensorAdd(n - d - 1 >= 0 ? Shifted(alpha[n - d - 1]) : offDiagonal, Row(labelEdges[i], n));
            if (blankEdges[i] is not { } blanks) return viaLabel;
            var viaBlank = engine.TensorAdd(n - d >= 0 ? alpha[n - d] : offDiagonal, Row(blanks, n));
            return engine.TensorLogAddExp(viaBlank, viaLabel);
        }
        for (int n = 1; n <= steps; n++)
        {
            Tensor<T>? total = null;
            for (int i = 0; i < count; i++) total = Join(total, Reached(n, i));
            alpha[n] = total ?? offDiagonal;
        }

        // log P: a closing blank of duration d >= 1 from (T - d, U) reaches frame T.
        Tensor<T>? likelihood = null;
        for (int i = 0; i < count; i++)
        {
            int d = durations[i], t = frames - d;
            if (d <= 0 || t < 0) continue;
            var indices = new Tensor<int>(new[] { 2 }, new Vector<int>(new[] { At(t, length) + blank, At(t, length) + vocabulary + i }));
            var edge = engine.ReduceSum(engine.TensorIndexSelect(flat, indices, 0), new[] { 0 }, keepDims: true);
            var term = engine.TensorAdd(engine.TensorSlice(alpha[t + length], new[] { length }, new[] { 1 }), edge);
            likelihood = Join(likelihood, term);
        }
        if (likelihood is null) throw new ArgumentException("No duration reaches the last frame.", nameof(durations));
        return engine.TensorMultiplyScalar(likelihood, ops.FromDouble(-1.0));
    }

    /// <summary>The token-and-duration loss by the host recursions with an explicit gradient: the reference the engine
    /// form is tested against.</summary>
    [AiDotNet.Attributes.ScalarKernel("The host reference recursion the engine form is tested against; no model trains through it.")]
    internal static Tensor<T> ComputeTokenAndDurationOnHost<T>(IEngine engine, Tensor<T> logProbs, int vocabulary, IReadOnlyList<int> durations,
        IReadOnlyList<int> labels, int blank)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], channels = logProbs.Shape[2], length = labels.Count;
        if (channels != vocabulary + durations.Count) throw new ArgumentException("The last axis must hold the tokens then the durations.", nameof(logProbs));
        if (width != length + 1) throw new ArgumentException($"The lattice has {width} label positions for {length} labels.", nameof(logProbs));
        var values = logProbs.ToArray();
        double Token(int t, int u, int k) => ops.ToDouble(values[(t * width + u) * channels + k]);
        double Duration(int t, int u, int i) => ops.ToDouble(values[(t * width + u) * channels + vocabulary + i]);
        double Add(double a, double b)
        {
            if (double.IsNegativeInfinity(a)) return b;
            if (double.IsNegativeInfinity(b)) return a;
            double m = Math.Max(a, b);
            return m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m));
        }

        // alpha(t, u) for t < T: log-probability of reaching node (t, u).
        var alpha = new double[frames, width];
        for (int t = 0; t < frames; t++)
            for (int u = 0; u < width; u++)
            {
                if (t == 0 && u == 0) continue;
                double total = double.NegativeInfinity;
                for (int i = 0; i < durations.Count; i++)
                {
                    int d = durations[i], from = t - d;
                    if (from < 0) continue;
                    if (d > 0) total = Add(total, alpha[from, u] + Token(from, u, blank) + Duration(from, u, i));
                    if (u > 0)
                        total = Add(total, alpha[from, u - 1] + Token(from, u - 1, labels[u - 1]) + Duration(from, u - 1, i));
                }
                alpha[t, u] = total;
            }
        // beta(t, u): log-probability of completing from node (t, u).
        var beta = new double[frames, width];
        for (int t = frames - 1; t >= 0; t--)
            for (int u = length; u >= 0; u--)
            {
                double total = double.NegativeInfinity;
                for (int i = 0; i < durations.Count; i++)
                {
                    int d = durations[i], to = t + d;
                    if (d > 0 && u == length && to == frames) total = Add(total, Token(t, u, blank) + Duration(t, u, i));
                    if (d > 0 && to < frames) total = Add(total, beta[to, u] + Token(t, u, blank) + Duration(t, u, i));
                    if (u < length && to < frames) total = Add(total, beta[to, u + 1] + Token(t, u, labels[u]) + Duration(t, u, i));
                }
                beta[t, u] = total;
            }
        double logLikelihood = beta[0, 0];
        var loss = new Tensor<T>(new[] { 1 });
        loss[0] = ops.FromDouble(-logLikelihood);

        DifferentiableOps.RecordUnary("TokenAndDurationTransducerLoss", loss, logProbs, (gradOutput, inputs, _, _, eng, grads) =>
        {
            double scale = ops.ToDouble(gradOutput.Data.Span[0]);
            var grad = new double[values.Length];
            for (int t = 0; t < frames; t++)
                for (int u = 0; u < width; u++)
                    for (int i = 0; i < durations.Count; i++)
                    {
                        int d = durations[i], to = t + d, at = (t * width + u) * channels;
                        double blankNext = d == 0 ? double.NegativeInfinity
                            : to == frames ? (u == length ? 0.0 : double.NegativeInfinity)
                            : to < frames ? beta[to, u] : double.NegativeInfinity;
                        if (!double.IsNegativeInfinity(blankNext))
                        {
                            double g = scale * Math.Exp(alpha[t, u] + Token(t, u, blank) + Duration(t, u, i) + blankNext - logLikelihood);
                            grad[at + blank] -= g;
                            grad[at + vocabulary + i] -= g;
                        }
                        if (u < length && to < frames)
                        {
                            double g = scale * Math.Exp(alpha[t, u] + Token(t, u, labels[u]) + Duration(t, u, i) + beta[to, u + 1] - logLikelihood);
                            grad[at + labels[u]] -= g;
                            grad[at + vocabulary + i] -= g;
                        }
                    }
            var tensor = new Tensor<T>(new[] { frames, width, channels });
            var span = tensor.Data.Span;
            for (int n = 0; n < grad.Length; n++) span[n] = ops.FromDouble(grad[n]);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], tensor, eng);
        });
        return loss;
    }
}
