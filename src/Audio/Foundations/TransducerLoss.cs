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
    public static Tensor<T> Compute<T>(IEngine engine, Tensor<T> logProbs, IReadOnlyList<int> labels, int blank, double fastEmitLambda = 0.0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (logProbs.Rank != 3) throw new ArgumentException("The lattice must be [frames, labels + 1, vocabulary].", nameof(logProbs));
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], vocabulary = logProbs.Shape[2], length = labels.Count;
        if (width != length + 1) throw new ArgumentException($"The lattice has {width} label positions for {length} labels.", nameof(logProbs));
        if (frames <= 0) throw new ArgumentException("The lattice has no frames.", nameof(logProbs));
        foreach (int label in labels)
            if (label < 0 || label >= vocabulary || label == blank)
                throw new ArgumentException($"Label {label} is outside the vocabulary or is the blank.", nameof(labels));
        var labelsArr = labels as int[] ?? labels.ToArray();
        if (AiDotNet.Tensors.Engines.Compilation.GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(engine);
            var capturedInputs = new[] { logProbs };
            return scope.RecordVariadic(
                AiDotNet.Tensors.Engines.Compilation.LazyNodeType.Custom,
                "TransducerLoss",
                capturedInputs,
                new[] { 1 },
                (_, output) =>
                {
                    var (lossVal, _) = EvaluateCore(capturedInputs[0], labelsArr, blank, fastEmitLambda);
                    output.AsWritableSpan()[0] = lossVal;
                },
                (gradOutput, inputs, _, _, eng, grads) =>
                {
                    var (_, unitGrad) = EvaluateCore(inputs[0], labelsArr, blank, fastEmitLambda);
                    var scaled = eng.TensorMultiplyScalar(unitGrad, gradOutput.Data.Span[0]);
                    DifferentiableOps.AccumulateGrad(grads, inputs[0], scaled, eng);
                });
        }

        var (eagerLossVal, eagerUnitGrad) = EvaluateCore(logProbs, labelsArr, blank, fastEmitLambda);
        var loss = new Tensor<T>(new[] { 1 });
        loss[0] = eagerLossVal;

        DifferentiableOps.RecordUnary("TransducerLoss", loss, logProbs, (gradOutput, inputs, _, _, eng, grads) =>
        {
            var grad = eng.TensorMultiplyScalar(eagerUnitGrad, gradOutput.Data.Span[0]);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, eng);
        });
        return loss;
    }

    private static (T Loss, Tensor<T> UnitGrad) EvaluateCore<T>(Tensor<T> logProbs, int[] labels, int blank, double fastEmitLambda)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], vocabulary = logProbs.Shape[2], length = labels.Length;
        var values = logProbs.ToArray();
        double P(int t, int u, int k) => ops.ToDouble(values[(t * width + u) * vocabulary + k]);
        static double Add(double a, double b)
        {
            if (double.IsNegativeInfinity(a)) return b;
            if (double.IsNegativeInfinity(b)) return a;
            double m = Math.Max(a, b);
            return m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m));
        }

        var alpha = new double[frames, width];
        for (int t = 0; t < frames; t++)
            for (int u = 0; u < width; u++)
            {
                if (t == 0 && u == 0) continue;
                double fromBlank = t > 0 ? alpha[t - 1, u] + P(t - 1, u, blank) : double.NegativeInfinity;
                double fromLabel = u > 0 ? alpha[t, u - 1] + P(t, u - 1, labels[u - 1]) : double.NegativeInfinity;
                alpha[t, u] = Add(fromBlank, fromLabel);
            }

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
        var unitGradArr = new double[frames * width * vocabulary];
        for (int t = 0; t < frames; t++)
            for (int u = 0; u < width; u++)
            {
                int rowBase = (t * width + u) * vocabulary;
                double blankNext = t < frames - 1 ? beta[t + 1, u] : (u == length ? 0.0 : double.NegativeInfinity);
                if (!double.IsNegativeInfinity(blankNext))
                    unitGradArr[rowBase + blank] = -Math.Exp(alpha[t, u] + P(t, u, blank) + blankNext - logLikelihood);
                if (u < length)
                    unitGradArr[rowBase + labels[u]] = -(1.0 + fastEmitLambda)
                        * Math.Exp(alpha[t, u] + P(t, u, labels[u]) + beta[t, u + 1] - logLikelihood);
            }

        Tensor<T> unitGrad;
        using (AiDotNet.Tensors.Engines.Compilation.GraphMode.SuspendRecording())
        {
            unitGrad = new Tensor<double>(unitGradArr, new[] { frames, width, vocabulary }).Cast<T>();
        }
        return (ops.FromDouble(-logLikelihood), unitGrad);
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
        var ops = MathHelper.GetNumericOperations<T>();
        if (logProbs.Rank != 3) throw new ArgumentException("The lattice must be [frames, labels + 1, tokens + durations].", nameof(logProbs));
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], channels = logProbs.Shape[2], length = labels.Count;
        if (frames <= 0) throw new ArgumentException("The lattice has no frames.", nameof(logProbs));
        if (vocabulary <= 0) throw new ArgumentOutOfRangeException(nameof(vocabulary), vocabulary, "The vocabulary must be positive.");
        if (durations.Count == 0) throw new ArgumentException("At least one duration is required.", nameof(durations));
        bool canReachEnd = false;
        foreach (int duration in durations)
        {
            if (duration < 0)
                throw new ArgumentException($"Duration {duration} is negative.", nameof(durations));
            if (duration > 0 && duration <= frames)
                canReachEnd = true;
        }
        if (!canReachEnd)
            throw new ArgumentException($"No positive duration in [{string.Join(", ", durations)}] fits within {frames} frame(s).", nameof(durations));
        if (blank < 0 || blank >= vocabulary)
            throw new ArgumentOutOfRangeException(nameof(blank), blank, $"The blank must be a token in [0, {vocabulary}).");
        foreach (int label in labels)
            if (label < 0 || label >= vocabulary || label == blank)
                throw new ArgumentException($"Label {label} is outside the vocabulary or is the blank.", nameof(labels));
        if (channels != vocabulary + durations.Count) throw new ArgumentException("The last axis must hold the tokens then the durations.", nameof(logProbs));
        if (width != length + 1) throw new ArgumentException($"The lattice has {width} label positions for {length} labels.", nameof(logProbs));
        var durationsArr = durations as int[] ?? durations.ToArray();
        var labelsArr = labels as int[] ?? labels.ToArray();
        if (AiDotNet.Tensors.Engines.Compilation.GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(engine);
            var capturedInputs = new[] { logProbs };
            return scope.RecordVariadic(
                AiDotNet.Tensors.Engines.Compilation.LazyNodeType.Custom,
                "TokenAndDurationTransducerLoss",
                capturedInputs,
                new[] { 1 },
                (_, output) =>
                {
                    var (lossVal, _) = EvaluateTokenAndDurationCore(capturedInputs[0], vocabulary, durationsArr, labelsArr, blank);
                    output.AsWritableSpan()[0] = lossVal;
                },
                (gradOutput, inputs, _, _, eng, grads) =>
                {
                    var (_, unitGrad) = EvaluateTokenAndDurationCore(inputs[0], vocabulary, durationsArr, labelsArr, blank);
                    var scaled = eng.TensorMultiplyScalar(unitGrad, gradOutput.Data.Span[0]);
                    DifferentiableOps.AccumulateGrad(grads, inputs[0], scaled, eng);
                });
        }

        var (tdLossVal, tdUnitGrad) = EvaluateTokenAndDurationCore(logProbs, vocabulary, durationsArr, labelsArr, blank);
        var loss = new Tensor<T>(new[] { 1 });
        loss[0] = tdLossVal;

        DifferentiableOps.RecordUnary("TokenAndDurationTransducerLoss", loss, logProbs, (gradOutput, inputs, _, _, eng, grads) =>
        {
            var tensor = eng.TensorMultiplyScalar(tdUnitGrad, gradOutput.Data.Span[0]);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], tensor, eng);
        });
        return loss;
    }

    private static (T Loss, Tensor<T> UnitGrad) EvaluateTokenAndDurationCore<T>(
        Tensor<T> logProbs, int vocabulary, int[] durations, int[] labels, int blank)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], channels = logProbs.Shape[2], length = labels.Length;
        var values = logProbs.ToArray();
        double Token(int t, int u, int k) => ops.ToDouble(values[(t * width + u) * channels + k]);
        double Duration(int t, int u, int i) => ops.ToDouble(values[(t * width + u) * channels + vocabulary + i]);
        static double Add(double a, double b)
        {
            if (double.IsNegativeInfinity(a)) return b;
            if (double.IsNegativeInfinity(b)) return a;
            double m = Math.Max(a, b);
            return m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m));
        }

        var alpha = new double[frames, width];
        for (int t = 0; t < frames; t++)
            for (int u = 0; u < width; u++)
            {
                if (t == 0 && u == 0) continue;
                double total = double.NegativeInfinity;
                for (int i = 0; i < durations.Length; i++)
                {
                    int d = durations[i], from = t - d;
                    if (from < 0) continue;
                    if (d > 0) total = Add(total, alpha[from, u] + Token(from, u, blank) + Duration(from, u, i));
                    if (u > 0)
                        total = Add(total, alpha[from, u - 1] + Token(from, u - 1, labels[u - 1]) + Duration(from, u - 1, i));
                }
                alpha[t, u] = total;
            }

        var beta = new double[frames, width];
        for (int t = frames - 1; t >= 0; t--)
            for (int u = length; u >= 0; u--)
            {
                double total = double.NegativeInfinity;
                for (int i = 0; i < durations.Length; i++)
                {
                    int d = durations[i], to = t + d;
                    if (d > 0 && u == length && to == frames) total = Add(total, Token(t, u, blank) + Duration(t, u, i));
                    if (d > 0 && to < frames) total = Add(total, beta[to, u] + Token(t, u, blank) + Duration(t, u, i));
                    if (u < length && to < frames) total = Add(total, beta[to, u + 1] + Token(t, u, labels[u]) + Duration(t, u, i));
                }
                beta[t, u] = total;
            }

        double logLikelihood = beta[0, 0];
        var grad = new double[values.Length];
        for (int t = 0; t < frames; t++)
            for (int u = 0; u < width; u++)
                for (int i = 0; i < durations.Length; i++)
                {
                    int d = durations[i], to = t + d, at = (t * width + u) * channels;
                    double blankNext = d == 0 ? double.NegativeInfinity
                        : to == frames ? (u == length ? 0.0 : double.NegativeInfinity)
                        : to < frames ? beta[to, u] : double.NegativeInfinity;
                    if (!double.IsNegativeInfinity(blankNext))
                    {
                        double g = Math.Exp(alpha[t, u] + Token(t, u, blank) + Duration(t, u, i) + blankNext - logLikelihood);
                        grad[at + blank] -= g;
                        grad[at + vocabulary + i] -= g;
                    }
                    if (u < length && to < frames)
                    {
                        double g = Math.Exp(alpha[t, u] + Token(t, u, labels[u]) + Duration(t, u, i) + beta[to, u + 1] - logLikelihood);
                        grad[at + labels[u]] -= g;
                        grad[at + vocabulary + i] -= g;
                    }
                }

        Tensor<T> unitGrad;
        using (AiDotNet.Tensors.Engines.Compilation.GraphMode.SuspendRecording())
        {
            unitGrad = new Tensor<double>(grad, new[] { frames, width, channels }).Cast<T>();
        }
        return (ops.FromDouble(-logLikelihood), unitGrad);
    }
}
