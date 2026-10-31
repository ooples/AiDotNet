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
}
