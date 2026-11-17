using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;

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
    public static Tensor<T> Compute<T>(IEngine engine, Tensor<T> logProbs, IReadOnlyList<int> labels, int blank, double fastEmitLambda = 0.0)
    {
        if (logProbs.Rank != 3) throw new ArgumentException("The lattice must be [frames, labels + 1, vocabulary].", nameof(logProbs));
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], vocabulary = logProbs.Shape[2], length = labels.Count;
        if (width != length + 1) throw new ArgumentException($"The lattice has {width} label positions for {length} labels.", nameof(logProbs));
        if (frames <= 0) throw new ArgumentException("The lattice has no frames.", nameof(logProbs));
        var labelCopy = labels.ToArray();
        for (int i = 0; i < labelCopy.Length; i++)
        {
            int label = labelCopy[i];
            if (label < 0 || label >= vocabulary || label == blank)
                throw new ArgumentException($"Label {label} is outside the vocabulary or is the blank.", nameof(labels));
        }

        if (GraphMode.Current is { } scope)
        {
            var capturedInputs = new[] { logProbs };
            return scope.RecordVariadic(
                LazyNodeType.Custom,
                "TransducerLoss",
                capturedInputs,
                new[] { 1 },
                (replayEngine, result) =>
                {
                    using var _ = new NoGradScope<T>();
                    var lossTensor = EvaluateStandardForward(capturedInputs[0], labelCopy, blank).Loss;
                    AiDotNet.Tensors.Engines.DirectGpuTensorEngine.CopyResultInto(replayEngine, lossTensor, result);
                },
                static (gradOutput, inputs, _, savedState, eng, grads) =>
                {
                    var savedLabels = (int[])savedState[0];
                    int savedBlank = (int)savedState[1];
                    double savedLambda = (double)savedState[2];
                    var forward = EvaluateStandardForward(inputs[0], savedLabels, savedBlank);
                    var grad = EvaluateStandardBackward(inputs[0], gradOutput, savedLabels, savedBlank, savedLambda,
                        forward.Alpha, forward.Beta, forward.LogLikelihood);
                    DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, eng);
                },
                new object[] { labelCopy, blank, fastEmitLambda },
                new[] { 0 });
        }

        var eager = EvaluateStandardForward(logProbs, labelCopy, blank);
        DifferentiableOps.RecordUnary("TransducerLoss", eager.Loss, logProbs, (gradOutput, inputs, _, _, eng, grads) =>
        {
            var grad = EvaluateStandardBackward(inputs[0], gradOutput, labelCopy, blank, fastEmitLambda,
                eager.Alpha, eager.Beta, eager.LogLikelihood);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, eng);
        });
        return eager.Loss;
    }

    private static (Tensor<T> Loss, double[,] Alpha, double[,] Beta, double LogLikelihood) EvaluateStandardForward<T>(
        Tensor<T> logProbs, int[] labels, int blank)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], vocabulary = logProbs.Shape[2], length = labels.Length;
        var values = logProbs.ToArray();
        double P(int t, int u, int k) => ops.ToDouble(values[(t * width + u) * vocabulary + k]);

        var alpha = new double[frames, width];
        for (int t = 0; t < frames; t++)
            for (int u = 0; u < width; u++)
            {
                if (t == 0 && u == 0) continue;
                double fromBlank = t > 0 ? alpha[t - 1, u] + P(t - 1, u, blank) : double.NegativeInfinity;
                double fromLabel = u > 0 ? alpha[t, u - 1] + P(t, u - 1, labels[u - 1]) : double.NegativeInfinity;
                alpha[t, u] = LogAddExp(fromBlank, fromLabel);
            }

        var beta = new double[frames, width];
        for (int t = frames - 1; t >= 0; t--)
            for (int u = length; u >= 0; u--)
            {
                if (t == frames - 1 && u == length) { beta[t, u] = P(t, u, blank); continue; }
                double viaBlank = t < frames - 1 ? beta[t + 1, u] + P(t, u, blank) : double.NegativeInfinity;
                double viaLabel = u < length ? beta[t, u + 1] + P(t, u, labels[u]) : double.NegativeInfinity;
                beta[t, u] = LogAddExp(viaBlank, viaLabel);
            }

        double logLikelihood = beta[0, 0];
        var loss = new Tensor<T>(new[] { 1 });
        loss[0] = ops.FromDouble(-logLikelihood);
        return (loss, alpha, beta, logLikelihood);
    }

    private static Tensor<T> EvaluateStandardBackward<T>(
        Tensor<T> logProbs, Tensor<T> gradOutput, int[] labels, int blank, double fastEmitLambda,
        double[,] alpha, double[,] beta, double logLikelihood)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], vocabulary = logProbs.Shape[2], length = labels.Length;
        var values = logProbs.ToArray();
        double P(int t, int u, int k) => ops.ToDouble(values[(t * width + u) * vocabulary + k]);
        double scale = ops.ToDouble(gradOutput.Data.Span[0]);
        var grad = new double[frames * width * vocabulary];
        for (int t = 0; t < frames; t++)
            for (int u = 0; u < width; u++)
            {
                int rowBase = (t * width + u) * vocabulary;
                double blankNext = t < frames - 1 ? beta[t + 1, u] : (u == length ? 0.0 : double.NegativeInfinity);
                if (!double.IsNegativeInfinity(blankNext))
                    grad[rowBase + blank] = -scale * Math.Exp(alpha[t, u] + P(t, u, blank) + blankNext - logLikelihood);
                if (u < length)
                    grad[rowBase + labels[u]] = -scale * (1.0 + fastEmitLambda)
                        * Math.Exp(alpha[t, u] + P(t, u, labels[u]) + beta[t, u + 1] - logLikelihood);
            }
        return TensorCopyHelper.CastLogical<double, T>(new Tensor<double>(new[] { frames, width, vocabulary }, new Vector<double>(grad)));
    }

    /// <summary>
    /// The token-and-duration transducer loss (Xu et al. 2023, "Efficient Sequence Transduction by Jointly Predicting
    /// Tokens and Durations", section 3).
    /// </summary>
    public static Tensor<T> ComputeTokenAndDuration<T>(IEngine engine, Tensor<T> logProbs, int vocabulary, IReadOnlyList<int> durations,
        IReadOnlyList<int> labels, int blank)
    {
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], channels = logProbs.Shape[2], length = labels.Count;
        if (channels != vocabulary + durations.Count) throw new ArgumentException("The last axis must hold the tokens then the durations.", nameof(logProbs));
        if (width != length + 1) throw new ArgumentException($"The lattice has {width} label positions for {length} labels.", nameof(logProbs));
        var durationCopy = durations.ToArray();
        var labelCopy = labels.ToArray();

        if (GraphMode.Current is { } scope)
        {
            var capturedInputs = new[] { logProbs };
            return scope.RecordVariadic(
                LazyNodeType.Custom,
                "TokenAndDurationTransducerLoss",
                capturedInputs,
                new[] { 1 },
                (replayEngine, result) =>
                {
                    using var _ = new NoGradScope<T>();
                    var lossTensor = EvaluateTokenDurationForward(capturedInputs[0], vocabulary, durationCopy, labelCopy, blank).Loss;
                    AiDotNet.Tensors.Engines.DirectGpuTensorEngine.CopyResultInto(replayEngine, lossTensor, result);
                },
                static (gradOutput, inputs, _, savedState, eng, grads) =>
                {
                    int savedVocabulary = (int)savedState[0];
                    var savedDurations = (int[])savedState[1];
                    var savedLabels = (int[])savedState[2];
                    int savedBlank = (int)savedState[3];
                    var forward = EvaluateTokenDurationForward(inputs[0], savedVocabulary, savedDurations, savedLabels, savedBlank);
                    var grad = EvaluateTokenDurationBackward(inputs[0], gradOutput, savedVocabulary, savedDurations, savedLabels, savedBlank,
                        forward.Alpha, forward.Beta, forward.LogLikelihood);
                    DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, eng);
                },
                new object[] { vocabulary, durationCopy, labelCopy, blank },
                new[] { 0 });
        }

        var eager = EvaluateTokenDurationForward(logProbs, vocabulary, durationCopy, labelCopy, blank);
        DifferentiableOps.RecordUnary("TokenAndDurationTransducerLoss", eager.Loss, logProbs, (gradOutput, inputs, _, _, eng, grads) =>
        {
            var grad = EvaluateTokenDurationBackward(inputs[0], gradOutput, vocabulary, durationCopy, labelCopy, blank,
                eager.Alpha, eager.Beta, eager.LogLikelihood);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, eng);
        });
        return eager.Loss;
    }

    private static (Tensor<T> Loss, double[,] Alpha, double[,] Beta, double LogLikelihood) EvaluateTokenDurationForward<T>(
        Tensor<T> logProbs, int vocabulary, int[] durations, int[] labels, int blank)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], channels = logProbs.Shape[2], length = labels.Length;
        var values = logProbs.ToArray();
        double Token(int t, int u, int k) => ops.ToDouble(values[(t * width + u) * channels + k]);
        double Duration(int t, int u, int i) => ops.ToDouble(values[(t * width + u) * channels + vocabulary + i]);

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
                    if (d > 0) total = LogAddExp(total, alpha[from, u] + Token(from, u, blank) + Duration(from, u, i));
                    if (u > 0)
                        total = LogAddExp(total, alpha[from, u - 1] + Token(from, u - 1, labels[u - 1]) + Duration(from, u - 1, i));
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
                    if (d > 0 && u == length && to == frames) total = LogAddExp(total, Token(t, u, blank) + Duration(t, u, i));
                    if (d > 0 && to < frames) total = LogAddExp(total, beta[to, u] + Token(t, u, blank) + Duration(t, u, i));
                    if (u < length && to < frames) total = LogAddExp(total, beta[to, u + 1] + Token(t, u, labels[u]) + Duration(t, u, i));
                }
                beta[t, u] = total;
            }

        double logLikelihood = beta[0, 0];
        var loss = new Tensor<T>(new[] { 1 });
        loss[0] = ops.FromDouble(-logLikelihood);
        return (loss, alpha, beta, logLikelihood);
    }

    private static Tensor<T> EvaluateTokenDurationBackward<T>(
        Tensor<T> logProbs, Tensor<T> gradOutput, int vocabulary, int[] durations, int[] labels, int blank,
        double[,] alpha, double[,] beta, double logLikelihood)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int frames = logProbs.Shape[0], width = logProbs.Shape[1], channels = logProbs.Shape[2], length = labels.Length;
        var values = logProbs.ToArray();
        double Token(int t, int u, int k) => ops.ToDouble(values[(t * width + u) * channels + k]);
        double Duration(int t, int u, int i) => ops.ToDouble(values[(t * width + u) * channels + vocabulary + i]);
        double scale = ops.ToDouble(gradOutput.Data.Span[0]);
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
        return TensorCopyHelper.CastLogical<double, T>(new Tensor<double>(new[] { frames, width, channels }, new Vector<double>(grad)));
    }

    private static double LogAddExp(double a, double b)
    {
        if (double.IsNegativeInfinity(a)) return b;
        if (double.IsNegativeInfinity(b)) return a;
        double m = Math.Max(a, b);
        return m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m));
    }
}
