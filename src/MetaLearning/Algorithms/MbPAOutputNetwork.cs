using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// MbPA's output network g_theta — the linear head whose parameters local adaptation modifies — and
/// the closed-form local adaptation step itself.
/// </summary>
/// <remarks>
/// <para>
/// Kept separate from the algorithm so that the adaptation performed at training time and the
/// adaptation performed inside <see cref="MbPAAdaptedModel{T, TInput, TOutput}"/> are literally the
/// same code. Two copies of an update rule drift apart, and a drift here would mean the adapted
/// model is not doing what the algorithm says it does.
/// </para>
/// <para>
/// Parameters are laid out flat as <c>[outputDim * featureDim weights | outputDim biases]</c>, with
/// the weight for output <c>o</c> and feature <c>f</c> at index <c>o * featureDim + f</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class MbPAOutputNetwork<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Evaluates <c>g_theta(h)</c>: a linear map, followed by a softmax for a categorical output.
    /// </summary>
    internal static Vector<T> Forward(
        Vector<T> parameters, Vector<T> key, int featureDim, int outputDim,
        MbPAOutputDistribution distribution)
    {
        var logits = new double[outputDim];
        int biasOffset = outputDim * featureDim;

        for (int o = 0; o < outputDim; o++)
        {
            double sum = biasOffset + o < parameters.Length ? Ops.ToDouble(parameters[biasOffset + o]) : 0.0;
            int rowStart = o * featureDim;
            int len = Math.Min(featureDim, key.Length);
            for (int f = 0; f < len; f++)
            {
                int idx = rowStart + f;
                if (idx < parameters.Length) sum += Ops.ToDouble(parameters[idx]) * Ops.ToDouble(key[f]);
            }
            logits[o] = sum;
        }

        if (distribution == MbPAOutputDistribution.Categorical)
        {
            double max = logits[0];
            for (int o = 1; o < outputDim; o++) if (logits[o] > max) max = logits[o];
            double total = 0.0;
            for (int o = 0; o < outputDim; o++) { logits[o] = Math.Exp(logits[o] - max); total += logits[o]; }
            if (total > 0.0) for (int o = 0; o < outputDim; o++) logits[o] /= total;
        }

        var result = new Vector<T>(outputDim);
        for (int o = 0; o < outputDim; o++) result[o] = Ops.FromDouble(logits[o]);
        return result;
    }

    /// <summary>
    /// The exact gradient of <c>-w * log p(v | h, theta)</c> with respect to the head's parameters.
    /// </summary>
    /// <remarks>
    /// Softmax with cross entropy, and a unit-variance Gaussian with squared error, have the SAME
    /// gradient on a linear head: <c>w (prediction - target) (x) h</c> for the weights and
    /// <c>w (prediction - target)</c> for the biases. Both of MbPA's task families are the former.
    /// Because the form is closed, the local step is exact — no finite differences, and no
    /// dependence on an autodiff graph that the adapted parameters are deliberately kept out of.
    /// </remarks>
    internal static Vector<T> Gradient(
        Vector<T> parameters, Vector<T> key, Vector<T> target, double weight,
        int featureDim, int outputDim, MbPAOutputDistribution distribution)
    {
        var prediction = Forward(parameters, key, featureDim, outputDim, distribution);
        var gradient = new Vector<T>(parameters.Length);
        int biasOffset = outputDim * featureDim;

        for (int o = 0; o < outputDim; o++)
        {
            double residual = weight *
                (Ops.ToDouble(prediction[o]) - (o < target.Length ? Ops.ToDouble(target[o]) : 0.0));
            int rowStart = o * featureDim;
            int len = Math.Min(featureDim, key.Length);
            for (int f = 0; f < len; f++)
            {
                int idx = rowStart + f;
                if (idx < gradient.Length) gradient[idx] = Ops.FromDouble(residual * Ops.ToDouble(key[f]));
            }
            if (biasOffset + o < gradient.Length) gradient[biasOffset + o] = Ops.FromDouble(residual);
        }
        return gradient;
    }

    /// <summary>
    /// The gradient of <c>-w * log p(v | h, theta)</c> with respect to the head's INPUT <c>h</c> —
    /// the embedding — holding the head's parameters fixed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>This is the piece that lets f_gamma be trained through the head instead of around it.</b>
    /// <see cref="Gradient"/> differentiates the same log-likelihood with respect to <c>theta</c>, which
    /// is what the local adaptation steps on. Meta-training needs the other factor: with the same
    /// residual <c>r_o = w (prediction_o - target_o)</c> shared by both of MbPA's distributions, the
    /// weight gradient is <c>r (x) h</c> and the input gradient is <c>W-transpose r</c>. Composing the
    /// latter with the embedding network's own backward pass gives <c>dL/dgamma</c> for the real,
    /// head-composed loss.
    /// </para>
    /// <para>
    /// The returned vector is <paramref name="featureDim"/> long — the head's input width — regardless
    /// of how long <paramref name="key"/> is, for the same reason <see cref="Forward"/> reads only the
    /// first <c>featureDim</c> entries of the key: the head is defined over that width, and an
    /// embedding of some other length has already been resized to it by the caller.
    /// </para>
    /// </remarks>
    internal static Vector<T> InputGradient(
        Vector<T> parameters, Vector<T> key, Vector<T> target, double weight,
        int featureDim, int outputDim, MbPAOutputDistribution distribution)
    {
        var prediction = Forward(parameters, key, featureDim, outputDim, distribution);
        var gradient = new double[featureDim];

        for (int o = 0; o < outputDim; o++)
        {
            double residual = weight *
                (Ops.ToDouble(prediction[o]) - (o < target.Length ? Ops.ToDouble(target[o]) : 0.0));
            if (residual == 0.0) continue;

            int rowStart = o * featureDim;
            for (int f = 0; f < featureDim; f++)
            {
                int idx = rowStart + f;
                if (idx < parameters.Length) gradient[f] += residual * Ops.ToDouble(parameters[idx]);
            }
        }

        var result = new Vector<T>(featureDim);
        for (int f = 0; f < featureDim; f++) result[f] = Ops.FromDouble(gradient[f]);
        return result;
    }

    /// <summary>
    /// The head's loss itself: <c>-w log p(v | h, theta)</c>, cross entropy for the categorical case
    /// and one-half squared error for the Gaussian one.
    /// </summary>
    /// <remarks>
    /// Reported so that meta-training's loss and its gradient describe the SAME objective. Computing
    /// the loss one way and the gradient another is how a training loop comes to look convergent while
    /// optimizing something else.
    /// </remarks>
    internal static double Loss(
        Vector<T> parameters, Vector<T> key, Vector<T> target, double weight,
        int featureDim, int outputDim, MbPAOutputDistribution distribution)
    {
        var prediction = Forward(parameters, key, featureDim, outputDim, distribution);
        double total = 0.0;

        for (int o = 0; o < outputDim; o++)
        {
            double v = o < target.Length ? Ops.ToDouble(target[o]) : 0.0;
            double p = Ops.ToDouble(prediction[o]);
            if (distribution == MbPAOutputDistribution.Categorical)
            {
                // Clamped because softmax underflows to exactly zero for a confidently wrong class,
                // and log(0) would poison the reported mean loss with -infinity for every later task.
                total -= v * Math.Log(Math.Max(p, 1e-12));
            }
            else
            {
                double d = p - v;
                total += 0.5 * d * d;
            }
        }

        return weight * total;
    }


    /// <summary>
    /// MbPA's local adaptation: T gradient steps on the retrieved neighbours' weighted likelihood,
    /// with the MAP prior pulling back toward the trained parameters. Returns theta_x.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Implements the paper's
    /// <c>Delta_M(x, theta) = -alpha_M grad sum_k w_k log p(v_k | h_k, theta_x, x) - beta (theta - theta_x)</c>.
    /// In descent form the second term is <c>-beta (theta_x - theta)</c>, the gradient of the
    /// quadratic prior <c>(beta/2)||theta_x - theta||^2</c>, which is what stops a handful of
    /// neighbours from dragging the head arbitrarily far from its trained solution.
    /// </para>
    /// <para>
    /// The caller owns the returned vector and is expected to DISCARD it after producing the output.
    /// </para>
    /// </remarks>
    /// <param name="trainedParameters">theta — the trained head, left unmodified.</param>
    /// <param name="neighbors">The retrieved <c>(h_k, v_k, w_k)</c> triples.</param>
    /// <param name="steps">T, the number of local gradient steps.</param>
    /// <param name="localLearningRate">alpha_M.</param>
    /// <param name="beta">The prior strength.</param>
    /// <param name="featureDim">Embedding width.</param>
    /// <param name="outputDim">Output width.</param>
    /// <param name="distribution">Which log-likelihood is being maximized.</param>
    internal static Vector<T> LocallyAdapt(
        Vector<T> trainedParameters,
        IReadOnlyList<(Vector<T> Key, Vector<T> Value, double Weight)> neighbors,
        int steps, double localLearningRate, double beta,
        int featureDim, int outputDim, MbPAOutputDistribution distribution)
    {
        // With nothing retrieved there is no evidence to adapt on, so the trained parameters stand.
        // Returning a copy anyway keeps the caller's discard-after-use contract uniform.
        var adapted = trainedParameters.Clone();
        if (neighbors.Count == 0) return adapted;

        for (int step = 0; step < steps; step++)
        {
            var accumulated = new double[adapted.Length];

            // sum_k w_k grad L_k, evaluated at the CURRENT theta_x.
            for (int k = 0; k < neighbors.Count; k++)
            {
                var (key, value, weight) = neighbors[k];
                var grad = Gradient(adapted, key, value, weight, featureDim, outputDim, distribution);
                for (int d = 0; d < accumulated.Length; d++)
                {
                    accumulated[d] += Ops.ToDouble(grad[d]);
                }
            }

            for (int d = 0; d < adapted.Length; d++)
            {
                double drift = Ops.ToDouble(adapted[d]) - Ops.ToDouble(trainedParameters[d]);
                adapted[d] = Ops.FromDouble(
                    Ops.ToDouble(adapted[d]) - localLearningRate * accumulated[d] - beta * drift);
            }
        }

        return adapted;
    }
}
