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

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// The head's weights <c>W</c> [outputDim, featureDim] and biases <c>b</c> [outputDim] read from the flat
    /// layout. Entries past the end of a short <paramref name="parameters"/> read as zero.
    /// </summary>
    internal static (Tensor<T> Weights, Tensor<T> Bias) Split(Vector<T> parameters, int featureDim, int outputDim)
    {
        int weightCount = outputDim * featureDim;
        var flat = MbPAConversions<T>.ResizeTo(parameters, weightCount + outputDim);
        var weights = Engine.Reshape(Tensor<T>.FromVector(flat.GetSubVector(0, weightCount)), new[] { outputDim, featureDim });
        return (weights, Tensor<T>.FromVector(flat.GetSubVector(weightCount, outputDim)));
    }

    /// <summary>
    /// Evaluates <c>g_theta(h)</c>: a linear map, followed by a softmax for a categorical output.
    /// </summary>
    internal static Vector<T> Forward(
        Vector<T> parameters, Vector<T> key, int featureDim, int outputDim,
        MbPAOutputDistribution distribution)
    {
        var (weights, bias) = Split(parameters, featureDim, outputDim);
        var logits = Logits(weights, bias, key, featureDim);

        if (distribution == MbPAOutputDistribution.Categorical)
        {
            // Max-shifted softmax; a non-positive total (NaN logits) leaves the exponentials unnormalized.
            var exps = Engine.TensorExp(Engine.TensorAddScalar(logits, Ops.Negate(Engine.TensorMaxValue(logits))));
            T total = Engine.TensorSum(exps);
            logits = Ops.GreaterThan(total, Ops.Zero) ? Engine.TensorDivideScalar(exps, total) : exps;
        }

        return logits.ToVector();
    }

    /// <summary><c>W h + b</c> for the first <paramref name="featureDim"/> entries of the key (zero-padded).</summary>
    private static Tensor<T> Logits(Tensor<T> weights, Tensor<T> bias, Vector<T> key, int featureDim)
    {
        var h = Tensor<T>.FromVector(MbPAConversions<T>.ResizeTo(key, featureDim), new[] { featureDim, 1 });
        return Engine.TensorAdd(Engine.Reshape(Engine.TensorMatMul(weights, h), new[] { bias.Length }), bias);
    }

    /// <summary>The residual <c>w (prediction - target)</c> with the target zero-padded to the output width.</summary>
    private static Tensor<T> Residual(Vector<T> prediction, Vector<T> target, double weight, int outputDim)
        => Engine.TensorMultiplyScalar(
            Tensor<T>.FromVector(Engine.Subtract(prediction, MbPAConversions<T>.ResizeTo(target, outputDim))),
            Ops.FromDouble(weight));

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
        var residual = Residual(prediction, target, weight, outputDim);

        // [r (x) h | r], laid out like the parameters and cut or zero-padded to their length.
        var h = Tensor<T>.FromVector(MbPAConversions<T>.ResizeTo(key, featureDim));
        var weightGradient = Engine.Reshape(Engine.TensorOuter(residual, h), new[] { outputDim * featureDim });
        var gradient = Engine.TensorConcatenate(new[] { weightGradient, residual }, 0).ToVector();
        return MbPAConversions<T>.ResizeTo(gradient, parameters.Length);
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
    /// <param name="parameters">theta, at least <c>(featureDim + 1) * outputDim</c> long.</param>
    /// <param name="embeddings">The head inputs, [batch, featureDim].</param>
    /// <param name="targets">The targets, [batch, outputDim].</param>
    /// <param name="weight">w, the per-example likelihood weight.</param>
    /// <param name="featureDim">Embedding width.</param>
    /// <param name="outputDim">Output width.</param>
    /// <param name="distribution">Which log-likelihood is being maximized.</param>
    /// <returns>The [batch, featureDim] input gradients <c>R W</c> for the residual rows <c>R</c>.</returns>
    internal static Tensor<T> InputGradient(
        Vector<T> parameters, Tensor<T> embeddings, Tensor<T> targets, double weight,
        int featureDim, int outputDim, MbPAOutputDistribution distribution)
    {
        var (weights, _) = Split(parameters, featureDim, outputDim);
        var logits = BatchLogits(parameters, embeddings, featureDim, outputDim);
        var prediction = distribution == MbPAOutputDistribution.Categorical ? Engine.Softmax(logits, 1) : logits;
        var residual = Engine.TensorMultiplyScalar(Engine.TensorSubtract(prediction, targets), Ops.FromDouble(weight));
        return Engine.TensorMatMul(residual, weights);
    }

    /// <summary>Row-wise <c>E W^T + b</c> for embeddings <c>E</c> [batch, featureDim].</summary>
    private static Tensor<T> BatchLogits(Vector<T> parameters, Tensor<T> embeddings, int featureDim, int outputDim)
    {
        var (weights, bias) = Split(parameters, featureDim, outputDim);
        var linear = Engine.TensorMatMulTransposed(embeddings, weights);
        return Engine.TensorAdd(linear, Engine.TensorBroadcastTo(Engine.Reshape(bias, new[] { 1, outputDim }), linear.Shape.ToArray()));
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
    /// <param name="parameters">theta, at least <c>(featureDim + 1) * outputDim</c> long.</param>
    /// <param name="embeddings">The head inputs, [batch, featureDim].</param>
    /// <param name="targets">The targets, [batch, outputDim].</param>
    /// <param name="weight">w, the per-example likelihood weight.</param>
    /// <param name="featureDim">Embedding width.</param>
    /// <param name="outputDim">Output width.</param>
    /// <param name="distribution">Which log-likelihood is being maximized.</param>
    /// <returns>The loss summed over the batch.</returns>
    internal static T Loss(
        Vector<T> parameters, Tensor<T> embeddings, Tensor<T> targets, double weight,
        int featureDim, int outputDim, MbPAOutputDistribution distribution)
    {
        // Evaluate categorical likelihood in logit space so underflow does not clip the objective.
        var logits = BatchLogits(parameters, embeddings, featureDim, outputDim);
        Tensor<T> perEntry;
        if (distribution == MbPAOutputDistribution.Categorical)
        {
            // v_o * (logsumexp(z) - z_o)
            var logSum = Engine.TensorLogSumExp(logits, 1, true);
            perEntry = Engine.TensorMultiply(targets,
                Engine.TensorSubtract(Engine.TensorBroadcastTo(logSum, logits.Shape.ToArray()), logits));
        }
        else
        {
            var diff = Engine.TensorSubtract(logits, targets);
            perEntry = Engine.TensorMultiplyScalar(Engine.TensorMultiply(diff, diff), Ops.FromDouble(0.5));
        }

        return Ops.Multiply(Ops.FromDouble(weight), Engine.TensorSum(perEntry));
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

        T learningRate = Ops.FromDouble(localLearningRate);
        T prior = Ops.FromDouble(beta);
        for (int step = 0; step < steps; step++)
        {
            // sum_k w_k grad L_k, evaluated at the CURRENT theta_x.
            var accumulated = new Vector<T>(adapted.Length);
            for (int k = 0; k < neighbors.Count; k++)
            {
                var (key, value, weight) = neighbors[k];
                accumulated = Engine.Add(accumulated,
                    Gradient(adapted, key, value, weight, featureDim, outputDim, distribution));
            }

            // theta_x - alpha_M * sum - beta * (theta_x - theta)
            var drift = Engine.Subtract(adapted, trainedParameters);
            adapted = Engine.Subtract(adapted,
                Engine.Add(Engine.Multiply(accumulated, learningRate), Engine.Multiply(drift, prior)));
        }

        return adapted;
    }
}
