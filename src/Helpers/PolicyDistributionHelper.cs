namespace AiDotNet.Helpers;

/// <summary>
/// Provides tape-differentiable policy distribution computations for reinforcement learning.
/// All methods are engine-routed so gradients flow through the gradient tape; the calling
/// agent passes its own <see cref="IEngine"/> so the helper never reaches for a static
/// <c>AiDotNetEngine.Current</c> (which would break per-call engine overrides, DI in
/// serving / dashboard scenarios, and test-time engine isolation).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> In reinforcement learning, the agent's "policy" is a probability
/// distribution over actions. To train the policy, we need to compute:
/// - <b>Log-probability:</b> How likely was the action the agent took? (used in policy gradient)
/// - <b>Entropy:</b> How "spread out" is the distribution? (used to encourage exploration)
///
/// These computations must use engine operations (not scalar math) so that the gradient
/// tape can automatically compute how to adjust the policy network's weights.
///
/// <b>Discrete actions:</b> The network outputs logits (raw scores) for each action.
/// We apply softmax to get probabilities, then take log of the selected action's probability.
///
/// <b>Continuous actions:</b> The network outputs mean and log-standard-deviation for a
/// Gaussian (normal) distribution. The log-probability follows the Gaussian formula.
/// </remarks>
public static class PolicyDistributionHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes log-probabilities for discrete actions from logits using engine ops.
    /// </summary>
    /// <param name="engine">The agent's <see cref="IEngine"/> (typically <c>this.Engine</c>
    /// from a <see cref="NeuralNetworks.NeuralNetworkBase{T}"/>-derived agent) so tape
    /// routing matches the surrounding forward pass.</param>
    /// <param name="logits">Network output logits [batchSize, numActions].</param>
    /// <param name="actionIndices">Selected action index per sample [batchSize].</param>
    /// <returns>Log-probabilities tensor [batchSize] — tape-differentiable.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This converts raw network scores into log-probabilities:
    /// 1. Apply softmax to get probabilities that sum to 1
    /// 2. Take the log of the probability of the action that was actually taken
    /// 3. The result tells us "how surprising was this action under the current policy?"
    ///
    /// A log-probability of 0 means probability = 1 (the agent was certain).
    /// A very negative log-probability means the action was unlikely.
    /// </remarks>
    public static Tensor<T> ComputeDiscreteLogProb(IEngine engine, Tensor<T> logits, int[] actionIndices)
    {
        // log_softmax = logits - log(sum(exp(logits))) — numerically stable via log-sum-exp
        var softmaxed = engine.Softmax(logits);
        var safeSoftmax = engine.TensorAddScalar(softmaxed, NumOps.FromDouble(1e-8));
        var logProbs = engine.TensorLog(safeSoftmax);

        // Gather log-prob at each action index via an engine-routed mask
        // multiply + reduce. The previous implementation copied values via
        // the indexer (`result[i] = logProbs[i * numActions + idx]`) which
        // broke the gradient tape — the freshly-constructed `result`
        // tensor has no producer node, so `tape.ComputeGradients` returns
        // zero w.r.t. the policy network's parameters and the optimizer
        // applies a no-op update. The same gradient-break-on-indexer-copy
        // pattern is documented in feedback_tensor_reshape_gradient.md.
        //
        // The mask is a one-hot tensor of the same shape as `logProbs`
        // ([batch, numActions]) with `mask[i, actionIndices[i]] = 1`.
        // Multiply element-wise and reduce along the action axis to gather
        // the selected log-prob per sample — all engine ops, all tape-tracked.
        int batchSize = actionIndices.Length;
        int numActions = logits.Shape.Length > 1 ? logits.Shape[1] : logits.Length;
        var maskData = new T[batchSize * numActions];
        for (int i = 0; i < batchSize; i++)
            maskData[i * numActions + actionIndices[i]] = NumOps.One;
        var mask = new Tensor<T>(maskData, [batchSize, numActions]);
        var masked = engine.TensorMultiply(logProbs, mask);
        return engine.ReduceSum(masked, [1], keepDims: false);
    }

    /// <summary>
    /// Computes entropy of a discrete distribution from logits using engine ops.
    /// </summary>
    /// <param name="engine">The agent's <see cref="IEngine"/>.</param>
    /// <param name="logits">Network output logits [batchSize, numActions].</param>
    /// <returns>Entropy tensor [batchSize] — tape-differentiable.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Entropy measures how "uncertain" the policy is:
    /// - High entropy = the agent considers many actions equally likely (exploring)
    /// - Low entropy = the agent strongly prefers one action (exploiting)
    ///
    /// We add entropy as a bonus to the loss to encourage the agent to keep exploring
    /// rather than getting stuck always choosing the same action.
    ///
    /// Formula: H = -sum(p * log(p)) for each action probability p.
    /// </remarks>
    public static Tensor<T> ComputeDiscreteEntropy(IEngine engine, Tensor<T> logits)
    {
        var softmaxed = engine.Softmax(logits);
        var safeSoftmax = engine.TensorAddScalar(softmaxed, NumOps.FromDouble(1e-8));
        var logProbs = engine.TensorLog(safeSoftmax);
        var pLogP = engine.TensorMultiply(softmaxed, logProbs);
        var negPLogP = engine.TensorNegate(pLogP);

        // Sum over action dimension (axis 1) to get per-sample entropy
        if (negPLogP.Shape.Length > 1)
        {
            return engine.ReduceSum(negPLogP, [1], keepDims: false);
        }

        // Single sample — sum all
        var allAxes = Enumerable.Range(0, negPLogP.Shape.Length).ToArray();
        return engine.ReduceSum(negPLogP, allAxes, keepDims: false);
    }

    /// <summary>
    /// Computes log-probabilities for continuous (Gaussian) actions using engine ops.
    /// </summary>
    /// <param name="engine">The agent's <see cref="IEngine"/>.</param>
    /// <param name="means">Gaussian means from network [batchSize, actionSize].</param>
    /// <param name="logStds">Log standard deviations [batchSize, actionSize].</param>
    /// <param name="actions">Actions taken [batchSize, actionSize].</param>
    /// <returns>Total log-probability per sample [batchSize] — tape-differentiable.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> For continuous actions (like steering angles or forces),
    /// the policy outputs a Gaussian (bell curve) distribution for each action dimension.
    ///
    /// The network outputs:
    /// - <b>Mean (μ):</b> The center of the bell curve — the "preferred" action
    /// - <b>Log-std (log σ):</b> How wide the bell curve is — how much to explore
    ///
    /// The log-probability tells us how likely the actual action was under this distribution:
    /// log p(a) = -0.5 * ((a - μ)/σ)² - log(σ) - 0.5 * log(2π)
    ///
    /// This is summed across all action dimensions for the total log-probability.
    /// </remarks>
    public static Tensor<T> ComputeGaussianLogProb(IEngine engine, Tensor<T> means, Tensor<T> logStds, Tensor<T> actions)
    {
        // log_prob = -0.5 * ((action - mean) / std)^2 - log_std - 0.5 * log(2π)
        var stds = engine.TensorExp(logStds);
        var safeStds = engine.TensorAddScalar(stds, NumOps.FromDouble(1e-8));
        var diff = engine.TensorSubtract(actions, means);
        var normalized = engine.TensorDivide(diff, safeStds);
        var normalizedSq = engine.TensorMultiply(normalized, normalized);
        var halfNormSq = engine.TensorMultiplyScalar(normalizedSq, NumOps.FromDouble(-0.5));

        var logStdTerm = engine.TensorNegate(logStds);
        var constTerm = NumOps.FromDouble(-0.5 * Math.Log(2.0 * Math.PI));

        var perElement = engine.TensorAdd(halfNormSq, engine.TensorAddScalar(logStdTerm, constTerm));

        // Sum over action dimensions to get total log-prob per sample
        if (perElement.Shape.Length > 1)
        {
            return engine.ReduceSum(perElement, [1], keepDims: false);
        }

        var allAxes = Enumerable.Range(0, perElement.Shape.Length).ToArray();
        return engine.ReduceSum(perElement, allAxes, keepDims: false);
    }

    /// <summary>
    /// Computes entropy of a Gaussian distribution using engine ops.
    /// </summary>
    /// <param name="engine">The agent's <see cref="IEngine"/>.</param>
    /// <param name="logStds">Log standard deviations [batchSize, actionSize].</param>
    /// <returns>Entropy per sample [batchSize] — tape-differentiable.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> For a Gaussian distribution, entropy depends only on
    /// the standard deviation (width of the bell curve), not the mean.
    ///
    /// Formula: H = 0.5 * log(2πe) + log_std (per dimension), summed across dimensions.
    ///
    /// Higher entropy = wider bell curves = more exploration.
    /// </remarks>
    public static Tensor<T> ComputeGaussianEntropy(IEngine engine, Tensor<T> logStds)
    {
        // H = 0.5 * log(2πe) + log_std = 0.5 * (1 + log(2π)) + log_std
        var halfLogTwoPiE = NumOps.FromDouble(0.5 * (1.0 + Math.Log(2.0 * Math.PI)));
        var perElement = engine.TensorAddScalar(logStds, halfLogTwoPiE);

        if (perElement.Shape.Length > 1)
        {
            return engine.ReduceSum(perElement, [1], keepDims: false);
        }

        var allAxes = Enumerable.Range(0, perElement.Shape.Length).ToArray();
        return engine.ReduceSum(perElement, allAxes, keepDims: false);
    }
}
