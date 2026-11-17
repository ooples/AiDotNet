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
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (actionIndices is null) throw new ArgumentNullException(nameof(actionIndices));

        int batchSize = actionIndices.Length;
        int numActions = logits.Shape.Length > 1 ? logits.Shape[1] : logits.Length;
        int logitsBatch = logits.Shape.Length > 1 ? logits.Shape[0] : 1;
        if (batchSize != logitsBatch)
        {
            throw new ArgumentException(
                $"actionIndices length ({batchSize}) must match logits batch size ({logitsBatch}).",
                nameof(actionIndices));
        }

        var maskData = new T[batchSize * numActions];
        for (int i = 0; i < batchSize; i++)
        {
            int idx = actionIndices[i];
            if ((uint)idx >= (uint)numActions)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(actionIndices),
                    $"Action index {idx} at sample {i} is outside [0, {numActions - 1}].");
            }
            maskData[i * numActions + idx] = NumOps.One;
        }

        var logits2D = logits.Shape.Length == 1 ? engine.Reshape(logits, [1, numActions]) : logits;
        var logProbs = engine.TensorLogSoftmax(logits2D, axis: 1);
        var mask = new Tensor<T>(maskData, [batchSize, numActions]);
        return SelectDiscreteLogProbFromLogProbs(engine, logProbs, mask);
    }

    /// <summary>
    /// Selects per-sample log-probabilities using a validated one-hot action mask without multiplying
    /// unselected <c>-Infinity</c> lanes by zero (which produces IEEE-754 <c>NaN</c> in forward/backward).
    /// </summary>
    internal static Tensor<T> SelectDiscreteLogProbFromLogProbs(
        IEngine engine,
        Tensor<T> logProbs,
        Tensor<T> oneHotActions)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (logProbs is null) throw new ArgumentNullException(nameof(logProbs));
        if (oneHotActions is null) throw new ArgumentNullException(nameof(oneHotActions));

        var zeros = engine.TensorZerosLike(logProbs);
        var selected = engine.TensorWhere(oneHotActions, logProbs, zeros);
        return engine.ReduceSum(selected, [1], keepDims: false);
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
    /// Formula: H = -sum(p * log(p)) for each action probability p, using the convention 0 * log(0) = 0.
    /// </remarks>
    public static Tensor<T> ComputeDiscreteEntropy(IEngine engine, Tensor<T> logits)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (logits is null) throw new ArgumentNullException(nameof(logits));

        int actionAxis = logits.Shape.Length > 1 ? 1 : 0;
        var logProbs = engine.TensorLogSoftmax(logits, axis: actionAxis);
        return ComputeDiscreteEntropyFromLogProbs(engine, logProbs);
    }

    /// <summary>
    /// Computes discrete entropy directly from log-softmax outputs while gating zero-probability lanes via
    /// <see cref="IEngine.TensorWhere{T}(Tensor{T}, Tensor{T}, Tensor{T})"/> so masked or underflowed
    /// <c>-Infinity</c> log-probabilities never evaluate <c>0 * (-Infinity) = NaN</c> in forward or backward.
    /// </summary>
    internal static Tensor<T> ComputeDiscreteEntropyFromLogProbs(IEngine engine, Tensor<T> logProbs)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (logProbs is null) throw new ArgumentNullException(nameof(logProbs));

        var probs = engine.TensorExp(logProbs);
        var zeros = engine.TensorZerosLike(logProbs);
        var safeLogProbs = engine.TensorWhere(probs, logProbs, zeros);
        var pLogP = engine.TensorMultiply(probs, safeLogProbs);
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

    /// <summary>
    /// Bounded action <c>a = scale * tanh(u)</c> for a squashed Gaussian policy — tape-differentiable.
    /// </summary>
    /// <param name="engine">The agent's <see cref="IEngine"/>.</param>
    /// <param name="preSquashActions">Pre-squash samples <c>u</c> [batchSize, actionSize].</param>
    /// <param name="scale">Half-width of the action interval, so the result lies in <c>(-scale, scale)</c>.</param>
    /// <remarks>
    /// <b>For Beginners:</b> A Gaussian can return any number at all, but a real actuator (or a trading
    /// position) has limits. Passing the sample through tanh squeezes it into a fixed range without
    /// clipping, so the policy can still be trained by gradients everywhere.
    /// </remarks>
    internal static Tensor<T> SquashAction(IEngine engine, Tensor<T> preSquashActions, double scale)
    {
        ValidateActionScale(scale);

        // Scaling unconditionally rather than testing scale == 1: multiplying by one is an identity, so
        // the guard would only trade an exact floating-point comparison for a negligible multiply.
        return engine.TensorMultiplyScalar(engine.TensorTanh(preSquashActions), NumOps.FromDouble(scale));
    }

    /// <summary>
    /// Log-probability of a tanh-squashed (and optionally scaled) Gaussian action — the bounded-action
    /// density from Soft Actor-Critic, Haarnoja et al. 2018, Appendix C.
    /// </summary>
    /// <param name="engine">The agent's <see cref="IEngine"/>.</param>
    /// <param name="means">Pre-squash means <c>mu</c> [batchSize, actionSize].</param>
    /// <param name="logStds">Pre-squash log standard deviations [batchSize, actionSize].</param>
    /// <param name="preSquashActions">The sampled <c>u</c> the action came from [batchSize, actionSize].</param>
    /// <param name="scale">The same half-width passed to <see cref="SquashAction"/>.</param>
    /// <returns>Total log-probability per sample [batchSize] — tape-differentiable.</returns>
    /// <remarks>
    /// <para>
    /// Squashing changes the density, so the Gaussian log-probability alone is wrong for the bounded
    /// action. The change-of-variables correction (Eq. 20/21) is
    /// <c>log pi(a|s) = log mu(u|s) - sum_i log(1 - tanh^2(u_i))</c>, and a constant scale divides the
    /// density once per dimension, contributing a further <c>-D * log(scale)</c>.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Squeezing a bell curve into a fixed range piles up probability near the
    /// edges. This term accounts for that, so "how likely was this action" stays honest — without it the
    /// entropy bonus is measured against the wrong distribution and the temperature drifts.
    /// </para>
    /// </remarks>
    internal static Tensor<T> ComputeSquashedGaussianLogProb(
        IEngine engine, Tensor<T> means, Tensor<T> logStds, Tensor<T> preSquashActions, double scale)
    {
        ValidateActionScale(scale);

        var gaussianLogProb = ComputeGaussianLogProb(engine, means, logStds, preSquashActions);

        // -sum_i log(1 - tanh^2(u_i)). The epsilon keeps a saturated tanh from producing log(0).
        var squashed = engine.TensorTanh(preSquashActions);
        var oneMinusSquared = engine.TensorAddScalar(
            engine.TensorNegate(engine.TensorSquare(squashed)), NumOps.One);
        var logJacobian = engine.TensorLog(
            engine.TensorAddScalar(oneMinusSquared, NumOps.FromDouble(1e-6)));

        var summedJacobian = logJacobian.Shape.Length > 1
            ? engine.ReduceSum(logJacobian, [1], keepDims: false)
            : engine.ReduceSum(
                logJacobian, Enumerable.Range(0, logJacobian.Shape.Length).ToArray(), keepDims: false);

        var corrected = engine.TensorSubtract(gaussianLogProb, summedJacobian);

        int actionSize = preSquashActions.Shape.Length > 1
            ? preSquashActions.Shape[preSquashActions.Shape.Length - 1]
            : preSquashActions.Length;

        // log(1) is 0, so an unscaled policy is unaffected and no float comparison is needed.
        return engine.TensorAddScalar(corrected, NumOps.FromDouble(-actionSize * Math.Log(scale)));
    }
    /// <summary>
    /// Rejects a scale that cannot describe a valid action range.
    /// </summary>
    /// <remarks>
    /// Zero collapses every action onto the origin and sends the log-probability to infinity via
    /// -D*log(scale); negative, NaN and infinite scales produce invalid actions or a non-finite
    /// density. Failing here beats propagating a silent NaN into the temperature update.
    /// </remarks>
    private static void ValidateActionScale(double scale)
    {
        if (double.IsNaN(scale) || double.IsInfinity(scale) || scale <= 0.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(scale), scale,
                "Action scale must be a finite positive value; it divides the squashed density once "
                + "per action dimension, so zero or a non-finite scale yields an undefined log-probability.");
        }
    }
}
