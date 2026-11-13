using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning;

/// <summary>
/// Whole-vector engine helpers shared by the reinforcement-learning agents and policies.
/// </summary>
/// <remarks>
/// Every helper keeps the semantics of the scalar loops it replaces: arg-max returns the first
/// index holding the maximum (a strict greater-than scan starting at index 0, so a NaN in slot 0
/// stays selected exactly as before), and stacking/concatenation are pure data movement.
/// </remarks>
internal static class RLTensorOps
{
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>First index of the largest element (strict greater-than scan from index 0).</summary>
    internal static int ArgMax<T>(Vector<T> values)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length == 0)
        {
            // The scalar scans read element 0 first; reading it here raises the same
            // out-of-range exception they did for an empty vector.
            _ = values[0];
        }

        return Engine.TensorArgMax(values.AsTensor(), 0)[0];
    }

    /// <summary>
    /// First index whose element is strictly greater than zero, or 0 when there is none
    /// (the "decode a one-hot action" scan the tabular agents use).
    /// </summary>
    internal static int FirstPositiveIndex<T>(Vector<T> values, T threshold)
    {
        return FirstPositiveIndex(values, threshold, out _);
    }

    /// <summary>
    /// First index whose element is strictly greater than zero; <paramref name="found"/> is false
    /// (and the result 0) when no element is positive.
    /// </summary>
    internal static int FirstPositiveIndex<T>(Vector<T> values, T threshold, out bool found)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        found = false;
        if (values.Length == 0)
        {
            return 0;
        }

        // The comparison is 1 where values > 0 and 0 elsewhere (NaN compares false), so the first
        // arg-max of the indicator is the first positive slot, and 0 when no slot is positive.
        var ops = MathHelper.GetNumericOperations<T>();
        var positive = Engine.TensorGreaterThan(values.AsTensor(), threshold);
        int index = Engine.TensorArgMax(positive, 0)[0];
        found = !ops.Equals(positive[index], ops.Zero);
        return index;
    }

    /// <summary>Arithmetic mean of a list of samples (sum, then divide by the count); zero when empty.</summary>
    internal static T Average<T>(IReadOnlyList<T>? samples)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (samples is null || samples.Count == 0)
        {
            return ops.Zero;
        }

        T sum = Engine.Sum(Vector<T>.FromEnumerable(samples));
        return ops.Divide(sum, ops.FromDouble(samples.Count));
    }

    /// <summary>Largest element (first-maximum scan semantics).</summary>
    internal static T Max<T>(Vector<T> values)
    {
        return values[ArgMax(values)];
    }

    /// <summary>
    /// True when every element equals the first one under <see cref="INumericOperations{T}.Equals"/>
    /// (so a NaN anywhere makes it false), the "all actions tied" test of the tabular agents.
    /// </summary>
    internal static bool AllEqual<T>(Vector<T> values, INumericOperations<T> ops)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length <= 1)
        {
            return true;
        }

        var equal = Engine.TensorEquals(values.AsTensor(), values[0]);
        return ops.Equals(Engine.TensorMinValue(equal), ops.One);
    }

    /// <summary>
    /// True when the vectors have the same length and every element pair compares equal (a NaN never does). The
    /// test is min(a == b) = 1 rather than a count of matches in T, which Half cannot hold exactly past 2048.
    /// </summary>
    internal static bool SameValues<T>(Vector<T> a, Vector<T> b)
    {
        if (a is null)
        {
            throw new ArgumentNullException(nameof(a));
        }

        if (b is null)
        {
            throw new ArgumentNullException(nameof(b));
        }

        if (a.Length != b.Length)
        {
            return false;
        }

        if (a.Length == 0)
        {
            return true;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        return ops.Equals(Engine.TensorMinValue(Engine.TensorEquals(a.AsTensor(), b.AsTensor())), ops.One);
    }

    /// <summary>
    /// Gathers actions 0..count-1 of a dictionary-backed table row into a vector (data movement
    /// only; a missing action throws <see cref="KeyNotFoundException"/> like the indexed reads).
    /// </summary>
    internal static Vector<T> Row<T>(Dictionary<int, T> row, int count)
    {
        if (row is null)
        {
            throw new ArgumentNullException(nameof(row));
        }

        return Vector<T>.FromEnumerable(Enumerable.Range(0, count).Select(action => row[action]));
    }

    /// <summary>
    /// The discount powers [1, g, g^2, ..., g^(count-1)], each formed by repeated multiplication
    /// (a cumulative product), exactly like the running "discount *= gamma" scalar update.
    /// </summary>
    internal static Vector<T> DiscountPowers<T>(T gamma, int count)
    {
        if (count <= 0)
        {
            return new Vector<T>(0);
        }

        var factors = Vector<T>.CreateDefault(count, gamma);
        factors[0] = MathHelper.GetNumericOperations<T>().One;
        return Engine.TensorCumProd(factors.AsTensor(), 0).ToVector();
    }

    /// <summary>
    /// The n-step discounted sum sum_i g^i r_i, plus the discount g^n that applies to whatever
    /// follows the last reward (the bootstrap value).
    /// </summary>
    internal static T DiscountedSum<T>(Vector<T> rewards, T gamma, out T nextDiscount)
    {
        if (rewards is null)
        {
            throw new ArgumentNullException(nameof(rewards));
        }

        var powers = DiscountPowers(gamma, rewards.Length + 1);
        nextDiscount = powers[rewards.Length];
        if (rewards.Length == 0)
        {
            return MathHelper.GetNumericOperations<T>().Zero;
        }

        return Engine.DotProduct(powers.GetSubVector(0, rewards.Length), rewards);
    }

    /// <summary>
    /// Scatters a vector into a dictionary-backed table row keyed 0..Length-1 (data movement only).
    /// </summary>
    internal static Dictionary<int, T> ToRow<T>(Vector<T> values)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        var row = new Dictionary<int, T>(values.Length);
        int action = 0;
        foreach (var value in values)
        {
            row[action++] = value;
        }

        return row;
    }

    /// <summary>
    /// Inverse-CDF categorical draw for an already-drawn uniform <paramref name="uniform"/>: the
    /// first index whose running probability sum reaches it (the "r &lt;= cumulative" scan), or
    /// <paramref name="fallback"/> when rounding leaves the total below it.
    /// </summary>
    internal static int SampleFromCdf<T>(Vector<T> probabilities, double uniform, int fallback)
    {
        return SampleFromCdf(probabilities, uniform, fallback, strict: false);
    }

    /// <summary>
    /// Inverse-CDF categorical draw; with <paramref name="strict"/> the first index whose running sum
    /// strictly exceeds the uniform ("r &lt; cumulative"), otherwise the first that reaches it.
    /// </summary>
    internal static int SampleFromCdf<T>(Vector<T> probabilities, double uniform, int fallback, bool strict)
    {
        if (probabilities is null)
        {
            throw new ArgumentNullException(nameof(probabilities));
        }

        if (probabilities.Length == 0)
        {
            return fallback;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var cdf = Engine.TensorCumSum(probabilities.AsTensor(), 0);
        var target = new Tensor<T>(new[] { ops.FromDouble(uniform) }, new[] { 1 });
        int index = Engine.TensorSearchSorted(cdf, target, strict)[0];
        return index < probabilities.Length ? index : fallback;
    }

    /// <summary>
    /// Exact log-softmax of one entry, z_index - max(z) - log(sum exp(z - max(z))) (torch's
    /// Categorical.log_prob).
    /// </summary>
    internal static T LogSoftmaxAt<T>(Vector<T> logits, int index)
    {
        if (logits is null)
        {
            throw new ArgumentNullException(nameof(logits));
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T max = Max(logits);
        T sum = Engine.TensorSum(Engine.TensorExp(Engine.TensorSubtractScalar(logits.AsTensor(), max)));
        return ops.Subtract(ops.Subtract(logits[index], max), ops.Log(sum));
    }

    /// <summary>
    /// Diagonal-Gaussian log-density summed over dimensions:
    /// sum_i [-0.5 ((x_i - mu_i) / sigma_i)^2 - log sigma_i - 0.5 log(2 pi)], sigma = exp(log_std).
    /// </summary>
    internal static T GaussianLogProb<T>(Vector<T> mean, Vector<T> logStd, Vector<T> value)
    {
        if (mean is null)
        {
            throw new ArgumentNullException(nameof(mean));
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var z = Engine.Divide(Engine.Subtract(value, mean), Engine.Exp(logStd));
        var perDimension = Engine.Subtract(Engine.Multiply(Engine.Multiply(z, z), ops.FromDouble(-0.5)), logStd);
        return ops.Subtract(Engine.Sum(perDimension), ops.FromDouble(0.5 * Math.Log(2.0 * Math.PI) * mean.Length));
    }

    /// <summary>
    /// log(p + floor) evaluated in double: a floor such as 1e-10 rounds to zero in Half, where the T sum would
    /// give log(0) = -infinity for a zero probability.
    /// </summary>
    internal static T LogWithFloor<T>(T probability, double floor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.FromDouble(Math.Log(ops.ToDouble(probability) + floor));
    }

    /// <summary>
    /// 1 - <paramref name="epsilon"/> in T, or Half's largest value below one (1 - 2^-11) when that rounds to one:
    /// a 1e-6 clip margin is lost in Half, which would collapse the clip onto the boundary.
    /// </summary>
    internal static T BelowOne<T>(double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T bound = ops.FromDouble(1.0 - epsilon);
        return ops.LessThan(bound, ops.One) ? bound : ops.FromDouble(1.0 - 4.8828125e-4);
    }

    /// <summary>
    /// The backward linear recurrence y_t = x_t + c_t y_{t+1} with y_T = 0, the shape of every discounted return
    /// and of GAE (Schulman et al. 2016, Eq. 16). Each step is one engine op on single-element slices, newest first.
    /// </summary>
    internal static Vector<T> ReverseLinearRecurrence<T>(Vector<T> inputs, Vector<T> coefficients)
    {
        if (inputs is null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (coefficients is null)
        {
            throw new ArgumentNullException(nameof(coefficients));
        }

        if (coefficients.Length != inputs.Length)
        {
            throw new ArgumentException("One coefficient per input is required.", nameof(coefficients));
        }

        int n = inputs.Length;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        var x = inputs.AsTensor();
        var c = coefficients.AsTensor();
        var steps = new Tensor<T>[n];
        var running = Engine.TensorSlice(x, [n - 1], [1]);
        steps[n - 1] = running;
        for (int t = n - 2; t >= 0; t--)
        {
            running = Engine.TensorAdd(Engine.TensorSlice(x, [t], [1]), Engine.TensorMultiply(Engine.TensorSlice(c, [t], [1]), running));
            steps[t] = running;
        }

        return Engine.TensorConcatenate(steps, 0).ToVector();
    }

    /// <summary>
    /// Generalized advantage estimation (Schulman et al. 2016, Eq. 16; Stable-Baselines3's
    /// compute_returns_and_advantage): delta_t = r_t + gamma V'_t - V_t and A_t = delta_t + gamma lambda A_{t+1},
    /// with the chain cut after every step that ends an episode or the rollout. <paramref name="nextValues"/> holds
    /// V'_t, the value bootstrapped after step t: zero for a terminated step, V(s_{t+1}) otherwise (including a
    /// truncated one). Returns are A_t + V_t.
    /// </summary>
    internal static (Vector<T> Advantages, Vector<T> Returns) Gae<T>(
        Vector<T> rewards, Vector<T> values, Vector<T> nextValues, bool[] chainCut, T gamma, T lambda)
    {
        if (rewards is null)
        {
            throw new ArgumentNullException(nameof(rewards));
        }

        if (chainCut is null)
        {
            throw new ArgumentNullException(nameof(chainCut));
        }

        if (rewards.Length == 0)
        {
            return (new Vector<T>(0), new Vector<T>(0));
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var deltas = Engine.Subtract(Engine.Add(rewards, Engine.Multiply(nextValues, gamma)), values);
        var coefficients = Engine.TensorWhere(
            chainCut,
            Vector<T>.CreateDefault(rewards.Length, ops.Zero).AsTensor(),
            Vector<T>.CreateDefault(rewards.Length, ops.Multiply(gamma, lambda)).AsTensor()).ToVector();
        var advantages = ReverseLinearRecurrence(deltas, coefficients);
        return (advantages, Engine.Add(advantages, values));
    }

    /// <summary>
    /// n-step discounted returns (Mnih et al. 2016, Algorithm S3): R_t = r_t + gamma R_{t+1}, and
    /// R_t = r_t + gamma V'_t after a step that ends an episode or the rollout, where <paramref name="nextValues"/>
    /// holds V'_t (zero for a terminated step, V(s_{t+1}) for a truncated one); entries of other steps are unused.
    /// </summary>
    internal static Vector<T> DiscountedReturns<T>(Vector<T> rewards, Vector<T> nextValues, bool[] chainCut, T gamma)
    {
        if (rewards is null)
        {
            throw new ArgumentNullException(nameof(rewards));
        }

        if (chainCut is null)
        {
            throw new ArgumentNullException(nameof(chainCut));
        }

        if (rewards.Length == 0)
        {
            return new Vector<T>(0);
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var zeros = Vector<T>.CreateDefault(rewards.Length, ops.Zero).AsTensor();
        var bootstrap = Engine.TensorWhere(chainCut, Engine.Multiply(nextValues, gamma).AsTensor(), zeros).ToVector();
        var coefficients = Engine.TensorWhere(chainCut, zeros, Vector<T>.CreateDefault(rewards.Length, gamma).AsTensor()).ToVector();
        return ReverseLinearRecurrence(Engine.Add(rewards, bootstrap), coefficients);
    }

    /// <summary>
    /// Log-density of a tanh-squashed diagonal Gaussian (Haarnoja et al. 2018, SAC, Appendix C; Stable-Baselines3's
    /// TanhBijector): a = tanh(u), u ~ N(mu, sigma), log pi(a) = log N(atanh(a); mu, sigma) - sum_i log(1 - a_i^2 + 1e-6),
    /// with a clipped just inside (-1, 1) first. The clip bound stays strictly below one in T (see
    /// <see cref="BelowOne{T}"/>), so a saturated action gives a finite log-density in Half too.
    /// </summary>
    internal static T SquashedGaussianLogProb<T>(Vector<T> mean, Vector<T> logStd, Vector<T> action)
    {
        if (action is null)
        {
            throw new ArgumentNullException(nameof(action));
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T upper = BelowOne<T>(1e-6);
        var clipped = Engine.Clamp(action, ops.Negate(upper), upper);
        T gaussianLogProb = GaussianLogProb(mean, logStd, Engine.Atanh(clipped));
        var oneMinusSquare = Engine.Subtract(Vector<T>.CreateDefault(clipped.Length, ops.One), Engine.Multiply(clipped, clipped));
        var correction = Engine.TensorLog(Engine.TensorAddScalar(oneMinusSquare.AsTensor(), ops.FromDouble(1e-6)));
        return ops.Subtract(gaussianLogProb, Engine.TensorSum(correction));
    }

    /// <summary>
    /// Advantage normalisation (A - mean) / (std + 1e-8) with the unbiased standard deviation, as Stable-Baselines3
    /// and CleanRL normalise (torch.std). When std + 1e-8 is not positive in T (Half rounds 1e-8 to zero, so equal
    /// advantages would give 0 / 0 = NaN) the centred advantages, all zero there, are returned as they are.
    /// </summary>
    internal static List<T> NormalizeAdvantages<T>(List<T> advantages)
    {
        if (advantages is null)
        {
            throw new ArgumentNullException(nameof(advantages));
        }

        if (advantages.Count == 0)
        {
            return new List<T>();
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T std = StatisticsHelper<T>.CalculateStandardDeviation(advantages);
        T mean = Average(advantages);
        var centred = Engine.Subtract(Vector<T>.FromEnumerable(advantages), Vector<T>.CreateDefault(advantages.Count, mean));
        T denominator = ops.Add(std, ops.FromDouble(1e-8));
        return ops.GreaterThan(denominator, ops.Zero)
            ? Engine.Divide(centred, denominator).ToList()
            : centred.ToList();
    }

    /// <summary>
    /// The UCB1 exploration ratio ln(t) / N(a) per arm (Auer et al. 2002), formed from the integer step and visit
    /// counts in double and converted to T once: in Half, t and N saturate (inexact past 2048, infinite past 65504),
    /// and ln(inf) would give every arm an infinite bonus. The ratio of an unvisited arm (N = 0) is not meaningful;
    /// callers pick unvisited arms before using it.
    /// </summary>
    internal static Vector<T> UcbRatio<T>(int totalSteps, Vector<int> counts)
    {
        if (counts is null)
        {
            throw new ArgumentNullException(nameof(counts));
        }

        if (counts.Length == 0)
        {
            return new Vector<T>(0);
        }

        var visits = counts.AsTensor().Cast<double>().ToVector();
        var ratio = Engine.Divide(Vector<double>.CreateDefault(counts.Length, Math.Log(totalSteps)), visits);
        return ratio.AsTensor().Cast<T>().ToVector();
    }

    /// <summary>
    /// <paramref name="count"/> standard-normal draws from <paramref name="random"/>, taken in index order exactly as
    /// a per-element <c>random.NextGaussian()</c> loop takes them (so a seeded stream is consumed identically).
    /// </summary>
    internal static Vector<T> StandardNormal<T>(Random random, int count)
    {
        if (random is null)
        {
            throw new ArgumentNullException(nameof(random));
        }

        if (count <= 0)
        {
            return new Vector<T>(0);
        }

        var draws = new Vector<double>(Enumerable.Range(0, count).Select(_ => AiDotNet.Extensions.RandomExtensions.NextGaussian(random)));
        return draws.AsTensor().Cast<T>().ToVector();
    }

    /// <summary>
    /// A diagonal-Gaussian sample mean + exp(log_std) * N(0, 1), one standard-normal draw per dimension in index order.
    /// </summary>
    internal static Vector<T> SampleGaussian<T>(Vector<T> mean, Vector<T> logStd, Random random)
    {
        if (mean is null)
        {
            throw new ArgumentNullException(nameof(mean));
        }

        var noise = StandardNormal<T>(random, mean.Length);
        return Engine.Add(mean, Engine.Multiply(Engine.Exp(logStd), noise));
    }

    /// <summary>
    /// <paramref name="count"/> draws of <c>MathHelper.GetNormalRandom(0, 1, random)</c>, taken in index order exactly
    /// as a per-element loop takes them (the agents' Gaussian exploration stream).
    /// </summary>
    internal static Vector<T> NormalRandom<T>(Random random, int count)
    {
        if (random is null)
        {
            throw new ArgumentNullException(nameof(random));
        }

        if (count <= 0)
        {
            return new Vector<T>(0);
        }

        var ops = MathHelper.GetNumericOperations<T>();
        return NormalRandom(random, count, ops.Zero, ops.One);
    }

    /// <summary>
    /// <paramref name="count"/> draws of <c>MathHelper.GetNormalRandom(mean, stdDev, random)</c>, in index order.
    /// </summary>
    internal static Vector<T> NormalRandom<T>(Random random, int count, T mean, T stdDev)
    {
        if (random is null)
        {
            throw new ArgumentNullException(nameof(random));
        }

        if (count <= 0)
        {
            return new Vector<T>(0);
        }

        return new Vector<T>(Enumerable.Range(0, count).Select(_ => MathHelper.GetNormalRandom(mean, stdDev, random)));
    }

    /// <summary>
    /// The agents' Gaussian action head: mean + exp(log_std) * GetNormalRandom(0, 1) per dimension when training
    /// (draws in index order), the mean otherwise. <paramref name="output"/> is [means | log stds].
    /// </summary>
    internal static Vector<T> GaussianHeadAction<T>(Vector<T> output, int actionSize, bool training, Random random)
    {
        if (output is null)
        {
            throw new ArgumentNullException(nameof(output));
        }

        var mean = output.GetSubVector(0, actionSize);
        if (!training)
        {
            return mean;
        }

        var std = Engine.Exp(output.GetSubVector(actionSize, actionSize));
        return Engine.Add(mean, Engine.Multiply(std, NormalRandom<T>(random, actionSize)));
    }

    /// <summary>The first element of a network output (a value head's V(s), a critic's Q(s, a)).</summary>
    internal static T FirstValue<T>(Tensor<T> output)
    {
        if (output is null)
        {
            throw new ArgumentNullException(nameof(output));
        }

        return output.ToVector()[0];
    }

    /// <summary>Stacks the first <paramref name="width"/> elements of each vector as a [count, width] batch.</summary>
    internal static Tensor<T> Batch<T>(IEnumerable<Vector<T>> rows, int width)
    {
        if (rows is null)
        {
            throw new ArgumentNullException(nameof(rows));
        }

        return StackRows(rows.Select(row => Head(row, width)).ToList());
    }

    /// <summary>The <paramref name="count"/> equal-width rows of a batched output, in order.</summary>
    internal static List<Vector<T>> Rows<T>(Tensor<T> batch, int count)
    {
        if (batch is null)
        {
            throw new ArgumentNullException(nameof(batch));
        }

        var flat = batch.ToVector();
        int width = count == 0 ? 0 : flat.Length / count;
        return Enumerable.Range(0, count).Select(row => flat.GetSubVector(row * width, width)).ToList();
    }

    /// <summary>
    /// Picks one entry per row of a [count, width] batch: element (row, columns[row]).
    /// </summary>
    internal static Vector<T> PickPerRow<T>(Tensor<T> batch, int count, int width, IReadOnlyList<int> columns)
    {
        if (batch is null)
        {
            throw new ArgumentNullException(nameof(batch));
        }

        if (count == 0)
        {
            return new Vector<T>(0);
        }

        var flatIndices = Enumerable.Range(0, count).Select(row => row * width + columns[row]).ToArray();
        var flat = Engine.Reshape(batch, [count * width]);
        return Engine.TensorGather(flat, new Tensor<int>(flatIndices, new[] { count }), 0).ToVector();
    }

    /// <summary>The first column of a [count, width] batched output (each row's V(s) or Q(s, a)).</summary>
    internal static Vector<T> FirstColumn<T>(Tensor<T> batch, int count)
    {
        if (batch is null)
        {
            throw new ArgumentNullException(nameof(batch));
        }

        int width = count == 0 ? 0 : batch.ToVector().Length / count;
        return PickPerRow(batch, count, width, new int[count]);
    }

    /// <summary>
    /// One-step TD targets: r for a terminal transition, r + gamma * bootstrap otherwise (a terminal row's
    /// bootstrap value is never used).
    /// </summary>
    internal static Vector<T> TdTargets<T>(Vector<T> rewards, Vector<T> bootstrap, bool[] done, T gamma)
    {
        if (rewards is null)
        {
            throw new ArgumentNullException(nameof(rewards));
        }

        if (rewards.Length == 0)
        {
            return new Vector<T>(0);
        }

        var bootstrapped = Engine.Add(rewards, Engine.Multiply(bootstrap, gamma));
        return Engine.TensorWhere(done, rewards.AsTensor(), bootstrapped.AsTensor()).ToVector();
    }

    /// <summary>
    /// Q-regression targets: the current [count, actions] Q-values with each row's taken action replaced by its TD
    /// target (the other actions regress onto themselves).
    /// </summary>
    internal static Tensor<T> ReplaceTakenActionValues<T>(Tensor<T> currentQ, int count, int actions,
        IReadOnlyList<int> takenActions, Vector<T> tdTargets)
    {
        if (currentQ is null)
        {
            throw new ArgumentNullException(nameof(currentQ));
        }

        if (count == 0)
        {
            return new Tensor<T>([0, actions]);
        }

        var current = new Tensor<T>([count, actions], currentQ.ToVector().GetSubVector(0, count * actions));
        var taken = StackRows(takenActions.Select(action => OneHot<T>(actions, action)).ToList());
        var targets = Engine.TensorTile(Engine.Reshape(tdTargets.AsTensor(), [count, 1]), [1, actions]);
        return Engine.TensorWhere(taken, targets, current);
    }

    /// <summary>
    /// Polyak target update tau * source + (1 - tau) * target over the target's parameters (Lillicrap et al. 2016).
    /// </summary>
    internal static Vector<T> SoftUpdate<T>(Vector<T> source, Vector<T> target, T tau)
    {
        if (target is null)
        {
            throw new ArgumentNullException(nameof(target));
        }

        var ops = MathHelper.GetNumericOperations<T>();
        T oneMinusTau = ops.Subtract(ops.One, tau);
        return Engine.Add(Engine.Multiply(Head(source, target.Length), tau), Engine.Multiply(target, oneMinusTau));
    }

    /// <summary>A one-hot vector of the given length with a one at <paramref name="index"/>.</summary>
    internal static Vector<T> OneHot<T>(int length, int index)
    {
        var oneHot = new Vector<T>(length);
        oneHot[index] = MathHelper.GetNumericOperations<T>().One;
        return oneHot;
    }

    /// <summary>Numerically stable softmax: exp(x - max) / sum(exp(x - max)).</summary>
    internal static Vector<T> StableSoftmax<T>(Vector<T> logits)
    {
        if (logits is null)
        {
            throw new ArgumentNullException(nameof(logits));
        }

        T max = Max(logits);
        var exp = Engine.TensorExp(Engine.TensorSubtractScalar(logits.AsTensor(), max));
        T sum = Engine.TensorSum(exp);
        return Engine.TensorDivideScalar(exp, sum).ToVector();
    }

    /// <summary>
    /// Adds per-element metrics named prefix + index (+ suffix), interleaving the series per index
    /// in the given order, exactly like a per-index loop that writes each series in turn (data
    /// movement only).
    /// </summary>
    internal static void AddIndexedMetrics<T>(Dictionary<string, T> metrics, int count, params (string Prefix, Vector<T> Values)[] series)
    {
        if (metrics is null)
        {
            throw new ArgumentNullException(nameof(metrics));
        }

        if (series is null)
        {
            throw new ArgumentNullException(nameof(series));
        }

        var enumerators = new IEnumerator<T>[series.Length];
        for (int k = 0; k < series.Length; k++)
        {
            enumerators[k] = ((IEnumerable<T>)series[k].Values).GetEnumerator();
        }

        try
        {
            for (int index = 0; index < count; index++)
            {
                string suffix = index.ToString(System.Globalization.CultureInfo.InvariantCulture);
                for (int k = 0; k < series.Length; k++)
                {
                    if (!enumerators[k].MoveNext())
                    {
                        throw new ArgumentException($"Metric series '{series[k].Prefix}' has fewer than {count} elements.", nameof(series));
                    }

                    metrics[series[k].Prefix + suffix] = enumerators[k].Current;
                }
            }
        }
        finally
        {
            foreach (var enumerator in enumerators)
            {
                enumerator.Dispose();
            }
        }
    }

    /// <summary>Converts integer counts to T in one bulk pass.</summary>
    internal static Vector<T> CountsAs<T>(Vector<int> counts)
    {
        if (counts is null)
        {
            throw new ArgumentNullException(nameof(counts));
        }

        return counts.Length == 0 ? new Vector<T>(0) : counts.AsTensor().Cast<T>().ToVector();
    }

    /// <summary>
    /// Writes a vector back into an existing dictionary-backed row, keys 0..Length-1, keeping the
    /// row object (data movement only).
    /// </summary>
    internal static void Fill<T>(Dictionary<int, T> row, Vector<T> values)
    {
        if (row is null)
        {
            throw new ArgumentNullException(nameof(row));
        }

        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        int action = 0;
        foreach (var value in values)
        {
            row[action++] = value;
        }
    }

    /// <summary>True when any element is strictly greater than <paramref name="threshold"/> (NaN never is).</summary>
    internal static bool AnyGreaterThan<T>(Vector<T> values, T threshold)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length == 0)
        {
            return false;
        }

        var above = Engine.TensorGreaterThan(values.AsTensor(), threshold);
        var ops = MathHelper.GetNumericOperations<T>();
        return !ops.Equals(Engine.TensorMaxValue(above), ops.Zero);
    }

    /// <summary>
    /// One scalar temporal-difference step on a single table entry: current + lr * (target - current).
    /// Sequential planning/trace loops apply it to one entry per iteration, where each update can
    /// change the inputs of the next, so it is inherently scalar rather than element-wise.
    /// </summary>
    internal static T TdStep<T>(INumericOperations<T> ops, T current, T target, T learningRate)
    {
        return ops.Add(current, ops.Multiply(learningRate, ops.Subtract(target, current)));
    }

    /// <summary>The one-step bootstrap target reward + discount * next.</summary>
    internal static T BootstrapTarget<T>(INumericOperations<T> ops, T reward, T discount, T next)
    {
        return ops.Add(reward, ops.Multiply(discount, next));
    }

    /// <summary>
    /// The "best = floor; if (v &gt; best) best = v" scan: the first index whose value is the largest
    /// value strictly above <paramref name="floor"/> (NaN never qualifies), or 0 with
    /// <paramref name="best"/> = floor when nothing is above it.
    /// </summary>
    internal static int ArgMaxAbove<T>(Vector<T> values, T floor, out T best)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        best = floor;
        if (values.Length == 0)
        {
            return 0;
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var tensor = values.AsTensor();
        var floorTensor = Engine.TensorFullLike(tensor, floor);

        // NaN entries can never win a strict greater-than test: give them the floor value.
        var cleaned = Engine.TensorWhere(Engine.TensorIsNan(tensor), floorTensor, tensor);
        int index = Engine.TensorArgMax(cleaned, 0)[0];
        T candidate = cleaned[index];
        if (!ops.GreaterThan(candidate, floor))
        {
            return 0;
        }

        best = candidate;
        return index;
    }

    /// <summary>The largest value strictly above <paramref name="floor"/>, or floor (NaN is ignored).</summary>
    internal static T MaxAbove<T>(Vector<T> values, T floor)
    {
        ArgMaxAbove(values, floor, out T best);
        return best;
    }

    /// <summary>The first <paramref name="count"/> elements of a vector (the vector itself when it is exactly that long).</summary>
    internal static Vector<T> Head<T>(Vector<T> values, int count)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        return values.Length == count ? values : values.GetSubVector(0, count);
    }

    /// <summary>Linear Q-values for every action: W * phi, using the first W.Columns features.</summary>
    internal static Vector<T> LinearQValues<T>(Matrix<T> weights, Vector<T> features)
    {
        if (weights is null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        return Engine.MatrixVectorMultiply(weights, Head(features, weights.Columns));
    }

    /// <summary>Linear Q-value of one action: w_a . phi over the first W.Columns features.</summary>
    internal static T LinearQValue<T>(Matrix<T> weights, int action, Vector<T> features)
    {
        if (weights is null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        return Engine.DotProduct(Engine.GetRow(weights, action), Head(features, weights.Columns));
    }

    /// <summary>The semi-gradient step w_a += scale * phi over the first W.Columns features.</summary>
    internal static void AddScaledToRow<T>(Matrix<T> weights, int action, Vector<T> features, T scale)
    {
        if (weights is null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        var row = Engine.GetRow(weights, action);
        var step = Engine.Multiply(Head(features, weights.Columns), scale);
        Engine.SetRow(weights, action, Engine.Add(row, step));
    }

    /// <summary>
    /// The LSTD(Q) normal equations A = sum phi (phi - gamma phi')^T and b = sum phi r, given the
    /// per-sample feature rows phi, difference rows d = phi - gamma phi', and rewards (A = Phi^T D,
    /// b = Phi^T r). Zero when there are no samples.
    /// </summary>
    internal static (Matrix<T> A, Vector<T> b) LeastSquaresTdSystem<T>(
        List<Vector<T>> phiRows, List<Vector<T>> diffRows, List<T> rewards, int featureSize)
    {
        if (phiRows is null || diffRows is null || rewards is null)
        {
            throw new ArgumentNullException(phiRows is null ? nameof(phiRows) : diffRows is null ? nameof(diffRows) : nameof(rewards));
        }

        var ops = MathHelper.GetNumericOperations<T>();
        if (phiRows.Count == 0)
        {
            return (Matrix<T>.CreateDefault(featureSize, featureSize, ops.Zero), Vector<T>.CreateDefault(featureSize, ops.Zero));
        }

        var phiT = Engine.MatrixTranspose(Matrix<T>.FromRowVectors(phiRows));
        var a = Engine.MatrixMultiply(phiT, Matrix<T>.FromRowVectors(diffRows));
        var b = Engine.MatrixVectorMultiply(phiT, Vector<T>.FromEnumerable(rewards));
        return (a, b);
    }

    /// <summary>A + value * I (ridge regularisation of a square matrix).</summary>
    internal static Matrix<T> AddToDiagonal<T>(Matrix<T> matrix, T value)
    {
        if (matrix is null)
        {
            throw new ArgumentNullException(nameof(matrix));
        }

        return Engine.MatrixAdd(matrix, Engine.MatrixMultiplyScalar(Matrix<T>.CreateIdentity(matrix.Rows), value));
    }

    /// <summary>Frobenius norm sqrt(sum w^2) of a matrix.</summary>
    internal static T FrobeniusNorm<T>(Matrix<T> weights)
    {
        if (weights is null)
        {
            throw new ArgumentNullException(nameof(weights));
        }

        var ops = MathHelper.GetNumericOperations<T>();
        if (weights.Rows == 0 || weights.Columns == 0)
        {
            return ops.Zero;
        }

        var flat = Vector<T>.FromSpan(weights.AsSpan());
        return ops.Sqrt(Engine.DotProduct(flat, flat));
    }

    /// <summary>Concatenates vectors end to end.</summary>
    internal static Vector<T> Concat<T>(params Vector<T>[] parts)
    {
        if (parts is null)
        {
            throw new ArgumentNullException(nameof(parts));
        }

        var tensors = new Tensor<T>[parts.Length];
        int total = 0;
        for (int p = 0; p < parts.Length; p++)
        {
            tensors[p] = parts[p].AsTensor();
            total += parts[p].Length;
        }

        if (total == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorConcatenate(tensors, 0).ToVector();
    }

    /// <summary>Stacks equal-length vectors as the rows of a [count, length] tensor.</summary>
    internal static Tensor<T> StackRows<T>(IReadOnlyList<Vector<T>> rows)
    {
        if (rows is null)
        {
            throw new ArgumentNullException(nameof(rows));
        }

        var tensors = new Tensor<T>[rows.Count];
        for (int r = 0; r < rows.Count; r++)
        {
            tensors[r] = rows[r].AsTensor();
        }

        return Engine.TensorStack(tensors, 0);
    }

    /// <summary>Packs a list of scalars into a rank-1 tensor of the given shape.</summary>
    internal static Tensor<T> FromList<T>(IReadOnlyList<T> values, params int[] shape)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        var vector = Vector<T>.FromEnumerable(values);
        return shape.Length == 0 ? vector.AsTensor() : new Tensor<T>(shape, vector);
    }
}
