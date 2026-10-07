using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Negative scale-invariant signal-to-noise ratio, with utterance-level permutation-invariant training
/// (Luo &amp; Mesgarani 2019, Sec. III-D; Le Roux et al. 2019).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// For an estimate <c>ŝ</c> and a reference <c>s</c>, both made zero-mean over time:
/// <c>s_target = (⟨ŝ, s⟩ / ‖s‖²) s</c>, <c>e_noise = ŝ − s_target</c>, and
/// <c>SI-SNR = 10 log10(‖s_target‖² / ‖e_noise‖²)</c>. Rescaling the estimate does not change it, so a
/// separator is not rewarded for matching loudness. The loss is its negative, so it can be below zero.
/// </para>
/// <para>
/// A separator's output channels have no fixed order, so for <c>[batch, sources, samples]</c> every
/// assignment of estimates to references is scored and each example takes its best one; the gradient
/// flows only through that assignment, which is permutation-invariant training. Rank 1 or 2 inputs are
/// a single source.
/// </para>
/// <para>
/// <b>For Beginners:</b> SI-SNR asks "how much of the estimate is the right signal, and how much is
/// leftover noise", ignoring overall volume. Higher is better, so training minimizes its negative.
/// </para>
/// </remarks>
[LossCategory(LossCategory.Reconstruction)]
[LossTask(LossTask.Denoising)]
[LossProperty(IsNonNegative = false, ZeroForIdentical = false, ZeroDerivativeForIdentical = false,
    HasStandardGradientSign = false, ExpectedOutput = OutputType.Continuous)]
public class NegativeSiSnrLoss<T> : LossFunctionBase<T>
{
    // Keeps both energies away from zero, as the reference implementations do.
    private const double Epsilon = 1e-8;
    private static readonly double DecibelsPerNeper = 10.0 / Math.Log(10.0);

    /// <summary>
    /// Negative SI-SNR of ONE estimated signal against ONE reference.
    /// </summary>
    /// <remarks>
    /// A flat vector carries no source axis, so it is scored as a single signal with no permutation
    /// search. For several sources keep the [batch, sources, samples] layout and use
    /// <see cref="ComputeTapeLoss"/>, which scores every source assignment and keeps the best; flattening
    /// separated sources into this overload would score them as one concatenated waveform.
    /// </remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        int n = predicted.Length;
        double meanP = 0, meanA = 0;
        for (int i = 0; i < n; i++)
        {
            meanP += NumOps.ToDouble(predicted[i]);
            meanA += NumOps.ToDouble(actual[i]);
        }

        meanP /= Math.Max(n, 1);
        meanA /= Math.Max(n, 1);
        double dot = 0, refEnergy = 0;
        for (int i = 0; i < n; i++)
        {
            double p = NumOps.ToDouble(predicted[i]) - meanP, a = NumOps.ToDouble(actual[i]) - meanA;
            dot += p * a;
            refEnergy += a * a;
        }

        double scale = dot / (refEnergy + Epsilon);
        double targetEnergy = 0, noiseEnergy = 0;
        for (int i = 0; i < n; i++)
        {
            double p = NumOps.ToDouble(predicted[i]) - meanP, a = NumOps.ToDouble(actual[i]) - meanA;
            double target = scale * a;
            targetEnergy += target * target;
            noiseEnergy += (p - target) * (p - target);
        }

        return NumOps.FromDouble(-10.0 * Math.Log10((targetEnergy + Epsilon) / (noiseEnergy + Epsilon)));
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (!predicted.Shape.ToArray().SequenceEqual(target.Shape.ToArray()))
            throw new ArgumentException(
                $"SI-SNR needs matching shapes; got [{string.Join(", ", predicted.Shape.ToArray())}] and " +
                $"[{string.Join(", ", target.Shape.ToArray())}].", nameof(target));

        var estimate = AsBatchSourcesSamples(predicted);
        var reference = AsBatchSourcesSamples(target);
        int batch = estimate.Shape[0], sources = estimate.Shape[1];

        // Per permutation, the mean negative SI-SNR over sources: [batch] each.
        var permutations = Permutations(sources);
        var perPermutation = new Tensor<T>[permutations.Count];
        for (int p = 0; p < permutations.Count; p++)
        {
            var permuted = sources == 1 ? reference : Reorder(reference, permutations[p]);
            var negative = Engine.TensorNegate(SiSnr(estimate, permuted));
            perPermutation[p] = Engine.ReduceMean(negative, new[] { 1 }, keepDims: true);
        }

        var losses = permutations.Count == 1 ? perPermutation[0] : Engine.TensorConcatenate(perPermutation, axis: 1);
        if (permutations.Count == 1)
        {
            return Engine.ReduceMean(losses, new[] { 0, 1 }, keepDims: false);
        }

        // Each example's best assignment, chosen on the values; the mask is a constant, so the gradient flows
        // only through the chosen permutation's loss.
        var choice = new Tensor<T>(new[] { batch, permutations.Count });
        for (int b = 0; b < batch; b++)
        {
            int best = 0;
            double bestValue = double.PositiveInfinity;
            for (int p = 0; p < permutations.Count; p++)
            {
                double value = NumOps.ToDouble(losses[b, p]);
                if (value < bestValue)
                {
                    bestValue = value;
                    best = p;
                }
            }

            choice[b, best] = NumOps.One;
        }

        var chosen = Engine.ReduceSum(Engine.TensorMultiply(losses, choice), new[] { 1 }, keepDims: false);
        return Engine.ReduceMean(chosen, new[] { 0 }, keepDims: false);
    }

    /// <summary>SI-SNR in dB for every (batch, source) pair of two [batch, sources, samples] tensors.</summary>
    private Tensor<T> SiSnr(Tensor<T> estimate, Tensor<T> reference)
    {
        var axis = new[] { 2 };
        var est = Engine.TensorSubtract(estimate, Engine.ReduceMean(estimate, axis, keepDims: true));
        var refc = Engine.TensorSubtract(reference, Engine.ReduceMean(reference, axis, keepDims: true));
        var eps = NumOps.FromDouble(Epsilon);

        var dot = Engine.ReduceSum(Engine.TensorMultiply(est, refc), axis, keepDims: true);
        var refEnergy = Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(refc, refc), axis, keepDims: true), eps);
        var projected = Engine.TensorMultiply(Engine.TensorDivide(dot, refEnergy), refc);
        var noise = Engine.TensorSubtract(est, projected);

        var targetEnergy = Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(projected, projected), axis, keepDims: false), eps);
        var noiseEnergy = Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(noise, noise), axis, keepDims: false), eps);
        return Engine.TensorMultiplyScalar(Engine.TensorLog(Engine.TensorDivide(targetEnergy, noiseEnergy)),
            NumOps.FromDouble(DecibelsPerNeper));
    }

    /// <summary>The sources of a [batch, sources, samples] tensor in the given order.</summary>
    private Tensor<T> Reorder(Tensor<T> tensor, int[] order)
    {
        var parts = new Tensor<T>[order.Length];
        for (int i = 0; i < order.Length; i++)
        {
            parts[i] = Engine.TensorNarrow(tensor, dim: 1, start: order[i], length: 1);
        }

        return Engine.TensorConcatenate(parts, axis: 1);
    }

    private Tensor<T> AsBatchSourcesSamples(Tensor<T> tensor) => tensor.Shape.Length switch
    {
        1 => Engine.Reshape(tensor, new[] { 1, 1, tensor.Shape[0] }),
        2 => Engine.Reshape(tensor, new[] { tensor.Shape[0], 1, tensor.Shape[1] }),
        3 => tensor,
        _ => throw new ArgumentException(
            $"SI-SNR expects [samples], [batch, samples] or [batch, sources, samples]; got rank {tensor.Shape.Length}.")
    };

    private static List<int[]> Permutations(int count)
    {
        var result = new List<int[]>();
        var current = Enumerable.Range(0, count).ToArray();
        Permute(current, 0, result);
        return result;
    }

    private static void Permute(int[] items, int start, List<int[]> result)
    {
        if (start == items.Length)
        {
            result.Add((int[])items.Clone());
            return;
        }

        for (int i = start; i < items.Length; i++)
        {
            (items[start], items[i]) = (items[i], items[start]);
            Permute(items, start + 1, result);
            (items[start], items[i]) = (items[i], items[start]);
        }
    }
}