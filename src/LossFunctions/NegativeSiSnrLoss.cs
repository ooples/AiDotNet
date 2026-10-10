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
/// A separator's output channels have no fixed order, so for <c>[batch, sources, samples]</c> each example
/// takes the assignment of estimates to references with the highest mean SI-SNR, found exactly by the
/// Hungarian method on the pairwise SI-SNR matrix; the gradient flows only through that assignment, which is
/// permutation-invariant training. Rank 1 or 2 inputs are a single source.
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
        int batch = estimate.Shape[0], sources = estimate.Shape[1], samples = estimate.Shape[2];

        if (sources == 1)
        {
            return Engine.ReduceMean(Engine.TensorNegate(SiSnr(estimate, reference, axis: 2)), new[] { 0, 1 }, keepDims: false);
        }

        // Every estimate against every reference at once: pairs[b, i, j] = SI-SNR(estimate i, reference j).
        // That is S^2 scores, where scoring each permutation separately was S! * S and stopped being usable
        // past a handful of sources.
        var pairShape = new[] { batch, sources, sources, samples };
        var pairs = SiSnr(
            Engine.TensorBroadcastTo(Engine.Reshape(estimate, new[] { batch, sources, 1, samples }), pairShape),
            Engine.TensorBroadcastTo(Engine.Reshape(reference, new[] { batch, 1, sources, samples }), pairShape),
            axis: 3);

        // Each example's best assignment (utterance-level PIT, Kolbaek et al. 2017), chosen on the values by the
        // Hungarian method. The mask is a constant, so the gradient flows only through the chosen pairs.
        var choice = new Tensor<T>(new[] { batch, sources, sources });
        var cost = new double[sources, sources];
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < sources; i++)
            {
                for (int j = 0; j < sources; j++)
                {
                    double score = NumOps.ToDouble(pairs[b, i, j]);
                    // A NaN score must not stall the assignment; it can only be chosen as a last resort.
                    cost[i, j] = double.IsNaN(score) ? double.MaxValue / 4 : -score;
                }
            }

            int[] assignment = MinimumCostAssignment(cost, sources);
            for (int i = 0; i < sources; i++) choice[b, i, assignment[i]] = NumOps.One;
        }

        var chosen = Engine.ReduceSum(Engine.TensorMultiply(pairs, choice), new[] { 1, 2 }, keepDims: false);
        var perExample = Engine.TensorMultiplyScalar(chosen, NumOps.FromDouble(-1.0 / sources));
        return Engine.ReduceMean(perExample, new[] { 0 }, keepDims: false);
    }

    /// <summary>SI-SNR in dB of two equally shaped tensors, measured along <paramref name="axis"/> (the samples).</summary>
    private Tensor<T> SiSnr(Tensor<T> estimate, Tensor<T> reference, int axis)
    {
        var axes = new[] { axis };
        var est = Engine.TensorSubtract(estimate, Engine.ReduceMean(estimate, axes, keepDims: true));
        var refc = Engine.TensorSubtract(reference, Engine.ReduceMean(reference, axes, keepDims: true));
        var eps = NumOps.FromDouble(Epsilon);

        var dot = Engine.ReduceSum(Engine.TensorMultiply(est, refc), axes, keepDims: true);
        var refEnergy = Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(refc, refc), axes, keepDims: true), eps);
        var projected = Engine.TensorMultiply(Engine.TensorDivide(dot, refEnergy), refc);
        var noise = Engine.TensorSubtract(est, projected);

        var targetEnergy = Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(projected, projected), axes, keepDims: false), eps);
        var noiseEnergy = Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(noise, noise), axes, keepDims: false), eps);
        return Engine.TensorMultiplyScalar(Engine.TensorLog(Engine.TensorDivide(targetEnergy, noiseEnergy)),
            NumOps.FromDouble(DecibelsPerNeper));
    }

    private Tensor<T> AsBatchSourcesSamples(Tensor<T> tensor) => tensor.Shape.Length switch
    {
        1 => Engine.Reshape(tensor, new[] { 1, 1, tensor.Shape[0] }),
        2 => Engine.Reshape(tensor, new[] { tensor.Shape[0], 1, tensor.Shape[1] }),
        3 => tensor,
        _ => throw new ArgumentException(
            $"SI-SNR expects [samples], [batch, samples] or [batch, sources, samples]; got rank {tensor.Shape.Length}.")
    };

    /// <summary>
    /// Minimum-cost perfect matching of rows to columns (the Hungarian method with potentials, O(n^3)).
    /// </summary>
    /// <returns>For each row, the column it is matched to.</returns>
    private static int[] MinimumCostAssignment(double[,] cost, int n)
    {
        // 1-based potentials u (rows) and v (columns); match[j] is the row holding column j, way[] the
        // augmenting path back to the free column 0.
        var u = new double[n + 1];
        var v = new double[n + 1];
        var match = new int[n + 1];
        var way = new int[n + 1];
        for (int row = 1; row <= n; row++)
        {
            match[0] = row;
            int column = 0;
            var minimum = Enumerable.Repeat(double.PositiveInfinity, n + 1).ToArray();
            var used = new bool[n + 1];
            do
            {
                used[column] = true;
                int current = match[column], next = 0;
                double delta = double.PositiveInfinity;
                for (int j = 1; j <= n; j++)
                {
                    if (used[j]) continue;
                    double reduced = cost[current - 1, j - 1] - u[current] - v[j];
                    if (reduced < minimum[j])
                    {
                        minimum[j] = reduced;
                        way[j] = column;
                    }

                    if (minimum[j] < delta)
                    {
                        delta = minimum[j];
                        next = j;
                    }
                }

                for (int j = 0; j <= n; j++)
                {
                    if (used[j])
                    {
                        u[match[j]] += delta;
                        v[j] -= delta;
                    }
                    else
                    {
                        minimum[j] -= delta;
                    }
                }

                column = next;
            }
            while (match[column] != 0);

            do
            {
                int previous = way[column];
                match[column] = match[previous];
                column = previous;
            }
            while (column != 0);
        }

        var assignment = new int[n];
        for (int j = 1; j <= n; j++) assignment[match[j] - 1] = j - 1;
        return assignment;
    }
}