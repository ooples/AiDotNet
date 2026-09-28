using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;

namespace AiDotNet.Document.OCR.TextDetection;

/// <summary>
/// PSENet's training objective (Wang et al. 2019, Eq. 5): L = lambda L_c + (1 - lambda) L_s with lambda = 0.7,
/// both terms Dice losses over the sigmoid of the kernel logits.
/// </summary>
/// <remarks>
/// <para>
/// Layout is [batch, kernels, H, W] (or [kernels, H, W]), kernels ordered smallest to largest, so the LAST
/// channel is the complete text map S_n. Dice is the paper's D(S, G) = 2 sum(S G) / (sum S^2 + sum G^2), with
/// the reference implementation's 0.001 added to each denominator term, computed per image and per kernel.
/// </para>
/// <para>
/// L_c uses Online Hard Example Mining on the complete map: every positive pixel plus the 3x as many
/// highest-scoring negatives. L_s scores the shrunk kernels only where the predicted complete map says
/// text (S_n &gt; 0.5), so they are not penalised outside a text region. The masks are constants; gradient
/// flows through the scores alone, as in the reference.
/// </para>
/// </remarks>
[LossCategory(LossCategory.Segmentation)]
[LossTask(LossTask.InstanceSegmentation)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, HandlesImbalancedData = true, RequiresProbabilityInputs = false, TestInputFormat = LossTestInputFormat.RawLogits, ExpectedOutput = OutputType.Logits)]
public sealed class PSENetLoss<T> : LossFunctionBase<T>
{
    /// <summary>Weight of the complete-text-map term (the paper's lambda).</summary>
    public const double CompleteMapWeight = 0.7;

    /// <summary>Hard negatives kept per positive by OHEM.</summary>
    public const int NegativeRatio = 3;

    private const double Smooth = 0.001;

    /// <inheritdoc />
    /// <remarks>A flat vector carries no kernel layout, so this is the smoothed Dice loss of the whole vector.</remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        double inter = 0, pp = 0, aa = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double s = 1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(predicted[i])));
            double g = NumOps.ToDouble(actual[i]);
            inter += s * g; pp += s * s; aa += g * g;
        }
        return NumOps.FromDouble(1.0 - 2.0 * inter / (pp + Smooth + aa + Smooth));
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (predicted.Length != target.Length)
            throw new ArgumentException("PSENet targets must match the prediction's [batch, kernels, H, W] layout.", nameof(target));

        // A flat map (rank 1 or 2) is one image with one kernel, i.e. OHEM Dice alone.
        if (predicted.Rank < 3)
            predicted = Engine.Reshape(predicted, new[] { 1, 1, predicted.Length, 1 });
        else if (predicted.Rank == 3)
            predicted = Engine.Reshape(predicted, new[] { 1, predicted.Shape[0], predicted.Shape[1], predicted.Shape[2] });
        if (predicted.Rank != 4)
            throw new ArgumentException($"Expected [batch, kernels, H, W], got rank {predicted.Rank}.", nameof(predicted));
        var shape = predicted.Shape.ToArray();
        if (target.Rank != 4 || !target.Shape.ToArray().SequenceEqual(shape))
            target = Engine.Reshape(target, shape);

        int batch = shape[0], kernels = shape[1];
        var scores = Engine.Sigmoid(predicted);
        var mask = BuildMask(scores.ToArray(), target.ToArray(), shape);

        var s = Engine.TensorMultiply(scores, mask);
        var g = Engine.TensorMultiply(target, mask);
        int[] spatial = { 2, 3 };
        var inter = Engine.ReduceSum(Engine.TensorMultiply(s, g), spatial, keepDims: false);
        var ss = Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(s, s), spatial, keepDims: false), NumOps.FromDouble(Smooth));
        var gg = Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(g, g), spatial, keepDims: false), NumOps.FromDouble(Smooth));
        var dice = Engine.TensorDivide(Engine.TensorMultiplyScalar(inter, NumOps.FromDouble(2.0)), Engine.TensorAdd(ss, gg));

        // One weight per (image, kernel): lambda on the complete map, (1 - lambda) shared by the shrunk
        // kernels, each averaged over the batch. The weights sum to 1, so L = 1 - sum(w * D).
        var weights = new Tensor<T>(new[] { batch, kernels });
        for (int b = 0; b < batch; b++)
            for (int k = 0; k < kernels; k++)
            {
                double w = kernels == 1 ? 1.0
                    : k == kernels - 1 ? CompleteMapWeight : (1.0 - CompleteMapWeight) / (kernels - 1);
                weights[b, k] = NumOps.FromDouble(w / batch);
            }

        var weighted = Engine.ReduceSum(Engine.TensorMultiply(dice, weights), new[] { 0, 1 }, keepDims: false);
        return Engine.ScalarMinusTensor(NumOps.One, Engine.Reshape(weighted, new[] { 1 }));
    }

    private Tensor<T> BuildMask(T[] scores, T[] target, int[] shape)
    {
        int batch = shape[0], kernels = shape[1], plane = shape[2] * shape[3];
        var mask = new Tensor<T>(shape);
        for (int b = 0; b < batch; b++)
        {
            int complete = (b * kernels + kernels - 1) * plane;
            var ohem = OhemMask(scores, target, complete, plane);
            for (int p = 0; p < plane; p++)
            {
                mask[complete + p] = ohem[p] ? NumOps.One : NumOps.Zero;
                // Shrunk kernels count only inside the predicted text region.
                var text = NumOps.ToDouble(scores[complete + p]) > 0.5 ? NumOps.One : NumOps.Zero;
                for (int k = 0; k < kernels - 1; k++)
                    mask[(b * kernels + k) * plane + p] = text;
            }
        }
        return mask;
    }

    // Reference ohem_single: every positive plus the NegativeRatio x as many highest-scoring negatives; every
    // pixel when there are no positives or no negatives to rank.
    private bool[] OhemMask(T[] scores, T[] target, int offset, int plane)
    {
        var keep = new bool[plane];
        int positives = 0;
        var negatives = new List<double>();
        for (int p = 0; p < plane; p++)
        {
            if (NumOps.ToDouble(target[offset + p]) > 0.5) positives++;
            else negatives.Add(NumOps.ToDouble(scores[offset + p]));
        }
        int negativeCount = Math.Min(positives * NegativeRatio, negatives.Count);
        if (positives == 0 || negativeCount == 0)
        {
            for (int p = 0; p < plane; p++) keep[p] = true;
            return keep;
        }
        negatives.Sort((x, y) => y.CompareTo(x));
        double threshold = negatives[negativeCount - 1];
        for (int p = 0; p < plane; p++)
            keep[p] = NumOps.ToDouble(target[offset + p]) > 0.5 || NumOps.ToDouble(scores[offset + p]) >= threshold;
        return keep;
    }
}