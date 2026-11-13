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
        var s = Engine.Sigmoid(predicted.AsTensor());
        var g = actual.AsTensor();
        T inter = Engine.TensorSum(Engine.TensorMultiply(s, g));
        T smooth = NumOps.FromDouble(Smooth);
        T denominator = NumOps.Add(NumOps.Add(Engine.TensorSum(Engine.TensorMultiply(s, s)), smooth),
            NumOps.Add(Engine.TensorSum(Engine.TensorMultiply(g, g)), smooth));
        return NumOps.Subtract(NumOps.One, NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(2.0), inter), denominator));
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
        var mask = BuildMask(predicted, scores, target, shape);

        var s = Engine.TensorMultiply(scores, mask);
        var g = Engine.TensorMultiply(target, mask);
        int[] spatial = { 2, 3 };
        var inter = Engine.ReduceSum(Engine.TensorMultiply(s, g), spatial, keepDims: false);
        var ss = Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(s, s), spatial, keepDims: false), NumOps.FromDouble(Smooth));
        var gg = Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(g, g), spatial, keepDims: false), NumOps.FromDouble(Smooth));
        var dice = Engine.TensorDivide(Engine.TensorMultiplyScalar(inter, NumOps.FromDouble(2.0)), Engine.TensorAdd(ss, gg));

        // One weight per (image, kernel): lambda on the complete map, (1 - lambda) shared by the shrunk
        // kernels, each averaged over the batch. The weights sum to 1, so L = 1 - sum(w * D).
        var kernelWeights = Enumerable.Range(0, kernels)
            .Select(k => (kernels == 1 ? 1.0 : k == kernels - 1 ? CompleteMapWeight : (1.0 - CompleteMapWeight) / (kernels - 1)) / batch)
            .Select(NumOps.FromDouble).ToArray();
        var weights = Engine.TensorBroadcastTo(new Tensor<T>(new[] { 1, kernels }, new Vector<T>(kernelWeights)), new[] { batch, kernels });

        var weighted = Engine.ReduceSum(Engine.TensorMultiply(dice, weights), new[] { 0, 1 }, keepDims: false);
        return Engine.ScalarMinusTensor(NumOps.One, Engine.Reshape(weighted, new[] { 1 }));
    }

    // The masks are constants: they are built from detached copies, so gradient flows through the scores alone.
    private Tensor<T> BuildMask(Tensor<T> logits, Tensor<T> scores, Tensor<T> target, int[] shape)
    {
        int batch = shape[0], kernels = shape[1], plane = shape[2] * shape[3];
        var flatShape = new[] { batch, plane };
        Tensor<T> Complete(Tensor<T> x) => Engine.Reshape(
            Engine.TensorSlice(Engine.Reshape(Engine.StopGradient(x), new[] { batch, kernels, plane }), new[] { 0, kernels - 1, 0 }, new[] { batch, 1, plane }),
            flatShape);

        var ohem = OhemMask(Complete(logits), Complete(target), batch, plane);
        // Shrunk kernels count only inside the predicted text region (S_n > 0.5, i.e. a positive logit).
        var text = Engine.TensorGreaterThan(Complete(scores), NumOps.FromDouble(0.5));
        var parts = Enumerable.Repeat(text, kernels - 1).Append(ohem)
            .Select(m => Engine.Reshape(m, new[] { batch, 1, plane })).ToArray();
        return Engine.Reshape(Engine.TensorConcatenate(parts, 1), shape);
    }

    // Reference ohem_single (whai362/PSENet): every positive plus the pixels scoring at least the k-th hardest
    // negative, k = min(NegativeRatio x positives, negatives); every pixel when there are no positives or no
    // negatives to rank. Negatives are ranked by the raw logits the reference passes in, not by the sigmoid,
    // whose saturation at 1.0 would tie the hardest negatives and keep more than k of them.
    private Tensor<T> OhemMask(Tensor<T> logits, Tensor<T> target, int batch, int plane)
    {
        var shape = new[] { batch, plane };
        var positive = Engine.TensorGreaterThan(target, NumOps.FromDouble(0.5));
        int[] positives = Engine.ReduceSum(positive, new[] { 1 }, false).Cast<int>().ToArray();
        int[] rank = positives.Select(p => Math.Min(p * NegativeRatio, plane - p)).ToArray();
        var useAll = DocumentTensorOps.IntTensor<T>(positives.Select((p, b) => p == 0 || rank[b] == 0 ? 1 : 0).ToArray());

        // Hardest negatives first: positives are pushed below every negative before sorting each image.
        T floor = AiDotNet.Audio.AudioEngineOps.NegativeInfinityOrMin<T>();
        var negatives = Engine.TensorWhere(positive, Tensor<T>.CreateDefault(shape, floor), logits);
        var (sorted, _) = Engine.TensorSort(negatives, 1, true);
        var kth = DocumentTensorOps.OneHot<T>(Engine, rank.Select(k => Math.Max(k, 1) - 1).ToArray(), plane);
        var threshold = Engine.ReduceSum(Engine.TensorWhere(kth, sorted, Tensor<T>.CreateDefault(shape, NumOps.Zero)), new[] { 1 }, true);
        var hard = Engine.TensorGreaterEqual(logits, Engine.TensorBroadcastTo(threshold, shape));
        var keep = Engine.TensorMax(positive, hard);
        var all = Engine.TensorBroadcastTo(Engine.Reshape(useAll, new[] { batch, 1 }), shape);
        return Engine.TensorMax(keep, all);
    }
}
