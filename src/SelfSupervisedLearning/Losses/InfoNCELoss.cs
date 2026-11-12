using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// InfoNCE (Noise Contrastive Estimation) Loss for contrastive learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> InfoNCE is the loss function used in MoCo (Momentum Contrast).
/// It's similar to NT-Xent but designed to work efficiently with a large memory queue of
/// negative samples.</para>
///
/// <para><b>Key differences from NT-Xent:</b></para>
/// <list type="bullet">
/// <item>Uses a memory queue for negatives instead of in-batch negatives</item>
/// <item>Typically uses more negatives (65536) but smaller batch sizes</item>
/// <item>Asymmetric: query from online encoder, keys from momentum encoder</item>
/// </list>
///
/// <para><b>Loss formula:</b></para>
/// <code>
/// L = -log( exp(q·k+ / τ) / (exp(q·k+ / τ) + Σ exp(q·k- / τ)) )
/// </code>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Momentum Contrast for Unsupervised Visual Representation Learning", "https://arxiv.org/abs/1911.05722", Year = 2020, Authors = "Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick")]
public class InfoNCELoss<T> : ContrastiveLossBase<T>
{

    private readonly double _temperature;
    private readonly bool _normalize;

    /// <summary>
    /// Gets the temperature parameter.
    /// </summary>
    public double Temperature => _temperature;

    /// <summary>
    /// Initializes a new instance of the InfoNCELoss class.
    /// </summary>
    /// <param name="temperature">Temperature scaling parameter (default: 0.07).</param>
    /// <param name="normalize">Whether to L2-normalize embeddings (default: true).</param>
    public InfoNCELoss(double temperature = 0.07, bool normalize = true)
    {
        if (temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be positive");

        _temperature = temperature;
        _normalize = normalize;
    }

    /// <summary>
    /// Computes the InfoNCE loss using queries, positive keys, and negative keys from memory bank.
    /// </summary>
    /// <param name="queries">Query embeddings from online encoder [batch_size, dim].</param>
    /// <param name="positiveKeys">Positive key embeddings from momentum encoder [batch_size, dim].</param>
    /// <param name="negativeKeys">Negative key embeddings from memory bank [num_negatives, dim].</param>
    /// <returns>The computed loss value.</returns>
    public T ComputeLoss(Tensor<T> queries, Tensor<T> positiveKeys, Tensor<T> negativeKeys)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        if (queries is null) throw new ArgumentNullException(nameof(queries));
        if (positiveKeys is null) throw new ArgumentNullException(nameof(positiveKeys));
        if (negativeKeys is null) throw new ArgumentNullException(nameof(negativeKeys));

        var batchSize = queries.Shape[0];

        // Normalize if required
        var q = _normalize ? L2Normalize(queries) : queries;
        var kPos = _normalize ? L2Normalize(positiveKeys) : positiveKeys;
        var kNeg = _normalize ? L2Normalize(negativeKeys) : negativeKeys;

        // Logits [batch, 1 + numNegatives], positive in column 0
        var logits = Engine.TensorDivideScalar(
            PositiveFirstSimilarities(q, kPos, kNeg), NumOps.FromDouble(_temperature));

        // Loss = logsumexp(logits) - positive logit (numerically stable log-softmax)
        var (logSumExp, _) = RowLogSumExpAndSoftmax(logits);
        var posLogits = Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { batchSize, 1 });
        T totalLoss = Engine.TensorSum(Engine.TensorSubtract(logSumExp, posLogits));

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Row-wise q . kPos in column 0 followed by q . kNeg_j, as a [batch, 1 + numNegatives] tensor.
    /// </summary>
    private Tensor<T> PositiveFirstSimilarities(Tensor<T> q, Tensor<T> kPos, Tensor<T> kNeg)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var posSims = Engine.ReduceSum(Engine.TensorMultiply(q, kPos), new[] { 1 }, keepDims: true);
        var negSims = Engine.TensorMatMul(q, Engine.TensorTranspose(kNeg));
        return Engine.TensorConcatenate(new[] { posSims, negSims }, 1);
    }

    /// <summary>
    /// Per row: max + log(sum exp(logits - max)) as [batch, 1], and the softmax probabilities.
    /// </summary>
    private (Tensor<T> LogSumExp, Tensor<T> Probabilities) RowLogSumExpAndSoftmax(Tensor<T> logits)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var maxLogit = Engine.ReduceMax(logits, new[] { 1 }, keepDims: true);
        var exps = Engine.TensorExp(Engine.TensorSubtract(logits, maxLogit));
        var sumExp = Engine.ReduceSum(exps, new[] { 1 }, keepDims: true);
        var logSumExp = Engine.TensorAdd(maxLogit, Engine.TensorLog(sumExp));
        return (logSumExp, Engine.TensorDivide(exps, sumExp));
    }

    /// <summary>
    /// Computes InfoNCE loss with in-batch negatives (SimCLR style but asymmetric).
    /// </summary>
    /// <param name="queries">Query embeddings [batch_size, dim].</param>
    /// <param name="keys">Key embeddings [batch_size, dim].</param>
    /// <returns>The computed loss value.</returns>
    public T ComputeLossInBatch(Tensor<T> queries, Tensor<T> keys)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        if (queries is null) throw new ArgumentNullException(nameof(queries));
        if (keys is null) throw new ArgumentNullException(nameof(keys));

        var batchSize = queries.Shape[0];

        var q = _normalize ? L2Normalize(queries) : queries;
        var k = _normalize ? L2Normalize(keys) : keys;

        // All logits: q_i . k_j / tau
        var logits = Engine.TensorDivideScalar(
            Engine.TensorMatMul(q, Engine.TensorTranspose(k)), NumOps.FromDouble(_temperature));

        // Positive is at index i (diagonal)
        var (logSumExp, _) = RowLogSumExpAndSoftmax(logits);
        var posLogits = Engine.TensorDiagonal(logits).Reshape(batchSize, 1);
        T totalLoss = Engine.TensorSum(Engine.TensorSubtract(logSumExp, posLogits));

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes InfoNCE loss with in-batch negatives and gradients.
    /// </summary>
    /// <param name="queries">Query embeddings [batch_size, dim].</param>
    /// <param name="keys">Key embeddings [batch_size, dim].</param>
    /// <returns>Tuple of (loss, gradQueries, gradKeys).</returns>
    public (T loss, Tensor<T> gradQueries, Tensor<T> gradKeys) ComputeLossInBatchWithGradients(
        Tensor<T> queries, Tensor<T> keys)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        if (queries is null) throw new ArgumentNullException(nameof(queries));
        if (keys is null) throw new ArgumentNullException(nameof(keys));

        var batchSize = queries.Shape[0];

        var q = _normalize ? L2Normalize(queries) : queries;
        var k = _normalize ? L2Normalize(keys) : keys;

        var invTemp = NumOps.FromDouble(1.0 / _temperature);

        // All logits: q_i . k_j / tau
        var logits = Engine.TensorMultiplyScalar(Engine.TensorMatMul(q, Engine.TensorTranspose(k)), invTemp);

        // Softmax probabilities; positive is at index i (diagonal)
        var (logSumExp, probs) = RowLogSumExpAndSoftmax(logits);
        var posLogits = Engine.TensorDiagonal(logits).Reshape(batchSize, 1);
        T totalLoss = Engine.TensorSum(Engine.TensorSubtract(logSumExp, posLogits));

        // dL/dlogit_ij = p_ij - 1 for the positive (j == i), p_ij for negatives, times 1/tau
        var gradScale = Engine.TensorMultiplyScalar(
            Engine.TensorSubtract(probs, Engine.TensorEye<T>(batchSize)), invTemp);

        // Gradient w.r.t. query_i: sum_j gradScale_ij k_j; w.r.t. key_j: sum_i gradScale_ij q_i
        var gradQ = Engine.TensorMatMul(gradScale, k);
        var gradK = Engine.TensorMatMul(Engine.TensorTranspose(gradScale), q);

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        var scale = NumOps.FromDouble(1.0 / batchSize);

        return (avgLoss, Engine.TensorMultiplyScalar(gradQ, scale), Engine.TensorMultiplyScalar(gradK, scale));
    }

    /// <summary>
    /// Computes InfoNCE loss with gradients.
    /// </summary>
    public (T loss, Tensor<T> gradQueries, Tensor<T> gradPositiveKeys) ComputeLossWithGradients(
        Tensor<T> queries, Tensor<T> positiveKeys, Tensor<T> negativeKeys)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var batchSize = queries.Shape[0];
        var numNegatives = negativeKeys.Shape[0];

        var q = _normalize ? L2Normalize(queries) : queries;
        var kPos = _normalize ? L2Normalize(positiveKeys) : positiveKeys;
        var kNeg = _normalize ? L2Normalize(negativeKeys) : negativeKeys;

        var invTemp = NumOps.FromDouble(1.0 / _temperature);

        // Logits [batch, 1 + numNegatives], positive in column 0
        var logits = Engine.TensorMultiplyScalar(PositiveFirstSimilarities(q, kPos, kNeg), invTemp);

        // Softmax
        var (logSumExp, probs) = RowLogSumExpAndSoftmax(logits);
        var posLogits = Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { batchSize, 1 });
        T totalLoss = Engine.TensorSum(Engine.TensorSubtract(logSumExp, posLogits));

        // Gradients: positive weight (p_pos - 1) / tau, negative weights p_neg / tau
        var posProbMinusOne = Engine.TensorAddScalar(
            Engine.TensorSlice(probs, new[] { 0, 0 }, new[] { batchSize, 1 }), NumOps.Negate(NumOps.One));
        var posWeight = Engine.TensorMultiplyScalar(posProbMinusOne, invTemp);
        var negWeights = Engine.TensorMultiplyScalar(
            Engine.TensorSlice(probs, new[] { 0, 1 }, new[] { batchSize, numNegatives }), invTemp);

        // Gradient w.r.t. query: posWeight * kPos + sum_j negWeight_j * kNeg_j
        var gradQ = Engine.TensorAdd(Engine.TensorMultiply(posWeight, kPos), Engine.TensorMatMul(negWeights, kNeg));

        // Gradient w.r.t. positive key
        var gradKPos = Engine.TensorMultiply(posWeight, q);

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        var scale = NumOps.FromDouble(1.0 / batchSize);

        return (avgLoss, Engine.TensorMultiplyScalar(gradQ, scale), Engine.TensorMultiplyScalar(gradKPos, scale));
    }

    /// <summary>
    /// Computes accuracy of the contrastive task (useful for monitoring).
    /// </summary>
    public double ComputeAccuracy(Tensor<T> queries, Tensor<T> positiveKeys, Tensor<T> negativeKeys)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var batchSize = queries.Shape[0];

        var q = _normalize ? L2Normalize(queries) : queries;
        var kPos = _normalize ? L2Normalize(positiveKeys) : positiveKeys;
        var kNeg = _normalize ? L2Normalize(negativeKeys) : negativeKeys;

        // A query is correct when no negative is strictly more similar than its positive
        var posSims = Engine.ReduceSum(Engine.TensorMultiply(q, kPos), new[] { 1 }, keepDims: true);
        var negSims = Engine.TensorMatMul(q, Engine.TensorTranspose(kNeg));
        var beatsPositive = Engine.TensorGreaterThan(Engine.TensorSubtract(negSims, posSims), NumOps.Zero);
        var violations = Engine.ReduceSum(beatsPositive, new[] { 1 }, keepDims: false);
        int correct = batchSize - Engine.TensorCountNonzero(violations);

        return (double)correct / batchSize;
    }

    private Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        // x / sqrt(sum(x^2) + 1e-8) per row
        var sumSquared = Engine.ReduceSum(Engine.TensorSquare(tensor), new[] { 1 }, keepDims: true);
        var norm = Engine.TensorSqrt(Engine.TensorAddScalar(sumSquared, NumOps.FromDouble(1e-8)));
        return Engine.TensorDivide(tensor, norm);
    }
    /// <summary>
    /// Differentiable in-batch InfoNCE: queries against keys, positives on the diagonal.
    /// </summary>
    /// <remarks>
    /// One-directional, matching MoCo (He et al. 2020), which contrasts a query against its
    /// positive key and the other keys as negatives. Built entirely from IEngine ops so it carries
    /// tape history; the scalar <see cref="ComputeLossInBatch"/> above cannot be trained on.
    /// </remarks>
    public override Tensor<T> ComputeLoss(Tensor<T> view1, Tensor<T> view2)
    {
        ContrastiveTapeOps<T>.RequireMatchingRank2(
            view1, view2, "InfoNCE", nameof(view1), nameof(view2));

        var logits = ObjectiveOps.SimilarityMatrix(view1, view2, _temperature, _normalize);
        var logProbs = ObjectiveOps.LogSoftmax(logits, axis: 1);
        return Engine.TensorNegate(ObjectiveOps.MeanDiagonal(logProbs));
    }

    /// <summary>Row-wise L2 normalization on the tape.</summary>

    /// <summary>Constant identity selector picking each query's own key. Constant DATA, no gradient.</summary>
}
