using AiDotNet.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// BYOL (Bootstrap Your Own Latent) Loss - a simple cosine similarity loss.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BYOL uses a simple Mean Squared Error (MSE) loss between
/// normalized predictions and normalized targets. Unlike contrastive methods, it doesn't
/// require negative samples.</para>
///
/// <para><b>Key insight:</b> BYOL avoids collapse through asymmetric architecture
/// (predictor on one branch) and momentum updates, not through negative samples.</para>
///
/// <para><b>Loss formula:</b></para>
/// <code>
/// L = 2 - 2 * (p · sg(z')) / (||p|| * ||sg(z')||)
///   = MSE(normalize(p), normalize(sg(z')))
/// </code>
/// <para>where p is prediction, z' is target projection, and sg() means stop-gradient.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning", "https://arxiv.org/abs/2006.07733", Year = 2020, Authors = "Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko")]
public class BYOLLoss<T> : ContrastiveLossBase<T>
{

    private readonly bool _normalize;
    private readonly bool _symmetric;

    /// <summary>
    /// Initializes a new instance of the BYOLLoss class.
    /// </summary>
    /// <param name="normalize">Whether to L2-normalize embeddings (default: true).</param>
    /// <param name="symmetric">Whether to use symmetric loss (default: true for BYOL).</param>
    public BYOLLoss(bool normalize = true, bool symmetric = true)
    {
        _normalize = normalize;
        _symmetric = symmetric;
    }

    /// <summary>
    /// Computes the BYOL loss between online predictions and target projections.
    /// </summary>
    /// <param name="onlinePrediction">Predictions from online network [batch_size, dim].</param>
    /// <param name="targetProjection">Projections from target network [batch_size, dim] (stop-gradient applied).</param>
    /// <returns>The computed loss value.</returns>
    /// <remarks>
    /// Differentiable: 2 - 2*cos(p, z) averaged over the batch, assembled from IEngine ops so the
    /// result carries tape history. This previously summed host-side dot products into a bare
    /// scalar, which could be reported but never backpropagated.
    /// </remarks>
    public override Tensor<T> ComputeLoss(Tensor<T> onlinePrediction, Tensor<T> targetProjection)
    {
        // Shapes, not just ranks: the engine broadcasts, so a [8, 1] against a [8, 768] returns a
        // finite loss computed against the wrong target instead of failing.
        ContrastiveTapeOps<T>.RequireMatchingRank2(
            onlinePrediction, targetProjection, "BYOL", nameof(onlinePrediction), nameof(targetProjection));

        if (onlinePrediction is null) throw new ArgumentNullException(nameof(onlinePrediction));
        if (targetProjection is null) throw new ArgumentNullException(nameof(targetProjection));

        var p2 = _normalize ? ObjectiveOps.L2NormalizeRows(onlinePrediction) : onlinePrediction;
        var z2 = _normalize ? ObjectiveOps.L2NormalizeRows(targetProjection) : targetProjection;

        // rowwise cosine similarity -> mean -> 2 - 2*mean
        var prod = Engine.TensorMultiply(p2, z2);
        var perRow = Engine.ReduceSum(prod, new[] { 1 }, keepDims: false);
        var meanCos = Engine.TensorDivideScalar(
            Engine.ReduceSum(perRow, null, keepDims: false),
            NumOps.FromDouble(onlinePrediction.Shape[0]));
        var twice = Engine.TensorMultiplyScalar(meanCos, NumOps.FromDouble(2.0));
        return Engine.TensorNegate(Engine.TensorAddScalar(twice, NumOps.FromDouble(-2.0)));
    }

    private T ComputeLossScalarLegacy(Tensor<T> onlinePrediction, Tensor<T> targetProjection)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        if (onlinePrediction is null) throw new ArgumentNullException(nameof(onlinePrediction));
        if (targetProjection is null) throw new ArgumentNullException(nameof(targetProjection));

        var batchSize = onlinePrediction.Shape[0];

        var p = _normalize ? L2Normalize(onlinePrediction) : onlinePrediction;
        var z = _normalize ? L2Normalize(targetProjection) : targetProjection;

        // Per-sample loss = 2 - 2 * cos(p, z), averaged over the batch
        var cosineSim = Engine.ReduceSum(Engine.TensorMultiply(p, z), new[] { 1 }, keepDims: false);
        var losses = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(cosineSim, NumOps.FromDouble(-2.0)), NumOps.FromDouble(2.0));

        return NumOps.Divide(Engine.TensorSum(losses), NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes symmetric BYOL loss for both views.
    /// </summary>
    /// <param name="pred1">Predictions from view 1 going to view 2.</param>
    /// <param name="proj2">Target projections from view 2.</param>
    /// <param name="pred2">Predictions from view 2 going to view 1.</param>
    /// <param name="proj1">Target projections from view 1.</param>
    /// <returns>The symmetric loss.</returns>
    /// <remarks>
    /// Returns a tape-carrying tensor, like <see cref="ComputeLoss"/>. This is the form BYOL
    /// actually trains on — both view orderings averaged — so it is the one that most needs to be
    /// differentiable.
    /// </remarks>
    public Tensor<T> ComputeSymmetricLoss(
        Tensor<T> pred1, Tensor<T> proj2,
        Tensor<T> pred2, Tensor<T> proj1)
    {
        var loss1 = ComputeLoss(pred1, proj2);
        var loss2 = ComputeLoss(pred2, proj1);

        return Engine.TensorMultiplyScalar(Engine.TensorAdd(loss1, loss2), NumOps.FromDouble(0.5));
    }

    /// <summary>
    /// Computes BYOL loss with gradients for backpropagation.
    /// </summary>
    public (T loss, Tensor<T> gradPrediction) ComputeLossWithGradients(
        Tensor<T> onlinePrediction, Tensor<T> targetProjection)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var batchSize = onlinePrediction.Shape[0];

        var p = _normalize ? L2Normalize(onlinePrediction) : onlinePrediction;
        var z = _normalize ? L2Normalize(targetProjection) : targetProjection;

        // Per-sample cosine similarity of the normalized rows, [batch, 1]
        var cosineSim = Engine.ReduceSum(Engine.TensorMultiply(p, z), new[] { 1 }, keepDims: true);

        // Norm of the ORIGINAL (unnormalized) input for the chain rule, [batch, 1]
        var origNormSq = Engine.ReduceSum(Engine.TensorSquare(onlinePrediction), new[] { 1 }, keepDims: true);
        var origNorm = Engine.TensorSqrt(Engine.TensorAddScalar(origNormSq, NumOps.FromDouble(1e-8)));

        // Loss = 2 - 2 * cos_sim, averaged over the batch
        var losses = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(cosineSim, NumOps.FromDouble(-2.0)), NumOps.FromDouble(2.0));
        var avgLoss = NumOps.Divide(Engine.TensorSum(losses), NumOps.FromDouble(batchSize));

        // Gradient w.r.t. unnormalized onlinePrediction:
        // d/da_d [cos(a,b)] = (z_d - cos_sim * p_d) / ||a||
        // where p = a/||a||, z = b/||b||
        // So dL/da_d = -2 * (z_d - cos_sim * p_d) / ||a||, then scaled by 1/batchSize
        var direction = Engine.TensorSubtract(z, Engine.TensorMultiply(cosineSim, p));
        var gradP = Engine.TensorDivide(Engine.TensorMultiplyScalar(direction, NumOps.FromDouble(-2.0)), origNorm);
        var scale = NumOps.FromDouble(1.0 / batchSize);

        return (avgLoss, Engine.TensorMultiplyScalar(gradP, scale));
    }

    /// <summary>
    /// Computes the mean squared error between normalized embeddings.
    /// </summary>
    /// <remarks>
    /// For normalized vectors: MSE = 2 - 2*cos_sim, which is equivalent to the BYOL loss.
    /// </remarks>
    public T ComputeMSELoss(Tensor<T> prediction, Tensor<T> target)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var batchSize = prediction.Shape[0];
        var dim = prediction.Shape[1];

        var p = _normalize ? L2Normalize(prediction) : prediction;
        var z = _normalize ? L2Normalize(target) : target;

        // Mean squared difference over every element
        T totalLoss = Engine.TensorSumOfSquares(Engine.TensorSubtract(p, z));
        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize * dim));
    }

    private Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        // x / sqrt(sum(x^2) + 1e-8) per row
        var sumSquared = Engine.ReduceSum(Engine.TensorSquare(tensor), new[] { 1 }, keepDims: true);
        var norm = Engine.TensorSqrt(Engine.TensorAddScalar(sumSquared, NumOps.FromDouble(1e-8)));
        return Engine.TensorDivide(tensor, norm);
    }

    /// <summary>Row-wise L2 normalization on the tape.</summary>
    /// <remarks>
    /// The epsilon sits inside the square root so a zero row scales by a finite value rather than
    /// dividing by zero, and its derivative stays finite there too.
    /// </remarks>
}
