using System;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// A classifier meta-learner's loss, expressed over the EMBEDDING network's output: the algorithm's own head or
/// metric turns the embeddings into logits, and cross-entropy scores those logits against class indices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>What this exists to fix.</b> A classifier meta-learner trains an embedding network that emits features,
/// not predictions: the prediction is a head (ANIL, BOIL), a distance to class prototypes (prototypical
/// networks) or an attention over the support set (matching networks) applied to those features. Asking the
/// embedding network for its gradient under the configured loss compares an embedding with a label and trains
/// the embedding to BE the label - the head, metric or attention never reaches the meta-objective.
/// </para>
/// <para>
/// Passed as the <c>lossOverride</c> of <c>MetaLearnerBase.ComputeGradients</c>, this loss makes the embedding
/// network differentiate the composed objective <c>L(g(f(x)), y)</c>. The closure builds the logits with engine
/// tensor ops, so the tape carries the chain rule through <c>g</c> to the embeddings.
/// </para>
/// <para>
/// <b>Whatever the closure reads is a constant here.</b> The derivative is with respect to the embeddings only;
/// a head or a set of prototypes computed outside the closure is treated as fixed, exactly as a first-order
/// meta-gradient treats it. Prototypes that should carry gradient must be computed INSIDE the closure, from the
/// embeddings it is handed.
/// </para>
/// </remarks>
internal sealed class EmbeddingClassificationLoss<T> : ILossFunction<T>
{
    private readonly Func<Tensor<T>, Tensor<T>> _logits;
    private readonly ILossFunction<T> _score;
    private readonly Tensor<T>? _scoredTarget;

    /// <summary>
    /// Creates the composed loss.
    /// </summary>
    /// <param name="logits">
    /// Builds <c>[rows, classes]</c> logits from <c>[rows, width]</c> embeddings using engine tensor ops.
    /// </param>
    /// <param name="score">
    /// The loss that scores the logits against class indices - the learner's configured loss. Null means
    /// cross-entropy on logits, the classification loss of every paper these learners implement.
    /// </param>
    /// <param name="scoredTarget">
    /// The class indices the logits are scored against, when they are not the rows the embedding network sees: a
    /// prototypical episode embeds support and query rows together but scores only the query rows. Null scores the
    /// target the caller passes.
    /// </param>
    internal EmbeddingClassificationLoss(
        Func<Tensor<T>, Tensor<T>> logits, ILossFunction<T>? score = null, Tensor<T>? scoredTarget = null)
    {
        _logits = logits ?? throw new ArgumentNullException(nameof(logits));
        _score = score ?? new CrossEntropyWithLogitsLoss<T>();
        _scoredTarget = scoredTarget;
    }

    /// <inheritdoc/>
    /// <remarks>The target is one class index per example, <c>[rows]</c>.</remarks>
    public Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));
        return _score.ComputeTapeLoss(_logits(AsRows(predicted, target.Length)), _scoredTarget ?? target);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Evaluated with gradient recording suppressed: a value-only call must never add entries to a tape the
    /// caller happens to have open.
    /// </remarks>
    public T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        using var noGrad = new NoGradScope<T>();
        var loss = ComputeTapeLoss(Tensor<T>.FromVector(predicted), Tensor<T>.FromVector(actual));
        return loss[0];
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The composition is a small head or metric over a handful of examples per task; there is nothing a device
    /// kernel would win back, so it runs on the tape like every other path.
    /// </remarks>
    public (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        var (loss, gradient) = LossFunctionExtensions.ComputeLossAndGradient(this, predicted, actual);
        return (loss, gradient);
    }

    /// <summary>
    /// Logits of a linear head, <c>h W^T + b</c>, recorded on the tape when one is live.
    /// </summary>
    /// <param name="embeddings">The embeddings, <c>[rows, width]</c>.</param>
    /// <param name="weights">The head's weights, <c>[classes, width]</c>.</param>
    /// <param name="bias">The head's bias, <c>[classes]</c>, or null for none.</param>
    internal static Tensor<T> LinearHead(Tensor<T> embeddings, Tensor<T> weights, Tensor<T>? bias)
    {
        var engine = AiDotNetEngine.Current;
        var logits = engine.TensorMatMul(embeddings, engine.TensorTranspose(weights));
        return bias is null ? logits : engine.TensorAdd(logits, engine.Reshape(bias, new[] { 1, bias.Length }));
    }

    /// <summary>
    /// The embeddings as <c>[rows, width]</c>. The vector forms of the loss carry no shape, so the row count comes
    /// from the number of class indices.
    /// </summary>
    private static Tensor<T> AsRows(Tensor<T> embeddings, int rows)
    {
        if (embeddings.Shape.Length == 2) return embeddings;
        if (rows <= 0 || embeddings.Length % rows != 0)
        {
            throw new ArgumentException(
                $"{embeddings.Length} embedding values do not divide evenly into {rows} examples.",
                nameof(embeddings));
        }

        return AiDotNetEngine.Current.Reshape(embeddings, new[] { rows, embeddings.Length / rows });
    }
}
