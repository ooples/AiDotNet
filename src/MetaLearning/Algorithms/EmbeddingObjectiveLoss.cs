using System;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// A meta-learner's whole per-task objective, expressed over the embedding network's output: the closure builds the
/// objective - inner loop, regularisers and all - from the embeddings with engine ops, so the embedding network
/// differentiates exactly what the learner minimises.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <see cref="EmbeddingClassificationLoss{T}"/> composes a head into a classification loss; this is its general
/// form for objectives that are not one loss over one set of logits, such as LEO's (Rusu et al. 2019, eq. 6).
/// Passed as the <c>lossOverride</c> of <c>MetaLearnerBase.ComputeGradients</c>, the embedding network's tape
/// carries the chain rule through the closure. The target handed in only fixes the row count.
/// </para>
/// <para>
/// The closure may open nested gradient tapes of its own, so no path here suppresses gradient recording: an objective
/// with an inner loop needs its inner tapes to record even when only its value is wanted.
/// </para>
/// </remarks>
internal sealed class EmbeddingObjectiveLoss<T> : ILossFunction<T>
{
    private readonly Func<Tensor<T>, Tensor<T>> _objective;

    /// <summary>Creates the composed objective.</summary>
    /// <param name="objective">Builds a one-element objective from <c>[rows, width]</c> embeddings.</param>
    internal EmbeddingObjectiveLoss(Func<Tensor<T>, Tensor<T>> objective)
    {
        _objective = objective ?? throw new ArgumentNullException(nameof(objective));
    }

    /// <inheritdoc/>
    /// <remarks>The target's length is the number of embedding rows; its values are not read.</remarks>
    public Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));
        return _objective(AsRows(predicted, target.Length));
    }

    /// <inheritdoc/>
    public T CalculateLoss(Vector<T> predicted, Vector<T> actual)
        => ComputeTapeLoss(Tensor<T>.FromVector(predicted), Tensor<T>.FromVector(actual))[0];

    /// <inheritdoc/>
    public (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        var (loss, gradient) = LossFunctionExtensions.ComputeLossAndGradient(this, predicted, actual);
        return (loss, gradient);
    }

    /// <summary>The embeddings as <c>[rows, width]</c>.</summary>
    private static Tensor<T> AsRows(Tensor<T> embeddings, int rows)
    {
        if (embeddings.Shape.Length == 2) return embeddings;
        if (rows <= 0 || embeddings.Length % rows != 0)
        {
            throw new ArgumentException(
                $"{embeddings.Length} embedding values do not divide evenly into {rows} examples.", nameof(embeddings));
        }

        return AiDotNetEngine.Current.Reshape(embeddings, new[] { rows, embeddings.Length / rows });
    }
}
