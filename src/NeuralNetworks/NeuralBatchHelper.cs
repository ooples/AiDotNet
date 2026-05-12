using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Shared dispatcher that swaps full-batch <see cref="IFullModel{T,TInput,TOutput}.Predict"/>
/// and <see cref="IFullModel{T,TInput,TOutput}.Train"/> calls for memory-bounded
/// chunked equivalents when the model is a <see cref="NeuralNetworkBase{T}"/>
/// and the input tensor's leading axis exceeds a chunk threshold. Non-neural-
/// network models and non-tensor inputs fall through to the existing call,
/// so the helper is a no-op for closed-form / tree / linear / clustering
/// pipelines.
///
/// <para>
/// Centralises the dispatch logic that surfaces at every full-batch site
/// audited in #1296 (optimizer evaluator, cross-validators, AutoML trial
/// loop, NAS supernet warmup). All sites share the same shape: a generic
/// <c>IFullModel&lt;T, TInput, TOutput&gt;</c> with a <c>TInput</c> that
/// may or may not be a <see cref="Tensor{T}"/>; pushing the type checks
/// into one helper means there's a single source of truth for the
/// "should this be chunked?" decision and a single point to evolve when
/// the threshold or chunk strategy changes.
/// </para>
/// </summary>
public static class NeuralBatchHelper
{
    /// <summary>
    /// Default chunk size used when callers don't specify one. Chosen so a
    /// Transformer at <c>d=128 / L=4 / heads=4 / ctx=64</c> peaks at
    /// roughly 16 MB of attention scores per chunk — comfortably within
    /// CI heap budgets while large enough to amortise per-call dispatch
    /// overhead on small / medium tensors.
    /// </summary>
    public const int DefaultBatchSize = 256;

    /// <summary>
    /// Routes <c>model.Predict(X)</c> through
    /// <see cref="NeuralNetworkBase{T}.PredictInBatches"/> when the model
    /// is a tensor-based neural network and <paramref name="X"/> is a
    /// <see cref="Tensor{T}"/> with a leading axis larger than
    /// <paramref name="batchSize"/>. Output is element-equivalent (modulo
    /// matmul reduction order) to a single-shot <c>Predict</c>.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <typeparam name="TInput">The model's input type (typically <see cref="Tensor{T}"/> for neural nets).</typeparam>
    /// <typeparam name="TOutput">The model's output type.</typeparam>
    /// <param name="model">The model to invoke <c>Predict</c> on. Must not be <see langword="null"/>.</param>
    /// <param name="X">Input batch to score.</param>
    /// <param name="batchSize">Maximum samples per forward pass. Defaults to <see cref="DefaultBatchSize"/>.</param>
    /// <returns>
    /// Predictions matching what an unchunked <c>Predict(X)</c> would have
    /// produced. Falls through to <c>model.Predict(X)</c> unchanged when
    /// the model isn't an NN, <paramref name="X"/> isn't a <see cref="Tensor{T}"/>,
    /// or the leading-axis size doesn't exceed <paramref name="batchSize"/>.
    /// </returns>
    public static TOutput PredictMaybeBatched<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        int batchSize = DefaultBatchSize)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (batchSize < 1) batchSize = 1;

        if (model is NeuralNetworkBase<T> nn
            && X is Tensor<T> xTensor
            && xTensor.Rank >= 1
            && xTensor.Shape[0] > batchSize)
        {
            var chunked = nn.PredictInBatches(xTensor, batchSize);
            if (chunked is TOutput typed)
            {
                return typed;
            }
            // PredictInBatches always returns Tensor<T>; if TOutput is some
            // other type the model's own Predict will produce, the cast
            // fails and we transparently fall through to the unchunked
            // path so semantics stay identical for every non-Tensor<T>
            // output type.
        }
        return model.Predict(X);
    }

    /// <summary>
    /// Routes <c>model.Train(X, Y)</c> through a sequence of mini-batched
    /// <see cref="NeuralNetworkBase{T}.Train"/> calls when the model is a
    /// tensor-based neural network and both tensors share a leading axis
    /// larger than <paramref name="batchSize"/>. For non-neural-network
    /// models, or when the input/target aren't tensors of matching shape,
    /// falls through to <c>model.Train(X, Y)</c> unchanged.
    /// </summary>
    /// <typeparam name="T">The numeric type used by the model.</typeparam>
    /// <typeparam name="TInput">The model's input type (typically <see cref="Tensor{T}"/> for neural nets).</typeparam>
    /// <typeparam name="TOutput">The model's target type.</typeparam>
    /// <param name="model">The model to invoke <c>Train</c> on. Must not be <see langword="null"/>.</param>
    /// <param name="X">Input batch.</param>
    /// <param name="Y">Target batch (must have the same leading-axis size as <paramref name="X"/>).</param>
    /// <param name="batchSize">Maximum samples per <c>Train</c> call.</param>
    /// <remarks>
    /// <para>
    /// <b>Semantic note:</b> mini-batching does change the gradient
    /// trajectory — instead of one SGD step on the full batch, the model
    /// receives <c>ceil(N / batchSize)</c> SGD steps on smaller batches.
    /// For neural-network models this is the standard mini-batch SGD
    /// regime every modern framework defaults to, and is strictly closer
    /// to the optimiser regime that <c>Adam.Optimize</c>'s epoch loop
    /// uses elsewhere. For closed-form / tree / linear models this would
    /// be incorrect (most fit in one pass), so they fall through to the
    /// unchunked path.
    /// </para>
    /// </remarks>
    public static void TrainMaybeBatched<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        TOutput Y,
        int batchSize = DefaultBatchSize)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (batchSize < 1) batchSize = 1;

        if (model is NeuralNetworkBase<T> nn
            && X is Tensor<T> xTensor
            && Y is Tensor<T> yTensor
            && xTensor.Rank >= 1
            && yTensor.Rank >= 1
            && xTensor.Shape[0] == yTensor.Shape[0]
            && xTensor.Shape[0] > batchSize)
        {
            int n = xTensor.Shape[0];
            int nChunks = (n + batchSize - 1) / batchSize;
            for (int chunkIdx = 0; chunkIdx < nChunks; chunkIdx++)
            {
                int start = chunkIdx * batchSize;
                int end = Math.Min(start + batchSize, n);
                var xChunk = xTensor.Slice(axis: 0, start: start, end: end).Contiguous();
                var yChunk = yTensor.Slice(axis: 0, start: start, end: end).Contiguous();
                nn.Train(xChunk, yChunk);
            }
            return;
        }
        model.Train(X, Y);
    }
}
