using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralRadianceFields.Data;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralRadianceFields.Helpers;

/// <summary>
/// Shared helpers for radiance-field <c>TrainOnImageBatch</c> implementations (#1834). Extracts
/// the code common across <c>NeRF</c>, <c>InstantNGP</c>, and <c>GaussianSplatting</c> so each
/// model's <c>TrainOnImageBatch</c> is a thin adapter over its own <c>RenderRays</c> +
/// per-attribute update path, and every model gets the same engine-backed loss computation
/// (which auto-records on the gradient tape — critical for the follow-up backprop-through-
/// render lands in #1834's paper-faithful continuation).
/// </summary>
public static class ImageTrainingHelpers
{
    /// <summary>
    /// Pulls one batch from the loader; returns <see langword="null"/> if the loader is
    /// exhausted. The single-batch shape is the paper-standard per-iteration unit for
    /// image-space training.
    /// </summary>
    public static PixelBatch<T>? PullOneBatch<T>(
        IDataLoader<ImageView<T>, PixelBatch<T>> loader,
        int raysPerBatch)
    {
        if (loader is null) throw new ArgumentNullException(nameof(loader));
        if (raysPerBatch <= 0) throw new ArgumentOutOfRangeException(nameof(raysPerBatch));

        // Dispose the enumerator — IEnumerator<T> is IDisposable and iterator-block
        // enumerators use finally to release resources; dropping it undisposed leaks.
        using var enumerator = loader.IterateBatches(raysPerBatch).GetEnumerator();
        return enumerator.MoveNext() ? enumerator.Current.Output : null;
    }

    /// <summary>
    /// Engine-based MSE between rendered and target pixel color tensors. Uses
    /// <see cref="IEngine.TensorSubtract"/> + <see cref="IEngine.TensorMultiply"/> +
    /// <see cref="IEngine.TensorSum"/> so the operation is vectorized and recorded on the
    /// gradient tape (unlocking the paper-faithful backprop follow-up).
    /// </summary>
    public static T PhotometricMSE<T>(IEngine engine, Tensor<T> rendered, Tensor<T> target)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (rendered is null) throw new ArgumentNullException(nameof(rendered));
        if (target is null) throw new ArgumentNullException(nameof(target));

        var numOps = MathHelper.GetNumericOperations<T>();
        var diff = engine.TensorSubtract(rendered, target);
        var squared = engine.TensorMultiply(diff, diff);
        var summed = engine.TensorSum(squared);
        return numOps.Divide(summed, numOps.FromDouble(rendered.Length));
    }

    /// <summary>
    /// Concatenates ray origins [N, 3] and directions [N, 3] into the [N, 6] shape GS's
    /// existing <c>Train</c> accepts (RAY mode). Uses <see cref="IEngine.TensorConcatenate"/>
    /// so the operation is tape-recorded.
    /// </summary>
    public static Tensor<T> RayOriginsDirectionsToNBy6<T>(IEngine engine, PixelBatch<T> pixels)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (pixels is null) throw new ArgumentNullException(nameof(pixels));
        return engine.TensorConcatenate(new[] { pixels.RayOrigins, pixels.RayDirections }, axis: 1);
    }
}
