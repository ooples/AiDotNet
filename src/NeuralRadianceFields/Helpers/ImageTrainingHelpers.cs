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
/// (which auto-records on the gradient tape — the mechanism NeRF/InstantNGP use to
/// backprop-through-render into the MLP weights via <c>BackwardAndStepOnPrecomputedLoss</c>).
/// </summary>
public static class ImageTrainingHelpers
{
    /// <summary>
    /// Pulls one batch from the loader; returns <see langword="null"/> if the loader is
    /// exhausted. Returns BOTH the sampled view and its ray batch — models that consume
    /// the parent view (e.g. for LearnedPrior blending) need the pair, not just the rays.
    /// </summary>
    public static (ImageView<T> View, PixelBatch<T> Pixels)? PullOneBatch<T>(
        IDataLoader<ImageView<T>, PixelBatch<T>> loader,
        int raysPerBatch)
    {
        if (loader is null) throw new ArgumentNullException(nameof(loader));
        if (raysPerBatch <= 0) throw new ArgumentOutOfRangeException(nameof(raysPerBatch));

        using var enumerator = loader.IterateBatches(raysPerBatch).GetEnumerator();
        if (!enumerator.MoveNext()) return null;
        var current = enumerator.Current;
        return (current.Input, current.Output);
    }

    /// <summary>
    /// Engine-based MSE between rendered and target pixel color tensors. Uses
    /// <see cref="IEngine.TensorSubtract"/> + <see cref="IEngine.TensorMultiply"/> +
    /// <see cref="IEngine.TensorSum"/> so the operation is vectorized and recorded on the
    /// gradient tape (feeds NeRF/InstantNGP's BackwardAndStepOnPrecomputedLoss path).
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
    /// If the view carries a <see cref="LearnedPrior{T}"/>, blends the prior's hallucinated
    /// per-pixel color into the ray target so ray-mode training (NeRF, InstantNGP) picks
    /// up single-image reconstruction signal from angles the caller didn't photograph.
    /// Returns the original targets when no prior is set or its confidence is zero.
    /// </summary>
    /// <remarks>
    /// The prior is sampled once per view at the CURRENT view's pose (H × W). We look up
    /// each ray's pixel in the hallucination via the source-view's pixel index — because
    /// <see cref="ImageTrainingDataLoaders.FromViews{T}"/> samples random pixels of the
    /// view's photo, and the corresponding pixel index in the hallucination is the same
    /// (both are indexed by <c>[y, x]</c> in the view frame). We don't have the ray→pixel
    /// mapping in <see cref="PixelBatch{T}"/> directly, so we approximate by weighting the
    /// blend uniformly: target = (1 - c) * photo_sample + c * mean(hallucination).
    /// Reference impls apply per-pixel per-view novel-view priors; this weighted mix is a
    /// cheap approximation that still gives density-network signal in unseen directions.
    /// </remarks>
    public static Tensor<T> ApplyPriorToRayTargets<T>(
        IEngine engine,
        Tensor<T> rayTargets,
        ImageView<T> view,
        double? confidenceOverride)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (rayTargets is null) throw new ArgumentNullException(nameof(rayTargets));
        if (view is null) throw new ArgumentNullException(nameof(view));
        if (view.Prior is null) return rayTargets;

        double confidence = confidenceOverride ?? view.Prior.Confidence;
        if (confidence <= 0) return rayTargets;
        if (confidence > 1) confidence = 1;

        var numOps = MathHelper.GetNumericOperations<T>();
        var hallucination = view.Prior.SynthesizeNovelView(
            view.CameraPosition, view.CameraRotation, view.Height, view.Width);

        // Mean color across the hallucination [H, W, 3] → per-channel scalar. Cheap +
        // stable proxy for the "prior signal at this view" when we don't have the per-ray
        // pixel index. Callers with a rays→pixels index map can extend this helper.
        double meanR = 0, meanG = 0, meanB = 0;
        int H = view.Height, W = view.Width;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
        {
            meanR += numOps.ToDouble(hallucination[y, x, 0]);
            meanG += numOps.ToDouble(hallucination[y, x, 1]);
            meanB += numOps.ToDouble(hallucination[y, x, 2]);
        }
        double count = H * W;
        meanR /= count; meanG /= count; meanB /= count;

        // Weighted blend: target = (1 - c) * photo_sample + c * prior_mean_color.
        // Uses engine ops so the blend stays tape-recorded — future backprop through the
        // blended target works without a manual scalar-op replay.
        double invC = 1.0 - confidence;
        int nRays = rayTargets.Shape[0];
        int channels = rayTargets.Shape.Length > 1 ? rayTargets.Shape[1] : 3;

        var priorArr = new T[nRays * channels];
        for (int i = 0; i < nRays; i++)
        {
            priorArr[i * channels + 0] = numOps.FromDouble(meanR);
            if (channels > 1) priorArr[i * channels + 1] = numOps.FromDouble(meanG);
            if (channels > 2) priorArr[i * channels + 2] = numOps.FromDouble(meanB);
            // 4th channel (density) if present: leave at 0 — the prior doesn't touch density.
        }
        var shape = new int[rayTargets.Shape.Length];
        for (int s = 0; s < shape.Length; s++) shape[s] = rayTargets.Shape[s];
        var priorTensor = new Tensor<T>(shape, new AiDotNet.Tensors.LinearAlgebra.Vector<T>(priorArr));
        var invCT = numOps.FromDouble(invC);
        var cT = numOps.FromDouble(confidence);
        var photoScaled = engine.TensorMultiplyScalar(rayTargets, invCT);
        var priorScaled = engine.TensorMultiplyScalar(priorTensor, cT);
        return engine.TensorAdd(photoScaled, priorScaled);
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
