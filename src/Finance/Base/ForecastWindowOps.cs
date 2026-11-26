using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Finance.Base;

/// <summary>
/// The window bookkeeping of autoregressive forecasting on <c>[batch, time, features]</c> tensors: sliding the
/// input window forward over the predictions just made, and stitching the predicted chunks into one forecast.
/// Both run as slices and one concatenation along the time axis.
/// </summary>
/// <remarks>
/// <para>
/// The results are detached: they are data for the next forward pass, not part of a gradient graph. An index that
/// the element-by-element copy would have read or written out of range is reported with the exception that copy
/// raised (<see cref="ArgumentOutOfRangeException"/>; <see cref="ArgumentException"/> for a prediction that is not
/// rank 3), before any tensor is built.
/// </para>
/// <para><b>For Beginners:</b> a forecaster that predicts a few steps at a time reaches further into the future by
/// feeding its predictions back in: drop the oldest steps of the window, append the newest predictions, predict
/// again. At the end the chunks are joined into one sequence.</para>
/// </remarks>
internal static class ForecastWindowOps
{
    /// <summary>
    /// The next input window <c>[batch, length, features]</c>: steps <c>steps .. length - 1</c> of
    /// <paramref name="input"/> followed by the first <paramref name="steps"/> steps of
    /// <paramref name="predictions"/>.
    /// </summary>
    /// <param name="input">The current window <c>[batch, at least length, features]</c>.</param>
    /// <param name="predictions">The predictions <c>[at least batch, at least steps, at least features]</c>.</param>
    /// <param name="length">The window length.</param>
    /// <param name="steps">How many steps the window moves forward, from 0 to <paramref name="length"/>.</param>
    internal static Tensor<T> ShiftWindow<T>(Tensor<T> input, Tensor<T>? predictions, int length, int steps)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        // Predictions are read only when the window moves: with steps == 0 they may be null, as before.
        if (predictions is null && steps != 0) throw new ArgumentNullException(nameof(predictions));

        int batch = input.Shape[0], features = input.Shape[2];
        if (batch <= 0 || features <= 0)
            return new Tensor<T>([batch, length, features]);

        int keep = length - steps;
        if (steps < 0)
            throw new ArgumentOutOfRangeException(nameof(steps), "The window cannot move backwards.");
        if (keep < 0)
            throw new ArgumentOutOfRangeException(nameof(steps), "The window cannot move further than its length.");
        if (keep > 0 && input.Shape[1] < length)
            throw new ArgumentOutOfRangeException(nameof(input), "The input is shorter than the window.");
        if (steps > 0)
        {
            if (predictions!.Rank != 3)
                throw new ArgumentException("Predictions must be rank 3 [batch, time, features].", nameof(predictions));
            if (predictions.Shape[0] < batch || predictions.Shape[1] < steps || predictions.Shape[2] < features)
                throw new ArgumentOutOfRangeException(nameof(predictions), "The predictions do not cover the shifted steps.");
        }

        if (length == 0)
            return new Tensor<T>([batch, length, features]);

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        if (steps == 0)
            return engine.TensorSlice(input, [0, 0, 0], [batch, length, features]).Contiguous();
        var newest = engine.TensorSlice(predictions!, [0, 0, 0], [batch, steps, features]);
        if (keep == 0)
            return newest.Contiguous();
        return engine.TensorConcatenate([engine.TensorSlice(input, [0, steps, 0], [batch, keep, features]), newest], 1);
    }

    /// <summary>
    /// One forecast <c>[batch, totalSteps, features]</c> from predicted chunks: the leading steps of each chunk in
    /// order until <paramref name="totalSteps"/> are filled, zero where the chunks run out. The batch size is that
    /// of the first chunk; with no chunks the result is <c>[1, totalSteps, features]</c> of zeros.
    /// </summary>
    /// <param name="predictions">Chunks <c>[at least batch, steps, at least features]</c>.</param>
    /// <param name="totalSteps">The forecast length.</param>
    /// <param name="features">The feature count of the forecast.</param>
    internal static Tensor<T> ConcatenateLeadingSteps<T>(IReadOnlyList<Tensor<T>> predictions, int totalSteps, int features)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (predictions.Count == 0)
            return new Tensor<T>([1, totalSteps, features]);

        int batch = predictions[0].Shape[0];
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var parts = new List<Tensor<T>>();
        int filled = 0;
        foreach (var chunk in predictions)
        {
            int take = Math.Min(chunk.Shape[1], totalSteps - filled);
            if (take <= 0 || batch <= 0 || features <= 0)
                continue;
            if (chunk.Rank != 3)
                throw new ArgumentException("Predictions must be rank 3 [batch, time, features].", nameof(predictions));
            if (chunk.Shape[0] < batch || chunk.Shape[2] < features)
                throw new ArgumentOutOfRangeException(nameof(predictions), "A prediction is smaller than the forecast.");
            parts.Add(engine.TensorSlice(chunk, [0, 0, 0], [batch, take, features]));
            filled += take;
        }

        if (parts.Count == 0)
            return new Tensor<T>([batch, totalSteps, features]);   // only when no chunk contributes a step
        if (filled < totalSteps)
            parts.Add(new Tensor<T>([batch, totalSteps - filled, features]));
        return parts.Count == 1 ? parts[0].Contiguous() : engine.TensorConcatenate(parts.ToArray(), 1);
    }
}
