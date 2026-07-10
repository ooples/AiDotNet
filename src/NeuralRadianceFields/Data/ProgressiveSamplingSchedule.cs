using System;

namespace AiDotNet.NeuralRadianceFields.Data;

/// <summary>
/// Progressive coarse-to-fine ray-sampling schedule (#1834 excellence goal #4). Reference NeRF
/// impls hard-code a fixed sample count (64 coarse + 128 fine); nerfstudio requires the schedule
/// as a YAML entry that only takes effect if the caller wired the appropriate scheduler class
/// into the training pipeline. Here it's first-class: instantiate a
/// <see cref="ProgressiveSamplingSchedule"/> and hand it to the image-space training call — the
/// facade selects the coarse count while density is still forming, then ramps to the fine count.
/// </summary>
public class ProgressiveSamplingSchedule
{
    /// <summary>Sample count in the coarse phase (early iterations). Paper default: 64.</summary>
    public int CoarseSamples { get; init; } = 64;

    /// <summary>Sample count in the fine phase. Paper default: 128 (hierarchical) or 192 (dense).</summary>
    public int FineSamples { get; init; } = 128;

    /// <summary>Iteration at which the schedule finishes ramping from coarse to fine.
    /// Paper default: 5000.</summary>
    public int RampEndIteration { get; init; } = 5000;

    /// <summary>
    /// Returns the sample count for iteration <paramref name="iteration"/>. Linearly ramps
    /// from CoarseSamples to FineSamples across [0, RampEndIteration], holds at FineSamples
    /// beyond. Callers who want a different curve (exponential, step) can subclass and override.
    /// </summary>
    public virtual int SamplesForIteration(int iteration)
    {
        ValidateSchedule();
        if (iteration <= 0) return CoarseSamples;
        if (iteration >= RampEndIteration) return FineSamples;
        double t = (double)iteration / RampEndIteration;
        return (int)Math.Round(CoarseSamples + t * (FineSamples - CoarseSamples));
    }

    /// <summary>
    /// Guard against invalid configuration: zero/negative sample counts break the render
    /// (empty allocations, no-op MSE) and zero/negative ramp windows would divide by zero
    /// or ramp instantaneously. Called at every <see cref="SamplesForIteration"/> — cheap
    /// per-call cost, catches misuse at the schedule boundary instead of inside the model.
    /// </summary>
    protected void ValidateSchedule()
    {
        if (CoarseSamples <= 0)
        {
            throw new InvalidOperationException(
                $"{nameof(ProgressiveSamplingSchedule)}.{nameof(CoarseSamples)} must be positive; got {CoarseSamples}.");
        }
        if (FineSamples <= 0)
        {
            throw new InvalidOperationException(
                $"{nameof(ProgressiveSamplingSchedule)}.{nameof(FineSamples)} must be positive; got {FineSamples}.");
        }
        if (RampEndIteration <= 0)
        {
            throw new InvalidOperationException(
                $"{nameof(ProgressiveSamplingSchedule)}.{nameof(RampEndIteration)} must be positive; got {RampEndIteration}.");
        }
    }

    /// <summary>
    /// Convenience — the paper's hierarchical schedule: 64 coarse + 128 fine, ramping over 5k iters.
    /// </summary>
    public static ProgressiveSamplingSchedule Paper() => new();

    /// <summary>
    /// Fast schedule — smaller sample counts for prototyping / low-budget training.
    /// </summary>
    public static ProgressiveSamplingSchedule Fast() => new()
    {
        CoarseSamples = 16,
        FineSamples = 32,
        RampEndIteration = 1000,
    };
}
