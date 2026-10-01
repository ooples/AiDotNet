using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Finance.Forecasting.Foundation;

public partial class CSDI<T> : ITrainingObjectiveProvider<T>
{
    /// <summary>
    /// The learner is denoising diffusion, not supervised regression of the forecast onto the
    /// target: <see cref="Train"/> minimizes the noise-prediction error of Tashiro et al. 2021
    /// Algorithm 1 (Ho et al. 2020's L_simple, conditioned on the observed values), and the
    /// forecast is the reverse sampler run on top of it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each <see cref="Train"/> call draws one diffusion step and one noise vector, so its loss is a
    /// single draw of the objective at a random noise level. Comparing two such draws compares two
    /// different objectives; the memorization probe measured 1.512 at step 1 and 1.549 at step 20 in
    /// a full shard and passed when rerun alone. Declaring the objective lets a loss-trajectory
    /// probe measure the expectation training descends, over a fixed quadrature.
    /// </para>
    /// </remarks>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind =>
        TrainingObjectiveKind.DiffusionDenoising;

    /// <summary>The supplied forecast target IS the x_0 the denoiser learns to recover.</summary>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget)
        => proposedTarget;

    /// <summary>
    /// Evaluates L_simple over the fixed (timestep, noise) quadrature of
    /// <see cref="Base.TimeSeriesFoundationModelBase{T}.BuildDeterministicDenoisingBatch"/>, through
    /// the loss <see cref="Train"/> uses and the same denoiser graph <see cref="Train"/> runs.
    /// </summary>
    /// <remarks>
    /// The denoiser scores one (x_t, t) pair per call, so each quadrature row is posed as the slot
    /// tuple <see cref="Train"/> builds, with its timestep and noise taken from the fixed batch
    /// instead of a fresh draw. The rows are averaged. Nothing here updates a parameter.
    /// </remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("The training objective is only defined in native mode.");

        int targetLength = target.Length;
        if (targetLength <= 0)
            throw new ArgumentException("The training target must not be empty.", nameof(target));

        var (_, noise, timesteps) = BuildDeterministicDenoisingBatch(
            target, targetLength, _numDiffusionSteps, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod);

        var conditioned = ApplyInstanceNormalization(input);
        if (conditioned.Rank == 1)
            conditioned = Engine.Reshape(conditioned, new[] { 1, conditioned.Length });

        T total = NumOps.Zero;
        for (int row = 0; row < timesteps.Length; row++)
        {
            int t = timesteps[row];
            var epsilon = new Tensor<T>(target._shape);
            noise.Data.Span.Slice(row * targetLength, targetLength).CopyTo(epsilon.Data.Span);

            var sqrtAlphaBar = new Tensor<T>(new[] { 1 });
            sqrtAlphaBar[0] = _sqrtAlphasCumprod[t];
            var sqrtOneMinus = new Tensor<T>(new[] { 1 });
            sqrtOneMinus[0] = _sqrtOneMinusAlphasCumprod[t];
            var sinT = new Tensor<T>(new[] { 1 });
            sinT[0] = NumOps.FromDouble(Math.Sin(2.0 * Math.PI * t / Math.Max(1, _numDiffusionSteps - 1)));

            var predicted = DenoiserForwardFromSlots(
                new[] { target, epsilon, sqrtAlphaBar, sqrtOneMinus, sinT, conditioned });
            total = NumOps.Add(total, LossFunction.ComputeLoss(predicted, epsilon));
        }

        return NumOps.Divide(total, NumOps.FromDouble(timesteps.Length));
    }
}
