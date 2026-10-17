using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

/// <summary>
/// One differentially private SGD update (Abadi et al. 2016, Algorithm 1): every example's gradient is clipped to a
/// global L2 norm BEFORE aggregation, the clipped gradients are averaged, Gaussian noise of standard deviation
/// <c>clipNorm * noiseMultiplier / batch</c> is added, and the optimizer takes one step on the result.
/// </summary>
/// <remarks>
/// <para>
/// MedGAN and DP-CTGAN each carried this step with its own eager per-example loop. It lives here once: the
/// GPU-resident <see cref="DpSgdFusedStep{T}"/> primitive when it is available, otherwise the eager per-example loop
/// below, then the optimizer update. The clip-before-aggregate order is what the privacy proof's L2-sensitivity bound
/// rests on, so it is fixed by this type's structure rather than re-implemented per model.
/// </para>
/// <para>
/// The per-example clipped sums are owned accumulators summed in place. The copies this replaces rebuilt each sum
/// with an engine add inside the example's tape scope; under the training arena (#1804) the tape's dispose recycles
/// that storage, so the next example's temporaries could overwrite the running sum.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class DpSgdTrainingStep<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Runs one DP-SGD update and returns the mean per-example loss.</summary>
    /// <param name="parameters">The tensors to update; each receives clipped, averaged, noised gradient.</param>
    /// <param name="exampleCount">The number of examples in the batch.</param>
    /// <param name="exampleSlots">The tensors one example's objective reads (e.g. its real and fake rows).</param>
    /// <param name="forward">The forward of one example from its slots.</param>
    /// <param name="computeLoss">One example's scalar loss from the forward output and its slots.</param>
    /// <param name="clipNorm">C, the per-example global L2 clip.</param>
    /// <param name="noiseMultiplier">sigma; the noise standard deviation is <c>C * sigma / batch</c> (0 = none).</param>
    /// <param name="random">The model's generator (used by the fused primitive's noise).</param>
    /// <param name="optimizer">The update rule applied to the noised average gradient.</param>
    public static T Step(
        IReadOnlyList<Tensor<T>> parameters,
        int exampleCount,
        Func<int, IReadOnlyList<Tensor<T>>> exampleSlots,
        Func<IReadOnlyList<Tensor<T>>, Tensor<T>> forward,
        Func<Tensor<T>, IReadOnlyList<Tensor<T>>, Tensor<T>> computeLoss,
        double clipNorm,
        double noiseMultiplier,
        Random random,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer)
    {
        Guard.NotNull(parameters);
        Guard.NotNull(exampleSlots);
        Guard.NotNull(forward);
        Guard.NotNull(computeLoss);
        Guard.NotNull(random);
        Guard.NotNull(optimizer);
        Guard.Positive(exampleCount);

        double lossSum = 0.0;
        Tensor<T> ExampleLoss(Tensor<T> output, IReadOnlyList<Tensor<T>> slots)
        {
            var loss = computeLoss(output, slots);
            if (loss.Length > 0) lossSum += NumOps.ToDouble(loss[0]);
            return loss;
        }

        // The primitive's gradients may live in its buffers, so it stays alive until the optimizer has stepped.
        using var primitive = new DpSgdFusedStep<T>();
        if (!primitive.TryStep(
                parameters, exampleSlots, forward, ExampleLoss, exampleCount, clipNorm, noiseMultiplier, random,
                out var noisedAverage))
        {
            lossSum = 0.0;
            noisedAverage = EagerNoisedAverage(
                parameters, exampleCount, exampleSlots, forward, ExampleLoss, clipNorm, noiseMultiplier);
        }

        T meanLoss = NumOps.FromDouble(lossSum / exampleCount);
        var updated = new List<Tensor<T>>(parameters.Count);
        foreach (var parameter in parameters)
            if (parameter is not null && noisedAverage.ContainsKey(parameter))
                updated.Add(parameter);

        // A line-searching optimizer re-evaluates the batch objective: the mean per-example loss (unclipped, noiseless).
        var engine = AiDotNetEngine.Current;
        Tensor<T> MeanObjective()
        {
            Tensor<T>? total = null;
            for (int i = 0; i < exampleCount; i++)
            {
                var slots = exampleSlots(i);
                var loss = computeLoss(forward(slots), slots);
                total = total is null ? loss : engine.TensorAdd(total, loss);
            }
            return engine.TensorMultiplyScalar(total ?? new Tensor<T>(new[] { 1 }), NumOps.FromDouble(1.0 / exampleCount));
        }

        var placeholder = new Tensor<T>(new[] { 1 });
        optimizer.Step(new TapeStepContext<T>(
            updated, noisedAverage, meanLoss, placeholder, placeholder,
            (_, _) => MeanObjective(), (value, _) => value, parameterBuffer: null));
        if (optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> scheduled)
            scheduled.OnBatchEnd();
        return meanLoss;
    }

    private static Dictionary<Tensor<T>, Tensor<T>> EagerNoisedAverage(
        IReadOnlyList<Tensor<T>> parameters,
        int exampleCount,
        Func<int, IReadOnlyList<Tensor<T>>> exampleSlots,
        Func<IReadOnlyList<Tensor<T>>, Tensor<T>> forward,
        Func<Tensor<T>, IReadOnlyList<Tensor<T>>, Tensor<T>> computeLoss,
        double clipNorm,
        double noiseMultiplier)
    {
        var engine = AiDotNetEngine.Current;
        var clippedSums = new Dictionary<Tensor<T>, Tensor<T>>(parameters.Count, TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var parameter in parameters)
        {
            if (parameter is null || clippedSums.ContainsKey(parameter)) continue;
            clippedSums[parameter] = new Tensor<T>(parameter._shape); // owned, zero-initialized
        }

        for (int example = 0; example < exampleCount; example++)
        {
            using var tape = new GradientTape<T>();
            var slots = exampleSlots(example);
            var loss = computeLoss(forward(slots), slots);
            var gradients = tape.ComputeGradients(loss, parameters);

            // The GLOBAL L2 norm across every parameter's gradient: the sensitivity the privacy proof bounds.
            double normSquared = 0.0;
            foreach (var gradient in gradients.Values)
            {
                var sum = engine.ReduceSum(engine.TensorMultiply(gradient, gradient), null, keepDims: false);
                normSquared += sum.Length > 0 ? NumOps.ToDouble(sum[0]) : 0.0;
            }
            T clipFactor = NumOps.FromDouble(Math.Min(1.0, clipNorm / Math.Sqrt(normSquared + 1e-12)));

            foreach (var pair in clippedSums)
                if (gradients.TryGetValue(pair.Key, out var gradient))
                    engine.TensorAddInPlace(pair.Value, engine.TensorMultiplyScalar(gradient, clipFactor));
        }

        double inverseCount = 1.0 / exampleCount;
        double noiseStd = clipNorm * noiseMultiplier * inverseCount;
        var noisedAverage = new Dictionary<Tensor<T>, Tensor<T>>(clippedSums.Count, TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var pair in clippedSums)
        {
            var average = pair.Value;
            engine.TensorMultiplyScalarInPlace(average, NumOps.FromDouble(inverseCount));
            if (noiseStd > 0)
            {
                var noise = new Tensor<T>(average._shape);
                engine.TensorRandomNormalInto(noise, NumOps.Zero, NumOps.FromDouble(noiseStd));
                engine.TensorAddInPlace(average, noise);
            }
            noisedAverage[pair.Key] = average;
        }

        return noisedAverage;
    }
}