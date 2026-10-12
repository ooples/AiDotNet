using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Issue #2312: the training path through <see cref="MultiHeadAttentionLayer{T}"/> must use the
/// fused, tape-aware <c>Engine.FlashAttention</c> kernel rather than the score-materializing
/// <c>Engine.ScaledDotProductAttention</c>, and <c>UseCausalMask</c> must actually mask.
/// </summary>
/// <remarks>Not parallel: the allocation test measures <c>GC.GetTotalAllocatedBytes</c>, which counts every
/// thread, and the engine's own work may run on worker threads, so a per-thread counter would undercount.</remarks>
[Collection("NonParallelIntegration")]
public sealed class MultiHeadAttentionFusedTrainingTests
{
    private const int HeadCount = 4;
    private const int HeadDimension = 8;
    private const int Embedding = HeadCount * HeadDimension;

    /// <summary>
    /// <c>UseCausalMask</c> promises a position never attends to a later one. The observable
    /// consequence is that changing the LAST token cannot change any earlier row's output.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task CausalMaskingKeepsTheFutureOutOfEveryEarlierRow()
    {
        await Task.Yield();

        const int sequence = 6;
        var layer = NewLayer(causal: true);
        layer.SetTrainingMode(false);

        var input = CreatePattern([1, sequence, Embedding], 0.021);
        var baseline = Forward(layer, input);

        // Perturb every feature of the final token only.
        for (int f = 0; f < Embedding; f++)
            input[((sequence - 1) * Embedding) + f] += 0.75;
        var perturbed = Forward(layer, input);

        double worst = 0.0;
        int worstRow = -1;
        for (int row = 0; row < sequence - 1; row++)
        {
            for (int f = 0; f < Embedding; f++)
            {
                int i = (row * Embedding) + f;
                double delta = Math.Abs(baseline[i] - perturbed[i]);
                if (delta > worst)
                {
                    worst = delta;
                    worstRow = row;
                }
            }
        }

        Assert.True(worst <= 1e-12,
            $"A causal layer let the last token reach row {worstRow}: largest change {worst:R} "
                + $"across rows 0..{sequence - 2}. UseCausalMask is not being applied.");
    }

    /// <summary>
    /// The same perturbation MUST move the last row, otherwise the test above would also pass
    /// on a layer that ignores its input entirely.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task TheLastRowStillSeesTheTokenThatWasPerturbed()
    {
        await Task.Yield();

        const int sequence = 6;
        var layer = NewLayer(causal: true);
        layer.SetTrainingMode(false);

        var input = CreatePattern([1, sequence, Embedding], 0.021);
        var baseline = Forward(layer, input);
        for (int f = 0; f < Embedding; f++)
            input[((sequence - 1) * Embedding) + f] += 0.75;
        var perturbed = Forward(layer, input);

        double moved = 0.0;
        for (int f = 0; f < Embedding; f++)
        {
            int i = ((sequence - 1) * Embedding) + f;
            moved = Math.Max(moved, Math.Abs(baseline[i] - perturbed[i]));
        }

        Assert.True(moved > 1e-6,
            $"The perturbed token did not move its own row ({moved:R}); the control arm is dead.");
    }

    /// <summary>
    /// Gradients through the training path must match central finite differences, which is what
    /// makes swapping the kernel safe rather than merely faster.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task TrainingGradientsMatchFiniteDifferences()
    {
        await Task.Yield();

        const int batch = 2;
        const int sequence = 5;
        var layer = NewLayer(causal: false);
        layer.SetTrainingMode(true);

        var input = CreatePattern([batch, sequence, Embedding], 0.017);
        var projection = CreatePattern([batch, sequence, Embedding], 0.029);

        Tensor<double>? gradient;
        using (var tape = new GradientTape<double>())
        {
            var output = layer.Forward(input);
            var projected = AiDotNetEngine.Current.TensorMultiply(output, projection);
            var objective = AiDotNetEngine.Current.ReduceSum(projected, [0, 1, 2], keepDims: false);
            var gradients = tape.ComputeGradients(objective, [input]);
            Assert.True(gradients.TryGetValue(input, out gradient),
                "The training path must keep the tape connected to its input.");
        }

        Assert.NotNull(gradient);
        const double step = 1e-6;
        int[] samples = [0, input.Length / 3, input.Length / 2, input.Length - 1];
        foreach (int index in samples)
        {
            double original = input[index];
            input[index] = original + step;
            double plus = Project(layer, input, projection);
            input[index] = original - step;
            double minus = Project(layer, input, projection);
            input[index] = original;

            double numerical = (plus - minus) / (2.0 * step);
            double analytical = gradient[index];
            double tolerance = Math.Max(1e-7, Math.Abs(numerical) * 1e-5);
            Assert.True(Math.Abs(numerical - analytical) <= tolerance,
                $"Input gradient mismatch at {index}: numerical={numerical:R}, "
                    + $"analytical={analytical:R}, tolerance={tolerance:R}.");
        }
    }

    /// <summary>
    /// The training forward must take the fused kernel, which is what removes the score matrix
    /// from attention's memory. A forward whose auxiliary losses read the scores is the
    /// documented exception and keeps the score-materializing kernel.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task TrainingTakesTheFusedKernelUnlessSomethingReadsTheScores()
    {
        await Task.Yield();

        var layer = NewLayer(causal: false);
        layer.SetTrainingMode(true);
        var input = CreatePattern([2, 12, Embedding], 0.013);

        using (var tape = new GradientTape<double>())
        {
            layer.Forward(input);
        }
        Assert.Equal(AttentionKernelPath.FusedFlashAttention, layer.LastAttentionKernelPath);

        layer.UseAuxiliaryLoss = true;
        using (var tape = new GradientTape<double>())
        {
            layer.Forward(input);
        }
        Assert.Equal(AttentionKernelPath.ScoreMaterializing, layer.LastAttentionKernelPath);
    }

#if NET6_0_OR_GREATER // GC.GetTotalAllocatedBytes does not exist on .NET Framework (the compat build's net471)
    /// <summary>
    /// The fused kernel's point is the memory it does not spend. Both arms are measured in this
    /// process, and the gap must be at least the score matrix the fused arm never allocates.
    /// </summary>
    /// <remarks>
    /// The score-materializing arm is selected through <c>UseAuxiliaryLoss</c>, which also
    /// allocates the loss terms themselves, so the gap is if anything understated by this
    /// comparison — it cannot be inflated into a false pass by the fused arm.
    /// </remarks>
    [Fact(Timeout = 120000)]
    public async Task TheFusedKernelSavesTheScoreMatrixAllocation()
    {
        await Task.Yield();

        const int batch = 2;
        const int sequence = 512;
        var layer = NewLayer(causal: false);
        layer.SetTrainingMode(true);
        var input = CreatePattern([batch, sequence, Embedding], 0.003);

        // Warm up so first-call lazy allocation is counted in neither arm.
        using (var warmTape = new GradientTape<double>())
        {
            layer.Forward(CreatePattern([batch, sequence, Embedding], 0.003));
        }

        long before = GC.GetTotalAllocatedBytes(precise: true);
        using (var tape = new GradientTape<double>())
        {
            layer.Forward(input);
        }
        long fused = GC.GetTotalAllocatedBytes(precise: true) - before;
        Assert.Equal(AttentionKernelPath.FusedFlashAttention, layer.LastAttentionKernelPath);

        layer.UseAuxiliaryLoss = true;
        using (var warmTape = new GradientTape<double>())
        {
            layer.Forward(input);
        }
        before = GC.GetTotalAllocatedBytes(precise: true);
        using (var tape = new GradientTape<double>())
        {
            layer.Forward(input);
        }
        long materialized = GC.GetTotalAllocatedBytes(precise: true) - before;
        Assert.Equal(AttentionKernelPath.ScoreMaterializing, layer.LastAttentionKernelPath);

        long scoreBytes = (long)batch * HeadCount * sequence * sequence * sizeof(double);
        Assert.True(materialized - fused >= scoreBytes * 4 / 5,
            $"The fused forward allocated {fused:N0} bytes and the score-materializing one "
                + $"{materialized:N0}; the gap {materialized - fused:N0} is below the "
                + $"{scoreBytes:N0}-byte score matrix the fused kernel is supposed to avoid.");
    }
#endif

    private static MultiHeadAttentionLayer<double> NewLayer(bool causal)
    {
        var layer = new MultiHeadAttentionLayer<double>(
            HeadCount,
            HeadDimension,
            activationFunction: (IActivationFunction<double>)new IdentityActivation<double>());
        layer.UseCausalMask = causal;
        return layer;
    }

    private static Tensor<double> Forward(
        MultiHeadAttentionLayer<double> layer, Tensor<double> input)
    {
        using var noGrad = new NoGradScope<double>();
        return layer.Forward(input);
    }

    private static double Project(
        MultiHeadAttentionLayer<double> layer, Tensor<double> input, Tensor<double> projection)
    {
        using var noGrad = new NoGradScope<double>();
        var output = layer.Forward(input);
        double sum = 0.0;
        for (int i = 0; i < output.Length; i++)
            sum += output[i] * projection[i];
        return sum;
    }

    private static Tensor<double> CreatePattern(int[] shape, double scale)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = (Math.Sin((i + 1) * scale) * 0.5) + (Math.Cos((i + 3) * scale * 0.5) * 0.25);
        return tensor;
    }
}
