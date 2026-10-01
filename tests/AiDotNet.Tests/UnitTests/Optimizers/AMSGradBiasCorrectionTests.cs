using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Pins AMSGrad's two bias-correction variants to their published formulas. Paper (the default) is Reddi, Kale and
/// Kumar (2018), Algorithm 2: no bias correction. PyTorch is <c>Adam(amsgrad=True)</c>: the first moment over
/// (1 - beta1^t) and the running max of the RAW second moment over (1 - beta2^t). The expected values are computed
/// here, independently, step by step.
/// </summary>
public sealed class AMSGradBiasCorrectionTests
{
    private const double Lr = 0.1, B1 = 0.9, B2 = 0.999, Eps = 1e-8;

    private static AMSGradOptimizer<double, Vector<double>, Vector<double>> Create(AMSGradBiasCorrection mode) =>
        new(null, new AMSGradOptimizerOptions<double, Vector<double>, Vector<double>>
        {
            InitialLearningRate = Lr, Beta1 = B1, Beta2 = B2, Epsilon = Eps, BiasCorrection = mode,
            UseAdaptiveLearningRate = false,
        });

    private static double[] Reference(AMSGradBiasCorrection mode, double[] p0, double[][] grads)
    {
        var p = (double[])p0.Clone();
        var m = new double[p.Length]; var v = new double[p.Length]; var vMax = new double[p.Length];
        for (int t = 1; t <= grads.Length; t++)
        {
            double bc1 = mode == AMSGradBiasCorrection.PyTorch ? 1 - Math.Pow(B1, t) : 1.0;
            double bc2 = mode == AMSGradBiasCorrection.PyTorch ? 1 - Math.Pow(B2, t) : 1.0;
            for (int i = 0; i < p.Length; i++)
            {
                double g = grads[t - 1][i];
                m[i] = B1 * m[i] + (1 - B1) * g;
                v[i] = B2 * v[i] + (1 - B2) * g * g;
                vMax[i] = Math.Max(vMax[i], v[i]);
                p[i] -= Lr * (m[i] / bc1) / (Math.Sqrt(vMax[i] / bc2) + Eps);
            }
        }
        return p;
    }

    [Theory]
    [InlineData(AMSGradBiasCorrection.Paper)]
    [InlineData(AMSGradBiasCorrection.PyTorch)]
    public void TwoSteps_MatchTheVariantsFormula(AMSGradBiasCorrection mode)
    {
        var p0 = new[] { 0.5, -1.0, 2.0 };
        // The second gradient is smaller, so vMax keeps step 1's value for the first coordinate: the max matters.
        var grads = new[] { new[] { 0.3, -0.2, 0.1 }, new[] { 0.01, -0.4, 0.05 } };
        var optimizer = Create(mode);

        var p = new Vector<double>(p0);
        foreach (var g in grads) p = optimizer.UpdateParameters(p, new Vector<double>(g));

        var expected = Reference(mode, p0, grads);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], p[i], 12);
    }

    [Fact]
    public void TheVariantsActuallyDiffer()
    {
        // Positive control for the theory above: if the option were ignored, both rows could pass against one formula.
        var p0 = new[] { 0.5, -1.0, 2.0 };
        var grads = new[] { new[] { 0.3, -0.2, 0.1 } };
        var paper = Reference(AMSGradBiasCorrection.Paper, p0, grads);
        var pytorch = Reference(AMSGradBiasCorrection.PyTorch, p0, grads);
        Assert.True(Math.Abs(paper[0] - pytorch[0]) > 1e-3, "the two variants' first steps must differ");
    }

    [Fact]
    public void ThePaperDefault_DoesNotMapToTheFusedKernel_ThePyTorchVariantDoes()
    {
        // The fused kernel implements the PyTorch variant only, so the paper default must decline it (and train
        // eagerly) rather than run a different formula than the one configured.
        Assert.Equal(AMSGradBiasCorrection.Paper, new AMSGradOptimizerOptions<double, Vector<double>, Vector<double>>().BiasCorrection);
        var paper = (AiDotNet.Optimizers.Fused.IFusedOptimizerSpec)Create(AMSGradBiasCorrection.Paper);
        var pytorch = (AiDotNet.Optimizers.Fused.IFusedOptimizerSpec)Create(AMSGradBiasCorrection.PyTorch);
        Assert.False(paper.TryGetFusedOptimizerConfig(out _));
        Assert.True(pytorch.TryGetFusedOptimizerConfig(out _));
    }

    /// <summary>
    /// The GPU path runs the PyTorch (bias-corrected) kernel for both variants, fed a rescaled learning rate and
    /// epsilon for the paper; both must still land on their own formula.
    /// </summary>
    [SkippableTheory]
    [Trait("Category", "GPU")]
    [InlineData(AMSGradBiasCorrection.Paper)]
    [InlineData(AMSGradBiasCorrection.PyTorch)]
    public void GpuUpdates_MatchTheVariantsFormula(AMSGradBiasCorrection mode)
    {
        using var gpu = new AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine();
        Skip.IfNot(gpu.IsAvailable && gpu.Backend is not null, "no GPU backend resolved");
        if (gpu.Backend is not { } backend) return;

        var p0 = new[] { 0.5, -1.0, 2.0 };
        var grads = new[] { new[] { 0.3, -0.2, 0.1 }, new[] { 0.01, -0.4, 0.05 } };
        var optimizer = new AMSGradOptimizer<float, Vector<float>, Vector<float>>(null, new AMSGradOptimizerOptions<float, Vector<float>, Vector<float>>
        {
            InitialLearningRate = Lr, Beta1 = B1, Beta2 = B2, Epsilon = Eps, BiasCorrection = mode,
            UseAdaptiveLearningRate = false,
        });

        using var parameters = backend.AllocateBuffer(p0.Select(x => (float)x).ToArray());
        foreach (var g in grads)
        {
            using var gradient = backend.AllocateBuffer(g.Select(x => (float)x).ToArray());
            optimizer.UpdateParametersGpu(parameters, gradient, p0.Length, backend);
        }

        var actual = backend.DownloadBuffer(parameters);
        optimizer.DisposeGpuState();
        var expected = Reference(mode, p0, grads);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5 * Math.Max(1, Math.Abs(expected[i])),
                $"{mode}: p[{i}] expected {expected[i]:R}, got {actual[i]:R}");
    }
}