using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Adam's tape step: the pre-clip gradient norm it reports to training monitors, and the moment state its fp32 fast
/// path reads after the eager moments are replaced.
/// </summary>
public class AdamTapeStepStateTests
{
    private static AdamOptimizer<float, Vector<float>, Vector<float>> Adam(bool valueClip)
    {
        var options = new AdamOptimizerOptions<float, Vector<float>, Vector<float>>
        {
            InitialLearningRate = 0.01,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = valueClip,
            GradientClippingMethod = GradientClippingMethod.ByValue,
            MaxGradientValue = 1.0,
        };
        return new AdamOptimizer<float, Vector<float>, Vector<float>>(model: null, options: options);
    }

    private static TapeStepContext<float> Context(Tensor<float> parameter, float[] gradient) =>
        new(
            parameters: new[] { parameter },
            gradients: new Dictionary<Tensor<float>, Tensor<float>>(
                AiDotNet.Helpers.TensorReferenceComparer<Tensor<float>>.Instance)
            {
                [parameter] = new Tensor<float>(new[] { gradient.Length }, new Vector<float>((float[])gradient.Clone())),
            },
            loss: 0f);

    private static float[] Values(Tensor<float> tensor)
    {
        var values = new float[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = tensor[i];
        return values;
    }

    [Fact(Timeout = 60000)]
    public async Task ValueClip_WithOptInNorm_ReportsThePreClipNorm()
    {
        await Task.CompletedTask;
        // Gradients [3, 4] clamp to [1, 1]: the monitored norm is the pre-clip 5, not the clamped sqrt(2).
        var adam = Adam(valueClip: true);
        adam.ComputeGradientNormEveryStep = true;
        var parameter = new Tensor<float>(new[] { 2 }, new Vector<float>(new[] { 0.5f, -0.5f }));

        adam.Step(Context(parameter, new[] { 3f, 4f }));

        Assert.NotNull(adam.LastStepGradientNorm);
        Assert.Equal(5.0, adam.LastStepGradientNorm!.Value, 5);
    }

    [Fact(Timeout = 60000)]
    public async Task ValueClip_WithoutOptInNorm_ReportsNoNormRatherThanThePostClampOne()
    {
        await Task.CompletedTask;
        // Adam's anomaly guard reduces the norm after the value clamp; that is not the pre-clip norm, so none is
        // reported.
        var adam = Adam(valueClip: true);
        var parameter = new Tensor<float>(new[] { 2 }, new Vector<float>(new[] { 0.5f, -0.5f }));

        adam.Step(Context(parameter, new[] { 3f, 4f }));

        Assert.Null(adam.LastStepGradientNorm);
    }

    [Fact(Timeout = 60000)]
    public async Task WithoutValueClip_ReportsTheGuardsNorm()
    {
        await Task.CompletedTask;
        var adam = Adam(valueClip: false);
        var parameter = new Tensor<float>(new[] { 2 }, new Vector<float>(new[] { 0.5f, -0.5f }));

        adam.Step(Context(parameter, new[] { 3f, 4f }));

        Assert.NotNull(adam.LastStepGradientNorm);
        Assert.Equal(5.0, adam.LastStepGradientNorm!.Value, 5);
    }

    [Fact(Timeout = 60000)]
    public async Task RestoredMoments_ReachTheFp32FastPath()
    {
        await Task.CompletedTask;
        // A steps once, is checkpointed, steps again (which caches its moment arrays for the fast path), then restores
        // the checkpoint. Its next step must use the restored moments, exactly as a fresh optimizer restored from the
        // same checkpoint does, and not the arrays it cached before the restore.
        var start = new[] { 0.25f, -0.75f, 1.5f };
        var g1 = new[] { 0.3f, -0.2f, 0.1f };
        var g2 = new[] { -0.9f, 0.4f, 0.7f };
        var g3 = new[] { 0.05f, 0.6f, -0.3f };

        var a = Adam(valueClip: false);
        var pa = new Tensor<float>(new[] { 3 }, new Vector<float>((float[])start.Clone()));
        a.Step(Context(pa, g1));
        float[] afterFirst = Values(pa);
        byte[] checkpoint = a.Serialize();
        a.Step(Context(pa, g2));
        a.Step(Context(pa, g2));
        a.Deserialize(checkpoint);
        for (int i = 0; i < afterFirst.Length; i++) pa[i] = afterFirst[i];   // in place: the same storage
        a.Step(Context(pa, g3));

        var b = Adam(valueClip: false);
        b.Deserialize(checkpoint);
        var pb = new Tensor<float>(new[] { 3 }, new Vector<float>((float[])afterFirst.Clone()));
        b.Step(Context(pb, g3));

        float[] ra = Values(pa), rb = Values(pb);
        for (int i = 0; i < ra.Length; i++)
            Assert.Equal(rb[i], ra[i]);
    }
}
