using System;
using System.IO;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Exercises the opt-in LSUV (Layer-Sequential Unit-Variance) data-dependent init
/// (Mishkin &amp; Matas 2015) wired into <see cref="NeuralNetworkBase{T}"/>. LSUV runs once on the
/// first training step: it forwards a calibration batch and rescales each trainable layer's weights
/// so the layer's output variance is ~1, which removes the poorly-conditioned-random-draw tail that
/// leaves a small net nearly input-insensitive. These tests prove (a) it actually normalizes a
/// deliberately badly-scaled net back toward unit output variance, and (b) it is strictly opt-in —
/// with the flag off (the default) it never touches the weights, so it can't change the training
/// trajectory of the many models whose standard He/Xavier init is already well-conditioned.
/// </summary>
public sealed class LsuvInitTests : IDisposable
{
    public void Dispose() => NeuralNetworkBase<float>.LsuvInitEnabled = false;

    /// <summary>Two bare Dense layers — enough for the per-layer calibration walk to compound.</summary>
    private sealed class TwoLayerNet : NeuralNetworkBase<float>
    {
        public TwoLayerNet()
            : base(lossFunction: new MeanSquaredErrorLoss<float>(), maxGradNorm: 1.0)
        {
            Layers.Add(new DenseLayer<float>(outputSize: 12));
            Layers.Add(new DenseLayer<float>(outputSize: 4));
        }

        protected override void InitializeLayers() { /* layers added in ctor */ }

        // Route UpdateParameters through the base layer-walking SetParameters so the test's
        // weight-inflation actually takes effect on the layers.
        public override void UpdateParameters(Vector<float> parameters) => SetParameters(parameters);

        public override ModelMetadata<float> GetModelMetadata() => new() { Name = "TwoLayerNet" };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }

        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
            => new TwoLayerNet();
    }

    private static Tensor<float> CalibrationBatch()
    {
        // 16 rows × 8 features so the output spread (variance) is well-defined.
        const int rows = 16, features = 8;
        var input = new Tensor<float>([rows, features]);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < features; c++)
                input[r, c] = (float)Math.Sin((r + 1) * 0.37 + c * 0.91); // deterministic, varied
        return input;
    }

    private static double OutputVariance(Tensor<float> t)
    {
        int n = t.Length;
        double mean = 0.0;
        for (int i = 0; i < n; i++) mean += t[i];
        mean /= n;
        double s2 = 0.0;
        for (int i = 0; i < n; i++) { double d = t[i] - mean; s2 += d * d; }
        return s2 / n;
    }

    private static void ScaleAllWeights(TwoLayerNet net, float factor)
    {
        var p = net.GetParameters();
        var scaled = new float[p.Length];
        for (int i = 0; i < p.Length; i++) scaled[i] = p[i] * factor;
        net.UpdateParameters(new Vector<float>(scaled));
    }

    [Fact]
    public void Lsuv_NormalizesBadlyScaledNet_BackTowardUnitOutputVariance()
    {
        var net = new TwoLayerNet();
        var input = CalibrationBatch();

        // Materialize the lazy Dense weights, then deliberately blow the weights up ×50 so the
        // output variance explodes far past unit scale — the pathological condition LSUV exists to fix.
        _ = net.Predict(input);
        ScaleAllWeights(net, 50f);

        double preVar = OutputVariance(net.Predict(input));
        Assert.True(preVar > 100.0,
            $"Test setup failed: inflating weights ×50 should explode output variance, but preVar={preVar}.");

        // Run LSUV once (the same calibration the first training step performs).
        NeuralNetworkBase<float>.LsuvInitEnabled = true;
        net.ApplyLsuvInitForTest(input);

        double postVar = OutputVariance(net.Predict(input));

        // LSUV calibrates each layer's output variance to ~1, so the final output variance lands in
        // an O(1) band — a massive reduction from the inflated value.
        Assert.True(postVar < preVar * 0.05,
            $"LSUV should have collapsed the inflated output variance (pre={preVar}, post={postVar}).");
        Assert.InRange(postVar, 0.05, 20.0);
    }

    [Fact]
    public void Lsuv_Disabled_IsNoOp_LeavesBadScaleUntouched()
    {
        var net = new TwoLayerNet();
        var input = CalibrationBatch();

        _ = net.Predict(input);
        ScaleAllWeights(net, 50f);
        double preVar = OutputVariance(net.Predict(input));

        // Flag OFF (the default). The calibration must NOT run — proving the feature is strictly
        // opt-in and can't silently alter any model's init.
        NeuralNetworkBase<float>.LsuvInitEnabled = false;
        net.ApplyLsuvInitForTest(input);

        double postVar = OutputVariance(net.Predict(input));
        Assert.Equal(preVar, postVar, 3); // unchanged to 3 decimal places
    }
}
