using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Robotics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Integration tests for the memory-bounded streaming training path
/// (<see cref="StreamingTrainingMode"/> → ComputeGradientsStreaming +
/// 8-bit Adam optimizer-in-backward). These exercise the streaming path
/// directly (ForceOn) on a small model that trains in milliseconds, so the
/// subsystem is regression-guarded independently of the paper-scale models that
/// actually trigger the autotuner.
/// </summary>
public class StreamingTrainingTests
{
    private static Helix<float> CreateTinyHelix()
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 4);
        // Reduced-scale dual-system config mirroring the validated ModelFamily
        // HelixTests (same dimensional shape as the ~6.7B paper model, ~4-8×
        // smaller). The lazy vision/decoder LayerNorms size from VisionDim/
        // DecoderDim, so the FixedSample input width MUST equal VisionDim — see
        // FixedSample below. An inconsistent width (e.g. VisionDim=32 with a
        // 224-derived arch warmup) bakes a mismatched LayerNorm gamma.
        var options = new HelixOptions
        {
            VisionDim = 256,
            DecoderDim = 512,
            NumVisionLayers = 4,
            NumDecoderLayers = 4,
            NumHeads = 8,
            System2LatentDim = 128,
            System1HiddenDim = 96,
            System1NumLayers = 2,
            System1NumHeads = 4,
            ActionDimension = 35,
            DropoutRate = 0.0,
        };
        return new Helix<float>(arch, options);
    }

    private static double Mse(Helix<float> model, Tensor<float> input, Tensor<float> target)
    {
        var pred = model.Predict(input);
        // Fail fast on length mismatch — the original Math.Min(pred, target) version
        // silently masked output/target shape regressions, letting these tests pass
        // even when the model started returning the wrong shape. The training
        // contract under test here REQUIRES pred and target to be identically
        // shaped; any drift should surface as a hard assertion failure.
        Assert.Equal(target.Length, pred.Length);
        double sum = 0;
        for (int i = 0; i < pred.Length; i++)
        {
            double d = Convert.ToDouble(pred[i]) - Convert.ToDouble(target[i]);
            sum += d * d;
        }
        return pred.Length > 0 ? sum / pred.Length : 0;
    }

    private static (Tensor<float> input, Tensor<float> target) FixedSample()
    {
        var rng = new Random(123);
        // Post-patch-embedding token features [batch, num_tokens, VisionDim] — Helix's
        // documented input contract (see ModelFamily HelixTests). Width MUST equal
        // VisionDim (256) so the lazy vision LayerNorm gamma matches.
        var input = new Tensor<float>(new[] { 1, 4, 256 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);
        // Output is the action head: [1, 4, 35].
        var target = new Tensor<float>(new[] { 1, 4, 35 });
        for (int i = 0; i < target.Length; i++) target[i] = (float)(rng.NextDouble() - 0.5);
        return (input, target);
    }

    /// <summary>
    /// The streaming path (ForceOn) must actually train: loss decreases over a
    /// handful of steps on a fixed sample, and every parameter stays finite.
    /// </summary>
    [Fact]
    public void Streaming_ReducesLoss_AndStaysFinite()
    {
        using var model = CreateTinyHelix();
        model.StreamingTraining = StreamingTrainingMode.ForceOn;
        model.StreamingTrainingLearningRate = 1e-3;
        var (input, target) = FixedSample();

        double firstLoss = Mse(model, input, target);
        model.SetTrainingMode(true);
        for (int step = 0; step < 12; step++) model.Train(input, target);
        model.SetTrainingMode(false);
        double lastLoss = Mse(model, input, target);

        Assert.True(!double.IsNaN(firstLoss) && !double.IsInfinity(firstLoss), "First streaming loss is not finite.");
        Assert.True(!double.IsNaN(lastLoss) && !double.IsInfinity(lastLoss), "Final streaming loss is not finite.");
        Assert.True(lastLoss < firstLoss,
            $"Streaming training did not reduce loss: first={firstLoss:E4}, last={lastLoss:E4}.");

        // Every parameter must remain finite after streaming training.
        var p = model.GetParameters();
        for (int i = 0; i < p.Length; i++)
        {
            double v = Convert.ToDouble(p[i]);
            Assert.True(!double.IsNaN(v) && !double.IsInfinity(v), $"Parameter[{i}] became non-finite under streaming training.");
        }
    }

    /// <summary>
    /// The streaming path must actually update parameters (optimizer-in-backward
    /// reaches every layer), not silently no-op.
    /// </summary>
    [Fact]
    public void Streaming_ChangesParameters()
    {
        using var model = CreateTinyHelix();
        model.StreamingTraining = StreamingTrainingMode.ForceOn;
        var (input, target) = FixedSample();

        model.SetTrainingMode(true);
        model.Predict(input); // materialize lazy params
        var before = model.GetParameters();
        var beforeCopy = new float[before.Length];
        for (int i = 0; i < before.Length; i++) beforeCopy[i] = before[i];

        model.Train(input, target);
        var after = model.GetParameters();
        model.SetTrainingMode(false);

        double maxDelta = 0;
        int n = Math.Min(beforeCopy.Length, after.Length);
        for (int i = 0; i < n; i++)
            maxDelta = Math.Max(maxDelta, Math.Abs(Convert.ToDouble(after[i]) - beforeCopy[i]));

        Assert.True(maxDelta > 0, "Streaming training changed no parameters.");
    }

    /// <summary>
    /// The autotuner default (Auto) must NOT engage streaming for a small model —
    /// it trains on the classic path. We assert it trains correctly all the same
    /// (loss finite), confirming Auto is a safe default for models that fit.
    /// </summary>
    [Fact]
    public void Auto_OnSmallModel_TrainsViaClassicPath()
    {
        using var model = CreateTinyHelix();
        // Default is Auto; a tiny model's footprint is far below the autotuner
        // threshold, so the classic in-memory path runs.
        Assert.Equal(StreamingTrainingMode.Auto, model.StreamingTraining);
        var (input, target) = FixedSample();

        model.SetTrainingMode(true);
        model.Train(input, target);
        model.SetTrainingMode(false);
        double loss = Mse(model, input, target);

        Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss), "Classic-path loss is not finite.");
    }
}
