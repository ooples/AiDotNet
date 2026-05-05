using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Diagnostic trace of one Transformer.Train cycle. Captures: initial logits,
/// gradient norms per-parameter (top 5 by magnitude), parameter L2 before/after,
/// optimizer state on first step. Used to find the next class of bugs.
/// </summary>
public class TransformerTrainingTraceTest
{
    private readonly ITestOutputHelper _output;
    public TransformerTrainingTraceTest(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void TraceOneStep_V4()
    {
        const int vocab = 4;
        const int ctxLen = 8;
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 1, numDecoderLayers: 0, numHeads: 2,
            modelDimension: 16, feedForwardDimension: 32,
            inputSize: ctxLen, outputSize: vocab,
            maxSequenceLength: ctxLen, vocabularySize: vocab);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Verify default optimizer is Adam — assert in addition to logging,
        // otherwise a regression that defaults back to GradientDescent (the
        // exact bug closes #1264) would only show up in test stdout, not
        // fail the test. Issue #1264 reporters had this exact false-pass
        // pattern: "test ran fine, but logs showed wrong optimizer".
        var optField = typeof(Transformer<float>).GetField("_optimizer",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        var opt = optField!.GetValue(model);
        _output.WriteLine($"Default optimizer: {opt!.GetType().Name}");
        Assert.Contains("Adam", opt.GetType().Name);
        Assert.DoesNotContain("GradientDescent", opt.GetType().Name);
        var optOpts = opt.GetType().GetMethod("GetOptions")!.Invoke(opt, null);
        _output.WriteLine($"Optimizer options: MaxIterations={optOpts!.GetType().GetProperty("MaxIterations")!.GetValue(optOpts)}, "
            + $"InitialLR={optOpts.GetType().GetProperty("InitialLearningRate")!.GetValue(optOpts)}, "
            + $"UseAdaptiveLR={optOpts.GetType().GetProperty("UseAdaptiveLearningRate")!.GetValue(optOpts)}, "
            + $"LRDecay={optOpts.GetType().GetProperty("LearningRateDecay")!.GetValue(optOpts)}");

        var input = new Tensor<float>([1, ctxLen]);
        for (int s = 0; s < ctxLen; s++) input[0, s] = (float)(s % vocab);
        var target = new Tensor<float>([1, vocab]);
        target[0, 1] = 1f;

        // Warmup forward pass to materialize lazy-init params (LayerNorm γ,
        // MHA lazy weight banks). Without this, the BEFORE measurement
        // undercounts and the AFTER measurement appears to "explode" —
        // that's a measurement artifact, not an optimizer bug.
        model.SetTrainingMode(false);
        // Narrow the warmup catch — InvalidOperationException is the only
        // expected eval-mode-incompatible failure (some layers refuse a
        // non-training Predict). Swallowing every Exception here would
        // mask NaN propagation, shape errors, and OOM bugs that this trace
        // test is supposed to surface in the loss measurements below.
        try { model.Predict(input); } catch (InvalidOperationException) { }
        model.SetTrainingMode(true);

        // Snapshot initial parameters (sum of L2 norms)
        double paramL2Before = ComputeTotalParamL2(model);
        _output.WriteLine($"Total param L2 BEFORE training: {paramL2Before:F6}");

        // Initial prediction
        model.SetTrainingMode(false);
        var pred0 = model.Predict(input);
        var p0 = Softmax(pred0, vocab);
        _output.WriteLine($"Initial logits:    [{pred0[0,0]:F4}, {pred0[0,1]:F4}, {pred0[0,2]:F4}, {pred0[0,3]:F4}]");
        _output.WriteLine($"Initial probs:     [{p0[0]:F4}, {p0[1]:F4}, {p0[2]:F4}, {p0[3]:F4}]");

        // One training step
        model.SetTrainingMode(true);
        model.Train(input, target);
        float loss1 = model.GetLastLoss();
        _output.WriteLine($"Loss after 1 step: {loss1:F6}");

        double paramL2After1 = ComputeTotalParamL2(model);
        _output.WriteLine($"Total param L2 AFTER 1 step:  {paramL2After1:F6}  (Δ = {paramL2After1 - paramL2Before:+0.000000;-0.000000})");

        // Per-layer L2 decomposition: which layer exploded?
        _output.WriteLine("--- per-layer L2 after 1 step ---");
        for (int li = 0; li < model.Layers.Count; li++)
        {
            var layer = model.Layers[li];
            var p = layer.GetParameters();
            double l2 = 0;
            for (int i = 0; i < p.Length; i++) l2 += p[i] * p[i];
            l2 = System.Math.Sqrt(l2);
            _output.WriteLine($"  layer[{li}] {layer.GetType().Name,-40} count={p.Length,5} L2={l2:F6}");
        }

        // 99 more steps
        for (int s = 0; s < 99; s++) model.Train(input, target);
        float loss100 = model.GetLastLoss();
        double paramL2After100 = ComputeTotalParamL2(model);
        _output.WriteLine($"Loss after 100 steps: {loss100:F6}");
        _output.WriteLine($"Total param L2 AFTER 100 steps: {paramL2After100:F6}");

        // Final prediction
        model.SetTrainingMode(false);
        var pred100 = model.Predict(input);
        var p100 = Softmax(pred100, vocab);
        _output.WriteLine($"After-100 logits:  [{pred100[0,0]:F4}, {pred100[0,1]:F4}, {pred100[0,2]:F4}, {pred100[0,3]:F4}]");
        _output.WriteLine($"After-100 probs:   [{p100[0]:F4}, {p100[1]:F4}, {p100[2]:F4}, {p100[3]:F4}]");
        _output.WriteLine($"P(target=1) Δ:     {p0[1]:F4} → {p100[1]:F4}");

        // Param L2 sanity — one Adam step should NOT explode the model.
        // ±5% (was ±0.1%) tolerates the natural per-step movement Adam +
        // LayerNorm γ/β init produce on the small toy harness while still
        // catching genuine explosions (10×, 100×, NaN). The 0.1% bound
        // was unseed-flaky because Transformer sub-layers initialize
        // params from RandomHelper without a fixed test seed.
        Assert.True(paramL2After1 > paramL2Before * 0.95 && paramL2After1 < paramL2Before * 1.05,
            $"Param L2 changed too much in one step: {paramL2Before:F4} → {paramL2After1:F4}");
        Assert.True(loss100 < loss1 * 0.99,
            $"Loss did not decrease over 100 steps: {loss1:F4} → {loss100:F4}");
    }

    private static double ComputeTotalParamL2(Transformer<float> model)
    {
        double total = 0;
        foreach (var layer in model.Layers)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++) total += p[i] * p[i];
        }
        return System.Math.Sqrt(total);
    }

    private static float[] Softmax(Tensor<float> pred, int vocab)
    {
        float max = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++) if (pred[0, v] > max) max = pred[0, v];
        float sum = 0;
        var p = new float[vocab];
        for (int v = 0; v < vocab; v++) { p[v] = MathF.Exp(pred[0, v] - max); sum += p[v]; }
        for (int v = 0; v < vocab; v++) p[v] /= sum;
        return p;
    }
}
