using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Repro + regression test for #1415 — MultiHeadAttentionLayer.Forward
/// produces all-NaN output for specific inputs at large vocab (V=50,257).
///
/// <para>Consumer-side diagnostic (HarmonicEngine) localized the bug to
/// MHA[10] in a 2-layer Transformer trained at V=50,257. ~25% of input
/// contexts produce all-NaN attention output even though:</para>
/// <list type="bullet">
///   <item>Trained weights are bounded (maxAbs ≤ 0.25, no NaN/Inf).</item>
///   <item>MHA input is finite and bounded ([-2.18, 1.75]).</item>
///   <item>Same MHA at layer 3 (earlier in stack, smaller-magnitude input)
///         produces finite output.</item>
/// </list>
///
/// <para>This test isolates the issue to bare MHA Forward (no Transformer
/// stack, no training). Replicates the layer-10 input distribution
/// (post-LayerNorm finite tensor with magnitudes up to ~2.2) and asserts
/// finite output across many random seeds.</para>
/// </summary>
public class Issue1415_LargeVocabForwardNaNTests
{
    private readonly ITestOutputHelper _output;
    public Issue1415_LargeVocabForwardNaNTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void MultiHeadAttention_Forward_ProducesFiniteOutput_OnFiniteInput_AtPostLayerNormScale()
    {
        AiDotNetEngine.ResetToCpu();
        const int batchSize = 1, seqLen = 64, dModel = 128, heads = 2;
        const int trials = 200;
        int nanTrials = 0;

        var rng = RandomHelper.CreateSeededRandom(0);

        // Construct MHA with deterministic init (RandomSeed=0 reproduces the
        // weight magnitudes seen in the consumer trace: maxAbs ~0.044).
        for (int trial = 0; trial < trials; trial++)
        {
            var mha = new MultiHeadAttentionLayer<float>(
                headCount: heads,
                headDimension: dModel / heads,
                activationFunction: null);
            ((LayerBase<float>)mha).RandomSeed = trial;

            // Build a post-LayerNorm-like input: zero mean, unit variance per
            // feature, magnitudes typical of LayerNorm output observed in the
            // consumer trace (range ~[-2.2, 1.75]).
            var input = new Tensor<float>([batchSize, seqLen, dModel]);
            for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < seqLen; s++)
            {
                // Per-token sample with mean 0 std 1 (Box-Muller-like uniform→normal)
                float sum = 0;
                for (int d = 0; d < dModel; d++)
                {
                    // Truncated normal via average of 12 uniforms (CLT).
                    float v = 0;
                    for (int k = 0; k < 12; k++) v += (float)rng.NextDouble();
                    v -= 6f;
                    input[b, s, d] = v;
                    sum += v;
                }
                // Subtract mean and normalize like LayerNorm.
                float mean = sum / dModel;
                float ss = 0;
                for (int d = 0; d < dModel; d++) ss += (input[b, s, d] - mean) * (input[b, s, d] - mean);
                float std = MathF.Sqrt(ss / dModel + 1e-5f);
                for (int d = 0; d < dModel; d++) input[b, s, d] = (input[b, s, d] - mean) / std;
            }

            // Verify input is finite (catch test-bug failures distinct from MHA failures).
            for (int i = 0; i < input.Length; i++)
            {
                float iv = input.Data.Span[i];
                Assert.True(!float.IsNaN(iv) && !float.IsInfinity(iv), $"input[{i}] = {iv} (test setup error)");
            }

            // Forward through MHA.
            var output = mha.Forward(input);

            // Check output.
            int nanCount = 0, infCount = 0;
            for (int i = 0; i < output.Length; i++)
            {
                float v = output.Data.Span[i];
                if (float.IsNaN(v)) nanCount++;
                else if (float.IsInfinity(v)) infCount++;
            }
            if (nanCount > 0 || infCount > 0)
            {
                nanTrials++;
                if (nanTrials <= 3)
                {
                    _output.WriteLine($"Trial {trial}: output NaN={nanCount}/{output.Length}, Inf={infCount}");
                }
            }
        }

        _output.WriteLine($"Total trials with NaN/Inf output: {nanTrials}/{trials}");
        Assert.Equal(0, nanTrials);
    }

    /// <summary>
    /// Full-stack repro — reproduces the bug consumer-side. Builds a
    /// 2-layer Transformer at V=50,257 with realistic training (140
    /// samples × 2 epochs via <c>AdamOptimizerOptions.MaxIterations = 2</c>,
    /// matching the consumer-side WT2 9000-token / stride-64 setup), then
    /// asserts that Transformer.Predict produces finite logits across 100
    /// random input contexts. Consumer-side data showed ~25% of inputs
    /// produce all-NaN logits.
    /// </summary>
    [Fact]
    public void Transformer_V50257_Predict_ProducesFiniteLogits_OnRandomContexts()
    {
        AiDotNetEngine.ResetToCpu();
        const int vocab = 50257;

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 2, numDecoderLayers: 0, numHeads: 2,
            modelDimension: 128, feedForwardDimension: 256,
            inputSize: 64, outputSize: vocab,
            maxSequenceLength: 64,
            vocabularySize: vocab,
            randomSeed: 0);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
        var opts = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-4, MaxIterations = 2, UseAdaptiveLearningRate = false,
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, opts);

        // Realistic training (140 samples × 1 epoch) — matches the consumer-side
        // training amount (9000 WT2 tokens / stride 64 = 140 samples × 2 epochs).
        // 10-sample run is below the threshold to trigger NaN.
        const int nTrain = 140;
        var xTrain = new Tensor<float>([nTrain, 64]);
        var yTrain = new Tensor<float>([nTrain, vocab]);
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < nTrain; i++)
        {
            for (int s = 0; s < 64; s++) xTrain[i, s] = rng.Next(0, vocab);
            yTrain[i, rng.Next(0, vocab)] = 1.0f;
        }

        // Build via AiModelBuilder facade — matches the consumer-side path.
        // Note: this test does NOT require an AiDotNet license to be set
        // because we go through the public AiModelBuilder ctor that doesn't
        // gate on license for in-process use in AiDotNet's own test suite.
        var builderType = typeof(AiModelBuilder<float, Tensor<float>, Tensor<float>>);
        // Find a parameterless or default ctor — falls back to direct nn.Train
        // if none works.
        var defaultCtor = builderType.GetConstructor(System.Type.EmptyTypes);
        if (defaultCtor != null)
        {
            var builder = (AiModelBuilder<float, Tensor<float>, Tensor<float>>)defaultCtor.Invoke(null);
            builder.ConfigureModel(model).ConfigureOptimizer(optimizer)
                   .ConfigureDataLoader(DataLoaders.FromTensors<float>(xTrain, yTrain))
                   .BuildAsync().GetAwaiter().GetResult();
        }
        else
        {
            // Fallback — call nn.Train directly (matches the consumer-side
            // pre-#1380-fix bypass path).
            model.Train(xTrain, yTrain);
        }

        // Scan 100 random input contexts for non-finite output (NaN or +/-Infinity).
        // The "finite logits" contract requires BOTH — NaN-only checks would let
        // saturated overflow-style failures (e.g. an exp() that diverges to +Inf
        // through the softmax/cross-entropy boundary) silently pass.
        // (Use explicit IsNaN || IsInfinity instead of float.IsFinite — IsFinite
        // is .NET Core 2.1+, but this test project multi-targets net471 which
        // doesn't expose it.)
        int nonFiniteInputs = 0;
        for (int trial = 0; trial < 100; trial++)
        {
            var input = new Tensor<float>([1, 64]);
            for (int s = 0; s < 64; s++) input[0, s] = rng.Next(0, vocab);
            var pred = model.Predict(input);
            for (int v = 0; v < vocab; v++)
            {
                float lv = pred[0, v];
                if (float.IsNaN(lv) || float.IsInfinity(lv)) { nonFiniteInputs++; break; }
            }
        }

        _output.WriteLine($"Non-finite-producing input contexts: {nonFiniteInputs}/100");
        // Strict assertion — any NaN or +/-Infinity logit is a forward-pass bug.
        Assert.Equal(0, nonFiniteInputs);
    }

    /// <summary>
    /// Direct-train repro WITH aggressive Gen2 GC.Collect between Train and
    /// Predict — the consumer's #1415 comment 2 identified this as the
    /// root-cause trigger ("AiDotNet's internal tensor state being corrupted
    /// by user-level GC.Collect(2, GCCollectionMode.Aggressive, blocking:
    /// true) between model.Train and model.Predict at V=50,257"). This test
    /// asserts the contract holds AND aggressive Gen2 GC between Train and
    /// Predict doesn't reclaim any tensor state the predict path still
    /// depends on. ~2.5 min wall time on CPU.
    /// </summary>
    [Fact]
    public void Transformer_V50257_DirectTrain_AggressiveGC_PredictProducesFiniteLogits()
    {
        // Consumer comment 2 on issue #1415 claims the bug surfaces ONLY when
        // an aggressive Gen2 GC.Collect runs between Train and Predict at
        // V=50,257. Same setup as DirectTrain test above, plus the
        // GC.Collect call the consumer reported as the actual trigger.
        // Expected to fail on the pre-fix code path; expected to pass after
        // the upstream allocator/state-tracking fix lands.
        AiDotNetEngine.ResetToCpu();
        const int vocab = 50257;

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 2, numDecoderLayers: 0, numHeads: 2,
            modelDimension: 128, feedForwardDimension: 256,
            inputSize: 64, outputSize: vocab,
            maxSequenceLength: 64,
            vocabularySize: vocab,
            randomSeed: 0);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        const int nTrain = 140;
        const int epochs = 2;
        var rng = RandomHelper.CreateSeededRandom(42);

        var trainXs = new Tensor<float>[nTrain];
        var trainYs = new Tensor<float>[nTrain];
        for (int i = 0; i < nTrain; i++)
        {
            var x = new Tensor<float>([1, 64]);
            for (int s = 0; s < 64; s++) x[0, s] = rng.Next(0, vocab);
            var y = new Tensor<float>([1, vocab]);
            y[0, rng.Next(0, vocab)] = 1.0f;
            trainXs[i] = x;
            trainYs[i] = y;
        }

        for (int epoch = 0; epoch < epochs; epoch++)
            for (int i = 0; i < nTrain; i++)
                model.Train(trainXs[i], trainYs[i]);

        // Drop training tensors and force aggressive Gen2 collection — exact
        // pattern the consumer reported reproducing the bug.
        for (int i = 0; i < nTrain; i++) { trainXs[i] = null!; trainYs[i] = null!; }
        System.GC.Collect(2, System.GCCollectionMode.Aggressive, blocking: true);
        System.GC.WaitForPendingFinalizers();
        System.GC.Collect(2, System.GCCollectionMode.Aggressive, blocking: true);

        int nonFiniteInputs = 0;
        for (int trial = 0; trial < 100; trial++)
        {
            var input = new Tensor<float>([1, 64]);
            for (int s = 0; s < 64; s++) input[0, s] = rng.Next(0, vocab);
            var pred = model.Predict(input);
            for (int v = 0; v < vocab; v++)
            {
                float lv = pred[0, v];
                if (float.IsNaN(lv) || float.IsInfinity(lv)) { nonFiniteInputs++; break; }
            }
        }

        _output.WriteLine($"Non-finite-producing input contexts (direct-train + Aggressive GC): {nonFiniteInputs}/100");
        Assert.Equal(0, nonFiniteInputs);
    }

    /// <summary>
    /// Direct-train repro WITHOUT the BuildAsync facade — calls model.Train
    /// per-sample for two explicit epochs (effective 280 steps), matching
    /// the consumer's reported training schedule. Verifies the contract
    /// holds whether the model is trained via AiModelBuilder.BuildAsync
    /// (above) or via direct model.Train calls. ~2.5 min wall time on CPU.
    /// </summary>
    [Fact]
    public void Transformer_V50257_DirectTrain_PredictProducesFiniteLogits()
    {
        AiDotNetEngine.ResetToCpu();
        const int vocab = 50257;

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 2, numDecoderLayers: 0, numHeads: 2,
            modelDimension: 128, feedForwardDimension: 256,
            inputSize: 64, outputSize: vocab,
            maxSequenceLength: 64,
            vocabularySize: vocab,
            randomSeed: 0);
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        const int nTrain = 140;
        const int epochs = 2;
        var rng = RandomHelper.CreateSeededRandom(42);

        // Per-sample tensors so we drive model.Train(x, y) directly — same
        // path the consumer reported reproducing on.
        var trainXs = new Tensor<float>[nTrain];
        var trainYs = new Tensor<float>[nTrain];
        for (int i = 0; i < nTrain; i++)
        {
            var x = new Tensor<float>([1, 64]);
            for (int s = 0; s < 64; s++) x[0, s] = rng.Next(0, vocab);
            var y = new Tensor<float>([1, vocab]);
            y[0, rng.Next(0, vocab)] = 1.0f;
            trainXs[i] = x;
            trainYs[i] = y;
        }

        for (int epoch = 0; epoch < epochs; epoch++)
            for (int i = 0; i < nTrain; i++)
                model.Train(trainXs[i], trainYs[i]);

        int nonFiniteInputs = 0;
        for (int trial = 0; trial < 100; trial++)
        {
            var input = new Tensor<float>([1, 64]);
            for (int s = 0; s < 64; s++) input[0, s] = rng.Next(0, vocab);
            var pred = model.Predict(input);
            for (int v = 0; v < vocab; v++)
            {
                float lv = pred[0, v];
                if (float.IsNaN(lv) || float.IsInfinity(lv)) { nonFiniteInputs++; break; }
            }
        }

        _output.WriteLine($"Non-finite-producing input contexts (direct-train): {nonFiniteInputs}/100");
        Assert.Equal(0, nonFiniteInputs);
    }
}
