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
    /// 2-layer Transformer at V=50,257 with brief training (10 samples
    /// to avoid the multi-minute build), then asserts that
    /// Transformer.Predict produces finite logits across 100 random
    /// input contexts. Consumer-side data showed ~25% of inputs produce
    /// all-NaN logits.
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

        // Scan 100 random input contexts for NaN output.
        int nanInputs = 0;
        for (int trial = 0; trial < 100; trial++)
        {
            var input = new Tensor<float>([1, 64]);
            for (int s = 0; s < 64; s++) input[0, s] = rng.Next(0, vocab);
            var pred = model.Predict(input);
            for (int v = 0; v < vocab; v++)
            {
                if (float.IsNaN(pred[0, v])) { nanInputs++; break; }
            }
        }

        _output.WriteLine($"NaN-producing input contexts: {nanInputs}/100");
        // Strict assertion — any NaN logit is a forward-pass bug.
        Assert.Equal(0, nanInputs);
    }
}
