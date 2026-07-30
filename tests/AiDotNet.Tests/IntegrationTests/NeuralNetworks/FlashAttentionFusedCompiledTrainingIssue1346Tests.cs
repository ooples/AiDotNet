using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Consumer-side regression coverage for AiDotNet#1346 (FlashAttentionLayer
/// degenerate output on the compiled fused-Adam path). The original root
/// cause — Engine.FlashAttention missing its GraphMode.IsActive lazy-graph
/// recording branch — was fixed by AiDotNet.Tensors PR #362 and ships in
/// AiDotNet.Tensors NuGet 0.81.3. This file pins TWO things:
/// <list type="number">
/// <item>The engine-side fix actually reaches the AiDotNet fused training
/// path: a Transformer&lt;float&gt; whose layer stack contains
/// <see cref="FlashAttentionLayer{T}"/> engages
/// <see cref="AiDotNet.Training.CompiledTapeTrainingStep{T}.TryStepWithFusedOptimizer"/>
/// when trained via the public network API (canary test below).</item>
/// <item>The Tensors#396 loss-readout regression remains covered through a
/// deterministic reference comparison: the fused plan must return the same
/// positive CCE value computed from the pre-step prediction, rather than a
/// stale or orphaned zero.</item>
/// </list>
/// </summary>
/// <remarks>
/// PR #1386 review (CodeRabbit C8Bm6 + Copilot Drjj5): both tests reset and
/// read <see cref="AiDotNet.Training.CompiledTapeTrainingStep{T}"/>'s
/// thread-static fused-step counter and cache. Default xUnit per-class
/// parallelization would race those resets/reads against any other test
/// touching the same global state (FusedOptimizerIntegrationTests etc.),
/// producing flaky engaged-count assertions or cross-test counter leak.
/// Join the existing "FusedOptimizerGlobalState" collection (defined in
/// <see cref="FusedOptimizerCollection"/>) so xUnit serializes every test
/// in this class with every other CompiledTapeTrainingStep-mutating test.
/// </remarks>
[Collection("FusedOptimizerGlobalState")]
public class FlashAttentionFusedCompiledTrainingIssue1346Tests
{
    private readonly ITestOutputHelper _output;

    public FlashAttentionFusedCompiledTrainingIssue1346Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    private const int SeqLen = 4;
    private const int EmbedDim = 16;
    private const int HeadCount = 2;
    private const int NumClasses = 8;

    /// <summary>
    /// Builds a small Transformer whose explicit layer list contains
    /// <see cref="FlashAttentionLayer{T}"/> as the attention block — the same
    /// drop-in-replacement pattern AiDotNet#1346 documented as broken on the
    /// fused-Adam path before AiDotNet.Tensors PR #362 landed.
    /// </summary>
    private static Transformer<float> BuildFlashAttentionTransformer(double learningRate = 0.01)
    {
        // No EmbeddingLayer: input is continuous-valued [1, seq, embed].
        // EmbeddingLayer-first trips a pre-existing TransformerArchitecture
        // input-dim validator quirk (see TransformerCustomLayerValidationIssue1317IntegrationTests
        // .CustomTransformerLayerStack_AcceptsFlashAttentionLayerAsDropInReplacement)
        // that is unrelated to #1346.
        var layers = new List<ILayer<float>>
        {
            new FlashAttentionLayer<float>(SeqLen, EmbedDim, HeadCount),
            new LayerNormalizationLayer<float>(),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(NumClasses, (IActivationFunction<float>)new IdentityActivation<float>())
        };

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 0, // explicit layers: list replaces the auto-built encoder block (#1382)
            numDecoderLayers: 0,
            numHeads: HeadCount,
            modelDimension: EmbedDim,
            feedForwardDimension: EmbedDim,
            complexity: NetworkComplexity.Medium,
            inputSize: SeqLen * EmbedDim,
            outputSize: NumClasses,
            dropoutRate: 0.0,
            maxSequenceLength: SeqLen,
            vocabularySize: NumClasses,
            usePositionalEncoding: false,
            temperature: 1.0,
            sequencePooling: null,
            layers: layers);

        var optOptions = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = learningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, optOptions);

        return new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            optimizer: optimizer);
    }

    private static Tensor<float> BuildFingerprintInput(int classIndex, int seed)
    {
        var t = new Tensor<float>([1, SeqLen, EmbedDim]);
        var rng = new System.Random(seed * 1000 + classIndex);
        for (int s = 0; s < SeqLen; s++)
        {
            for (int e = 0; e < EmbedDim; e++)
            {
                t[0, s, e] = (float)(classIndex + 0.05 * rng.NextDouble());
            }
        }
        return t;
    }

    private static Tensor<float> BuildOneHotTarget(int classIndex)
    {
        var t = new Tensor<float>([1, NumClasses]);
        t[0, classIndex] = 1f;
        return t;
    }

    /// <summary>
    /// CANARY for AiDotNet.Tensors PR #362's reach into the public AiDotNet
    /// fused-Adam training path. A Transformer whose layer stack contains
    /// FlashAttentionLayer must engage
    /// <see cref="AiDotNet.Training.CompiledTapeTrainingStep{T}.TryStepWithFusedOptimizer"/>
    /// on the first Train() call. Pre-fix (before #362) the GraphMode lazy
    /// trace inside the fused path would still record everything except
    /// FlashAttention; the fused step would run successfully (so this
    /// counter would still increment) but downstream gradient flow would
    /// be broken. This test specifically verifies the canary — a regression
    /// that prevents fused-path engagement at all (e.g. a future Tensors
    /// change that throws during GraphMode trace) would flip this red.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task FlashAttentionLayer_TrainViaFusedCompiledAdam_EngagesFusedPath()
    {
        await Task.Yield();
        AiDotNet.Training.CompiledTapeTrainingStep<float>.ResetFusedStepCount();
        AiDotNet.Training.CompiledTapeTrainingStep<float>.Invalidate();

        var model = BuildFlashAttentionTransformer();
        model.SetTrainingMode(true);

        var input = BuildFingerprintInput(0, seed: 7);
        var target = BuildOneHotTarget(0);

        model.Train(input, target);

        long fusedSteps = AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount();
        _output.WriteLine($"Fused step count after 1 Train() call: {fusedSteps}");

        Assert.True(fusedSteps > 0,
            $"FlashAttentionLayer Transformer fell back to eager on first Train() — " +
            $"CompiledTapeTrainingStep<float>.GetFusedStepCount() = {fusedSteps}. " +
            "This indicates Engine.FlashAttention threw during GraphMode trace OR a " +
            "downstream compile gate rejected the FA-containing graph. See AiDotNet.Tensors " +
            "PR #362 and AiDotNet issue #1346.");
    }

    /// <summary>
    /// Regression for the consumer-side loss-readout gap surfaced during
    /// AiDotNet#1346. It computes the exact pre-step CCE reference from the
    /// initialized model, chooses a class whose clamped probability yields a
    /// positive loss, and requires fused Adam to report the same value through
    /// <see cref="NeuralNetworkBase{T}.GetLastLoss"/>.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task DenseIdentity_CCE_OnFusedAdam_MatchesDeterministicReferenceLoss()
    {
        await Task.Yield();
        AiDotNet.Training.CompiledTapeTrainingStep<float>.ResetFusedStepCount();
        AiDotNet.Training.CompiledTapeTrainingStep<float>.Invalidate();

        // Identity activations keep the final values observable so the test can
        // independently apply CCE's documented [1e-7, 1] clamp.
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(EmbedDim, (IActivationFunction<float>)new IdentityActivation<float>()),
            new LayerNormalizationLayer<float>(),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(NumClasses, (IActivationFunction<float>)new IdentityActivation<float>())
        };
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 0, numDecoderLayers: 0,
            numHeads: HeadCount, modelDimension: EmbedDim, feedForwardDimension: EmbedDim,
            complexity: NetworkComplexity.Medium,
            inputSize: SeqLen * EmbedDim, outputSize: NumClasses,
            dropoutRate: 0.0, maxSequenceLength: SeqLen, vocabularySize: NumClasses,
            usePositionalEncoding: false, temperature: 1.0,
            sequencePooling: null, layers: layers);
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8
            });
        var model = new Transformer<float>(arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            optimizer: optimizer);
        var input = BuildFingerprintInput(0, seed: 42);

        // Select the lowest-scoring class from the exact initialized model and
        // compute CCE independently. This makes the regression deterministic:
        // the previous fixed class could legitimately have a raw logit >= 1,
        // which CCE correctly clamps to 1 and therefore gives an exact zero
        // loss. That was indistinguishable from the historic silent-zero bug.
        model.SetTrainingMode(false);
        // Use the public training-forward surface so the reference sees the
        // exact explicit [batch, seq, dim] shape that Train will replay.
        var prediction = model.ForwardForTraining(input);
        int classOffset = prediction.Length - NumClasses;
        int targetClass = 0;
        float selectedPrediction = prediction[classOffset];
        for (int c = 1; c < NumClasses; c++)
        {
            float candidate = prediction[classOffset + c];
            if (candidate < selectedPrediction)
            {
                selectedPrediction = candidate;
                targetClass = c;
            }
        }

        float clampedPrediction = Math.Clamp(selectedPrediction, 1e-7f, 1.0f);
        float expectedLoss = -MathF.Log(clampedPrediction);
        Assert.True(expectedLoss > 0.0f,
            $"Reference precondition failed: selected class {targetClass} had " +
            $"prediction={selectedPrediction}, clamped={clampedPrediction}.");

        var target = BuildOneHotTarget(targetClass);
        model.SetTrainingMode(true);

        model.Train(input, target);
        long fusedSteps = AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount();
        float lastLoss = model.GetLastLoss();

        _output.WriteLine($"Identity+CCE on fused-Adam: class={targetClass}, " +
            $"prediction={selectedPrediction}, expectedLoss={expectedLoss}, " +
            $"fusedSteps={fusedSteps}, lastLoss={lastLoss}, " +
            $"IsNaN={float.IsNaN(lastLoss)}, IsInfinity={float.IsInfinity(lastLoss)}, " +
            $"IsZero={lastLoss == 0f}");

        Assert.True(fusedSteps > 0, "Fused path must have engaged");

        // A live fused readout must match the independent forward reference;
        // a stale/copy-backed loss buffer would remain at zero and fail here.
        Assert.True(float.IsFinite(lastLoss),
            $"Fused CCE loss must be finite; got {lastLoss}.");
        float tolerance = Math.Max(1e-5f, expectedLoss * 1e-4f);
        Assert.InRange(lastLoss, expectedLoss - tolerance, expectedLoss + tolerance);
    }
}
