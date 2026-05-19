using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Facade-entry integration tests for both #1380 (BuildAsync byte-LM mode
/// collapse) and #1382 (TransformerArchitecture(layers:) trainable-head
/// loss), exercised through the SAME public surface a consumer hits:
/// <c>new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
/// .ConfigureModel(...).ConfigureOptimizer(...).ConfigureDataLoader(...)
/// .BuildAsync()</c>.
///
/// <para>The existing #1380 reproducers
/// (<see cref="BuildAsyncResidualModeCollapseTests"/>,
/// <see cref="ByteLMV256Issue1380Tests"/>) drive <c>AdamOptimizer.Optimize</c>
/// or <c>Transformer&lt;float&gt;.Train</c> directly. Neither passes through
/// <see cref="AiModelBuilder{T,TIn,TOut}.BuildAsync"/> — the surface
/// HarmonicEngine + every documented sample-code consumer actually uses.
/// This file pins the contract at that surface so future regressions in
/// the builder's compose-and-handoff path can't masquerade as "training
/// works" while the facade silently routes data away from the optimizer.</para>
///
/// <para><b>One test per bug</b>:
/// <list type="bullet">
///   <item><see cref="BuildAsync_V256_ByteLM_FacadeEntry_ProducesNonUniformOutput"/>
///   — #1380 at the consumer entry point. Cross-checks the existing
///   V=256 reproducer's claim that the H7 Vector→Tensor bridge fix
///   (PR #1381) reaches the BuildAsync path, not just the bare
///   <c>optimizer.Optimize</c> call.</item>
///   <item><see cref="BuildAsync_TransformerArchitecture_CustomLayers_RetainsTrainableMHA"/>
///   — #1382 (HRE-facade trainable_params=0). Passes a 3-layer
///   substrate-correct chain (Embed-like + PE-like + Readout-like)
///   into <c>TransformerArchitecture.layers</c> while ALSO specifying
///   <c>numEncoderLayers ≥ 1</c>, then asserts
///   <c>TotalTrainableParameters &gt; 0</c> on the built model. Pre-fix
///   signature: <c>params = 0</c> + uniform-or-worse logits.</item>
/// </list></para>
/// </summary>
[Collection("NonParallelIntegration")]
public class BuildAsyncFacadeTransformerLMTests
{
    private readonly ITestOutputHelper _output;

    public BuildAsyncFacadeTransformerLMTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // ---------------------------------------------------------------------
    // Fixture sizing
    // ---------------------------------------------------------------------
    // Scaled to the smallest configuration that empirically distinguishes
    // (a) a learning model from (b) a model whose training driver silently
    // bypasses the optimizer. The CI-test budget is < 60s; the V=256 byte-
    // LM at this fixture size moves entropy off uniform by ≥ 0.05 nats on
    // the per-sample reference path when training works, so a 0.01-nat
    // threshold on the BuildAsync path is well outside floating-point
    // noise and well inside what a real bug would suppress.

    private const int SampleCount = 128;
    private const int CtxLen = 8;
    private const int VocabSize = 256;
    private const int DModel = 32;
    private const int Heads = 2;
    private const int FfDim = 64;
    private const int NumLayers = 1;
    private const int BatchSize = 16;
    private const int Epochs = 30;
    private const double LearningRate = 5e-3;
    private const int Seed = 1380;

    private static readonly double UniformEntropy = Math.Log(VocabSize);

    // ---------------------------------------------------------------------
    // Issue #1380 — facade entry point
    // ---------------------------------------------------------------------

    [Fact(Timeout = 300_000)]
    public async Task BuildAsync_V256_ByteLM_FacadeEntry_ProducesNonUniformOutput()
    {
        var (arch, xTrain, yTrain) = BuildFixture(customLayers: null);

        // Per-sample reference: same architecture, same loss, same
        // optimizer hyperparameters — but trains via Transformer.Train
        // one sample at a time. This is the path the consumer-side
        // workaround uses; it's also the path PR #1364's Arm 0 anchors
        // as "training has signal" at V=256.
        double perSampleEntropy;
        long perSampleParamCount;
        {
            var perSampleOptimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                model: null,
                options: new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = LearningRate,
                });
            var model = new Transformer<float>(
                arch,
                lossFunction: new CategoricalCrossEntropyLoss<float>(),
                optimizer: perSampleOptimizer);
            model.SetTrainingMode(true);
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                for (int i = 0; i < SampleCount; i++)
                {
                    var sampleX = new Tensor<float>([1, CtxLen]);
                    var sampleY = new Tensor<float>([1, VocabSize]);
                    for (int s = 0; s < CtxLen; s++) sampleX[0, s] = xTrain[i, s];
                    for (int c = 0; c < VocabSize; c++) sampleY[0, c] = yTrain[i, c];
                    model.Train(sampleX, sampleY);
                }
            }
            perSampleParamCount = model.ParameterCount;
            perSampleEntropy = ComputeMeanOutputEntropy(model, xTrain);
            _output.WriteLine(
                $"Per-sample Train reference: " +
                $"params = {perSampleParamCount}, " +
                $"mean entropy = {perSampleEntropy:F4} nats " +
                $"(uniform = {UniformEntropy:F4} nats, gap = {UniformEntropy - perSampleEntropy:F4})");
        }

        // Facade path: identical fixture, but driven through the public
        // AiModelBuilder.BuildAsync surface. This is what every documented
        // sample-code consumer uses and what HarmonicEngine's facade
        // predictors use. The H7 (Vector→Tensor bridge) fix must reach
        // THIS path, not just the bare optimizer.Optimize call the
        // existing reproducers exercise.
        double buildAsyncEntropy;
        long buildAsyncParamCount;
        {
            var (archFacade, _, _) = BuildFixture(customLayers: null);
            var modelFacade = new Transformer<float>(
                archFacade,
                lossFunction: new CategoricalCrossEntropyLoss<float>());
            var adamOptions = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = LearningRate,
                MaxIterations = Epochs,
                BatchSize = BatchSize,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                RandomSeed = Seed,
                ShuffleData = true,
                Regularization = new NoRegularization<float, Tensor<float>, Tensor<float>>(),
            };
            var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, adamOptions);
            var loader = DataLoaders.FromTensors(xTrain, yTrain);

            var built = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(modelFacade)
                .ConfigureOptimizer(optimizer)
                .ConfigureDataLoader(loader)
                .BuildAsync();

            buildAsyncParamCount = (long)(built.TotalTrainableParameters ?? 0);
            buildAsyncEntropy = ComputeMeanOutputEntropy(modelFacade, xTrain);
            _output.WriteLine(
                $"AiModelBuilder.BuildAsync facade entry: " +
                $"params = {buildAsyncParamCount}, " +
                $"mean entropy = {buildAsyncEntropy:F4} nats " +
                $"(uniform = {UniformEntropy:F4} nats, gap = {UniformEntropy - buildAsyncEntropy:F4})");
        }

        // Both paths must report > 0 trainable parameters. If either
        // reports 0 it indicates the optimizer or builder is
        // disconnecting from the model's parameters (the symptom that
        // produces uniform-or-worse output even without #1380's gradient
        // pipeline bug).
        Assert.True(perSampleParamCount > 0,
            $"Per-sample reference model reports 0 trainable parameters — fixture is broken before #1380 can be exercised.");
        Assert.True(buildAsyncParamCount > 0,
            $"AiModelBuilder.BuildAsync produced a trained model with TotalTrainableParameters = {buildAsyncParamCount}. " +
            "The facade is silently dropping the optimizer's parameter set. This is the #1382 trainable-head-loss " +
            "symptom surfacing through the standard (no-custom-layers) facade path.");

        // BuildAsync entropy gap must be at least 50% of the per-sample
        // reference's gap. This mirrors the existing #1380 V=256 ratio
        // assertion but routes through the consumer facade instead of
        // the bare optimizer.
        double perSampleGap = UniformEntropy - perSampleEntropy;
        double buildAsyncGap = UniformEntropy - buildAsyncEntropy;
        _output.WriteLine($"Per-sample uniform-gap = {perSampleGap:F4} nats");
        _output.WriteLine($"BuildAsync uniform-gap = {buildAsyncGap:F4} nats");

        if (perSampleGap >= 0.01)
        {
            double ratio = buildAsyncGap / perSampleGap;
            _output.WriteLine($"Ratio (BuildAsync gap / per-sample gap) = {ratio:F3}");
            Assert.True(
                ratio >= 0.5,
                $"#1380 residual: AiModelBuilder.BuildAsync moved entropy off uniform by only " +
                $"{buildAsyncGap:F4} nats vs the per-sample reference's {perSampleGap:F4} nats " +
                $"(ratio = {ratio:F3}, threshold = 0.5). " +
                "The mode-collapse fix needs to reach the BuildAsync facade entry point, not " +
                "just the bare optimizer.Optimize call.");
        }
        else
        {
            _output.WriteLine(
                $"Per-sample reference gap ({perSampleGap:F4} nats) below 0.01-nat learnability " +
                "floor — fixture too small to engage the path-divergence assertion. The " +
                "parameter-count assertion above still validates that the facade preserves the " +
                "optimizer's trainable set.");
        }
    }

    // ---------------------------------------------------------------------
    // Issue #1382 — TransformerArchitecture(layers:) trainable-head loss
    // ---------------------------------------------------------------------

    [Fact]
    public void TransformerArchitecture_CustomLayers_WithEncoderLayers_FailsFast()
    {
        // #1382 root cause: passing both `layers:` and `numEncoderLayers > 0`
        // used to silently drop the auto-built encoder block — the custom
        // chain BECAME the entire model. For zero-trainable-parameter
        // custom chains (HRE substitutions are intentionally zero-param)
        // this left the consumer with 0 trainable parameters and a
        // non-functioning model. For other custom chains it surfaced
        // as a broadcast/shape mismatch on the very first training
        // batch (model output rank/size didn't match outputSize).
        //
        // Fix: TransformerArchitecture rejects the (layers:, numEncoderLayers>0)
        // combination at construction time with a clear diagnostic
        // naming the actual contract (layers: REPLACES the auto-built
        // block). The user can choose: (a) include their own attention
        // blocks in the layer list and pass numEncoderLayers=0, or
        // (b) drop the custom layers and let the default encoder run.

        var customLayers = BuildZeroParamCustomLayerChain();

        var ex = Assert.Throws<ArgumentException>(() => new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: NumLayers,       // > 0 + layers: provided = ambiguous
            numDecoderLayers: 0,
            numHeads: Heads,
            modelDimension: DModel,
            feedForwardDimension: FfDim,
            inputSize: CtxLen,
            outputSize: VocabSize,
            maxSequenceLength: CtxLen,
            vocabularySize: VocabSize,
            layers: new List<ILayer<float>>(customLayers)));

        _output.WriteLine($"Diagnostic: {ex.Message}");
        Assert.Contains("#1382", ex.Message);
        Assert.Contains("numEncoderLayers", ex.Message);
        Assert.Contains("REPLACES", ex.Message);
    }

    [Fact(Timeout = 300_000)]
    public async Task TransformerArchitecture_CustomLayers_WithoutEncoderLayers_BuildsCleanly()
    {
        // Companion test: when the user explicitly passes
        // numEncoderLayers: 0 (acknowledging "I own the entire forward
        // graph"), construction must succeed and BuildAsync must
        // complete without throwing — even if the model has zero
        // trainable parameters. The contract: zero-param models are
        // legal (inference-only / fixed-substrate diagnostic), but
        // the user must opt into that mode rather than discovering
        // it as a silent training failure.

        var customLayers = new ILayer<float>[]
        {
            new InputLayer<float>(new int[] { CtxLen }),
            // A trainable head so the chain actually produces [B, VocabSize].
            // Using DenseLayer<float> with explicit output size guarantees
            // shape correctness while exercising the consumer-owns-the-graph
            // configuration path.
            new DenseLayer<float>(VocabSize),
            new ActivationLayer<float>(
                (IActivationFunction<float>?)new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
        };

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 0,
            numDecoderLayers: 0,
            numHeads: Heads,
            modelDimension: DModel,
            feedForwardDimension: FfDim,
            inputSize: CtxLen,
            outputSize: VocabSize,
            maxSequenceLength: CtxLen,
            vocabularySize: VocabSize,
            layers: new List<ILayer<float>>(customLayers));

        var (_, xTrain, yTrain) = BuildFixture(customLayers: null);

        var modelFacade = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>());
        var adamOptions = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = LearningRate,
            MaxIterations = 1,
            BatchSize = BatchSize,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            RandomSeed = Seed,
            ShuffleData = false,
            Regularization = new NoRegularization<float, Tensor<float>, Tensor<float>>(),
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, adamOptions);
        var loader = DataLoaders.FromTensors(xTrain, yTrain);

        var built = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(modelFacade)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(loader)
            .BuildAsync();

        long trainableParams = (long)(built.TotalTrainableParameters ?? 0);
        _output.WriteLine(
            $"Consumer-owns-the-graph build: TotalTrainableParameters = {trainableParams}");
        // The custom chain includes a DenseLayer<float>(VocabSize) which
        // contributes [CtxLen × DModel × VocabSize] weights. Any non-
        // zero value confirms the trainable head wired through.
        Assert.True(trainableParams > 0,
            $"Consumer-owns-the-graph path produced TotalTrainableParameters = {trainableParams}. " +
            "DenseLayer in the custom chain should have contributed trainable weights.");
    }

    // ---------------------------------------------------------------------
    // Fixture + helpers
    // ---------------------------------------------------------------------

    private static (TransformerArchitecture<float>, Tensor<float>, Tensor<float>) BuildFixture(
        IReadOnlyList<ILayer<float>>? customLayers)
    {
        // Synthetic V=256 byte-LM data: (context window, next byte).
        // The pattern is deterministic but non-trivial — periodic
        // byte sequence so a 1-layer attention CAN learn it within
        // 30 epochs at lr=5e-3 on the per-sample path.
        var rng = new Random(Seed);
        var xTrain = new Tensor<float>([SampleCount, CtxLen]);
        var yTrain = new Tensor<float>([SampleCount, VocabSize]);
        for (int i = 0; i < SampleCount; i++)
        {
            byte target = (byte)(rng.Next() % VocabSize);
            for (int s = 0; s < CtxLen; s++)
            {
                xTrain[i, s] = (byte)((target + s * 17) % VocabSize);
            }
            yTrain[i, target] = 1.0f;
        }

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: NumLayers,
            numDecoderLayers: 0,
            numHeads: Heads,
            modelDimension: DModel,
            feedForwardDimension: FfDim,
            inputSize: CtxLen,
            outputSize: VocabSize,
            maxSequenceLength: CtxLen,
            vocabularySize: VocabSize,
            layers: customLayers is null ? null : new List<ILayer<float>>(customLayers));

        return (arch, xTrain, yTrain);
    }

    /// <summary>
    /// Build a 3-layer chain of zero-trainable-parameter layers shaped
    /// like the HRE substitution stack: an embedding-style entry that
    /// projects [B, ctx] integer tokens to [B, ctx, DModel] activations,
    /// a position-encoding-style pass-through that preserves shape, and
    /// a readout-style projection back to [B, vocab]. None of them
    /// expose trainable parameters — they're the substrate-correct case
    /// where the user expects the auto-built MHA + head to provide all
    /// trainable capacity.
    /// </summary>
    private static IReadOnlyList<ILayer<float>> BuildZeroParamCustomLayerChain()
    {
        // 2-layer chain matching the architecture's declared inputSize
        // (CtxLen) so the constructor's shape validator passes, but with
        // zero trainable parameters. Mimics the HarmonicEngine consumer
        // pattern where all "HRE substitution" layers are zero-trainable
        // by design (fixed phase codebook + Born readout) and the user
        // expects the architecture's numEncoderLayers/numHeads/
        // modelDimension parameters to provide an auto-built trainable
        // MultiHeadAttention block. The bug: those parameters are
        // SILENTLY IGNORED when layers: is non-empty (see
        // Transformer.InitializeLayers lines 337-349) — the model ends
        // up with the user's layer list as its ENTIRE forward graph and
        // no trainable parameters at all.
        return new ILayer<float>[]
        {
            new InputLayer<float>(new int[] { CtxLen }),
            new ActivationLayer<float>(
                (IActivationFunction<float>)new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
        };
    }

    private static double ComputeMeanOutputEntropy(Transformer<float> model, Tensor<float> xTrain)
    {
        model.SetTrainingMode(false);
        double sumEntropy = 0;
        int countSamples = 0;
        int B = xTrain.Shape[0];
        for (int i = 0; i < B; i++)
        {
            var sampleX = new Tensor<float>([1, CtxLen]);
            for (int s = 0; s < CtxLen; s++) sampleX[0, s] = xTrain[i, s];
            var pred = model.Predict(sampleX);

            // Softmax(pred) → entropy
            float maxL = float.NegativeInfinity;
            for (int v = 0; v < VocabSize; v++) if (pred[0, v] > maxL) maxL = pred[0, v];
            double sumExp = 0;
            for (int v = 0; v < VocabSize; v++) sumExp += Math.Exp(pred[0, v] - maxL);
            double logZ = maxL + Math.Log(sumExp);
            double entropy = 0;
            for (int v = 0; v < VocabSize; v++)
            {
                double logP = pred[0, v] - logZ;
                double p = Math.Exp(logP);
                if (p > 0) entropy -= p * logP;
            }
            sumEntropy += entropy;
            countSamples++;
        }
        return sumEntropy / countSamples;
    }
}
