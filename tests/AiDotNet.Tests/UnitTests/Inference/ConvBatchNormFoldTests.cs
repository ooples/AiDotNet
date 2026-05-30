using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Inference;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// Phase 6 (freeze-time super-folding): verifies that
/// <c>InferenceOptimizer</c> folds a <c>Conv2D(identity)→BatchNorm</c> block into
/// the convolution's weights/bias and removes the BatchNorm layer, with the
/// inference output unchanged (lossless to floating-point rounding). Folding is
/// the canonical ResNet/VGG inference optimization: at inference BatchNorm is a
/// fixed per-channel affine, so <c>BN(Conv(x)) = Conv'(x)</c> for adjusted
/// weights/bias.
/// </summary>
public class ConvBatchNormFoldTests
{
    private static ConvolutionalNeuralNetwork<float> BuildConvBnModel()
    {
        // Conv2D(2→8, 3×3, pad 1, IDENTITY activation) → BatchNorm(8) → Flatten → Dense(4).
        // The conv must have identity activation for the fold to be valid (BN sits
        // directly on the linear conv output, before any nonlinearity).
        var layers = new System.Collections.Generic.List<ILayer<float>>
        {
            new ConvolutionalLayer<float>(outputDepth: 8, kernelSize: 3, stride: 1, padding: 1,
                                          activationFunction: new IdentityActivation<float>()),
            new BatchNormalizationLayer<float>(),
            new FlattenLayer<float>(),
            new DenseLayer<float>(4, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 6, inputWidth: 6, inputDepth: 2,
            outputSize: 4,
            layers: layers);
        return new ConvolutionalNeuralNetwork<float>(arch);
    }

    private static Tensor<float> RandomInput(Random rng, int batch = 1)
    {
        var data = new float[batch * 2 * 6 * 6];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return new Tensor<float>(data, new[] { batch, 2, 6, 6 });
    }

    // Accumulate non-trivial BatchNorm running mean/variance by pushing a few
    // random batches through the layer stack in training mode (BN updates its
    // running stats via momentum on each training forward). No reflection, no
    // full Train() stack — just the public Layer.Forward path.
    private static void WarmBatchNormStats(ConvolutionalNeuralNetwork<float> model, Random rng)
    {
        model.SetTrainingMode(true);
        for (int step = 0; step < 8; step++)
        {
            Tensor<float> t = RandomInput(rng, batch: 4);
            foreach (var layer in model.Layers)
                t = layer.Forward(t);
        }
        model.SetTrainingMode(false);
    }

    [Fact]
    public void FoldBatchNorm_RemovesLayer_AndPreservesOutput()
    {
        var rng = new Random(20260530);
        var model = BuildConvBnModel();

        // Give gamma/beta non-trivial values (default gamma=1, beta=0 would make BN
        // a pure normalization — set them so the fold exercises the affine too).
        var bn = (BatchNormalizationLayer<float>)model.Layers[1];
        var p = bn.GetParameters();
        for (int i = 0; i < p.Length; i++) p[i] = (float)(rng.NextDouble() * 1.5 + 0.25); // gamma then beta, all > 0
        bn.SetParameters(p);

        WarmBatchNormStats(model, rng);

        // Reference output BEFORE folding (inference mode → BN uses running stats).
        var x = RandomInput(rng);
        var reference = model.Predict(x).ToArray();
        Assert.Contains(model.Layers, l => l is BatchNormalizationLayer<float>);

        // Fold via the inference optimizer (no clone — mutate this instance).
        var config = new InferenceOptimizationConfig
        {
            EnableLayerFusion = true,
            EnableKVCache = false,
            EnableFlashAttention = false,
            EnableSpeculativeDecoding = false,
            EnableWeightOnlyQuantization = false,
        };
        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied, "Expected layer fusion to apply (Conv→BN present).");
        Assert.DoesNotContain(optimized.Layers, l => l is BatchNormalizationLayer<float>);

        // Folded output must match the pre-fold reference (lossless).
        var folded = optimized.Predict(x).ToArray();
        Assert.Equal(reference.Length, folded.Length);
        double maxDiff = 0;
        for (int i = 0; i < reference.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(reference[i] - folded[i]));
        Assert.True(maxDiff <= 1e-4, $"Folded output diverged from reference by {maxDiff:E3}");
    }

    [Fact]
    public void FoldBatchNorm_Disabled_LeavesModelUnchanged()
    {
        var model = BuildConvBnModel();
        int bnBefore = model.Layers.FindAll(l => l is BatchNormalizationLayer<float>).Count;

        var config = new InferenceOptimizationConfig
        {
            EnableLayerFusion = false,
            EnableKVCache = false,
            EnableFlashAttention = false,
            EnableSpeculativeDecoding = false,
            EnableWeightOnlyQuantization = false,
        };
        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, _) = optimizer.OptimizeForInference(model, cloneModel: false);

        int bnAfter = optimized.Layers.FindAll(l => l is BatchNormalizationLayer<float>).Count;
        Assert.Equal(bnBefore, bnAfter);   // fusion off → BN retained
        Assert.True(bnAfter > 0);
    }

    [Fact]
    public void FoldBatchNorm_IntoDense_PreservesOutput()
    {
        var rng = new Random(20260601);
        // Dense(16→8, IDENTITY) → BatchNorm → Dense(4). BN folds into the first Dense.
        var layers = new System.Collections.Generic.List<ILayer<float>>
        {
            new DenseLayer<float>(8, activationFunction: new IdentityActivation<float>()),
            new BatchNormalizationLayer<float>(),
            new DenseLayer<float>(4, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 16,
            outputSize: 4,
            layers: layers);
        var model = new FeedForwardNeuralNetwork<float>(arch);

        var bn = (BatchNormalizationLayer<float>)model.Layers[1];
        var p = bn.GetParameters();
        for (int i = 0; i < p.Length; i++) p[i] = (float)(rng.NextDouble() * 1.5 + 0.25);
        bn.SetParameters(p);

        // Warm BN running stats with a few training-mode forwards.
        model.SetTrainingMode(true);
        for (int step = 0; step < 8; step++)
        {
            var dd = new float[4 * 16];
            for (int i = 0; i < dd.Length; i++) dd[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            Tensor<float> t = new Tensor<float>(dd, new[] { 4, 16 });
            foreach (var layer in model.Layers) t = layer.Forward(t);
        }
        model.SetTrainingMode(false);

        var xd = new float[16];
        for (int i = 0; i < xd.Length; i++) xd[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var x = new Tensor<float>(xd, new[] { 1, 16 });
        var reference = model.Predict(x).ToArray();

        var config = new InferenceOptimizationConfig
        {
            EnableLayerFusion = true,
            EnableKVCache = false,
            EnableFlashAttention = false,
            EnableSpeculativeDecoding = false,
            EnableWeightOnlyQuantization = false,
        };
        var (optimized, anyApplied) = new InferenceOptimizer<float>(config).OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        Assert.DoesNotContain(optimized.Layers, l => l is BatchNormalizationLayer<float>);
        var folded = optimized.Predict(x).ToArray();
        double maxDiff = 0;
        for (int i = 0; i < reference.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(reference[i] - folded[i]));
        Assert.True(maxDiff <= 1e-4, $"Dense→BN folded output diverged by {maxDiff:E3}");
    }

    [Fact]
    public void FoldBatchNorm_SkipsWhenConvHasNonIdentityActivation()
    {
        // Conv with ReLU activation followed by BN: folding would be INVALID
        // (a nonlinearity sits between the linear op and BN), so it must be skipped.
        var layers = new System.Collections.Generic.List<ILayer<float>>
        {
            new ConvolutionalLayer<float>(outputDepth: 8, kernelSize: 3, stride: 1, padding: 1,
                                          activationFunction: new ReLUActivation<float>()),
            new BatchNormalizationLayer<float>(),
            new FlattenLayer<float>(),
            new DenseLayer<float>(4, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 6, inputWidth: 6, inputDepth: 2,
            outputSize: 4,
            layers: layers);
        var model = new ConvolutionalNeuralNetwork<float>(arch);

        var config = new InferenceOptimizationConfig
        {
            EnableLayerFusion = true,
            EnableKVCache = false,
            EnableFlashAttention = false,
            EnableSpeculativeDecoding = false,
            EnableWeightOnlyQuantization = false,
        };
        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, _) = optimizer.OptimizeForInference(model, cloneModel: false);

        // ReLU between conv and BN → fold must NOT fire; BN stays.
        Assert.Contains(optimized.Layers, l => l is BatchNormalizationLayer<float>);
    }
}
