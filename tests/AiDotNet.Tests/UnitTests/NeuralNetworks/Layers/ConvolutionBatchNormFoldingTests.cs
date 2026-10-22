using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.TestInfrastructure;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

[CollectionDefinition("InferenceFusion", DisableParallelization = true)]
public sealed class InferenceFusionCollection { }

/// <summary>
/// Inference folds a batch norm (and a ReLU after it) into the preceding convolution
/// (<see cref="InferenceFusionSettings.FoldBatchNormIntoConvolution"/>). The folded convolution must equal the unfused
/// layers up to rounding, refold when any source tensor changes in place, and never apply in training.
/// </summary>
[Collection("InferenceFusion")]
public class ConvolutionBatchNormFoldingTests
{
    // Conv (no activation), BN with trained running statistics and non-trivial affine parameters, ReLU; in eval mode.
    private static List<ILayer<T>> Block<T>(int inChannels, int outChannels, int kernel, int stride, int padding, bool bias,
        int seed)
    {
        var conv = new ConvolutionalLayer<T>(outChannels, kernel, stride, padding, (IActivationFunction<T>)new IdentityActivation<T>(),
            biasMode: bias ? BiasMode.Always : BiasMode.Never);
        var norm = new BatchNormalizationLayer<T>();
        var relu = new ActivationLayer<T>(new ReLUActivation<T>() as IActivationFunction<T>);
        var layers = new List<ILayer<T>> { conv, norm, relu };
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var ops = MathHelper.GetNumericOperations<T>();
        // Training forwards move the running statistics off their initial 0 / 1.
        foreach (var layer in layers) layer.SetTrainingMode(true);
        for (int step = 0; step < 4; step++)
        {
            Tensor<T> h = Random<T>(rng, 2, inChannels, 9, 8, ops, scale: 2.0, offset: 0.5);
            foreach (var layer in layers) h = layer.Forward(h);
        }
        // Gamma and beta away from 1 and 0 (the parameter vector also carries the running statistics, so every entry stays
        // positive: a negative variance has no square root).
        var p = norm.GetParameters();
        for (int i = 0; i < p.Length; i++) p[i] = ops.FromDouble(0.25 + rng.NextDouble() * 1.25);
        norm.SetParameters(p);
        foreach (var layer in layers) layer.SetTrainingMode(false);
        return layers;
    }

    private static Tensor<T> Random<T>(Random rng, int b, int c, int h, int w,
        INumericOperations<T> ops, double scale = 1.0, double offset = 0.0)
    {
        var t = new Tensor<T>(new[] { b, c, h, w });
        for (int i = 0; i < t.Length; i++) t[i] = ops.FromDouble((rng.NextDouble() * 2 - 1) * scale + offset);
        return t;
    }

    private static Tensor<T> Unfused<T>(IReadOnlyList<ILayer<T>> layers, Tensor<T> x, int count)
    {
        bool previous = InferenceFusionSettings.FoldBatchNormIntoConvolution;
        InferenceFusionSettings.FoldBatchNormIntoConvolution = false;
        try
        {
            var h = x;
            for (int i = 0; i < count; i++) h = layers[i].Forward(h);
            return h;
        }
        finally
        {
            InferenceFusionSettings.FoldBatchNormIntoConvolution = previous;
        }
    }

    private static void AssertClose<T>(Tensor<T> expected, Tensor<T> actual, double tolerance, string what)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        double scale = 0;
        for (int i = 0; i < expected.Length; i++) scale = Math.Max(scale, Math.Abs(ops.ToDouble(expected[i])));
        for (int i = 0; i < expected.Length; i++)
        {
            double e = ops.ToDouble(expected[i]), a = ops.ToDouble(actual[i]);
            Assert.True(Math.Abs(e - a) <= tolerance * (1 + scale), $"{what} [{i}]: expected {e}, got {a}");
        }
    }

    // Float rounds the folded weights once to float, so it agrees to float precision; double to double precision.
    [Theory]
    [InlineData(3, 1, 1, true, true)]
    [InlineData(3, 2, 1, false, true)]
    [InlineData(1, 1, 0, true, false)]
    public void Folded_block_matches_the_unfused_layers(int kernel, int stride, int padding, bool bias, bool relu)
    {
        TestEngines.EnsureCpu();
        Check<float>(kernel, stride, padding, bias, relu, 2e-5);
        Check<double>(kernel, stride, padding, bias, relu, 1e-12);

        static void Check<T>(int kernel, int stride, int padding, bool bias, bool relu, double tolerance)
        {
            var layers = Block<T>(5, 7, kernel, stride, padding, bias, seed: 41);
            if (!relu) layers.RemoveAt(2);
            var x = Random<T>(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(43), 2, 5, 9, 8,
                MathHelper.GetNumericOperations<T>());
            Assert.True(ConvolutionBatchNormFolding<T>.TryForwardSequence(layers, 0, x, out var folded, out int consumed));
            Assert.Equal(layers.Count, consumed);
            AssertClose(Unfused(layers, x, layers.Count), folded, tolerance, $"{typeof(T).Name} k{kernel} s{stride}");
        }
    }

    [Fact]
    public void An_in_place_parameter_change_refolds()
    {
        TestEngines.EnsureCpu();
        var layers = Block<float>(4, 6, 3, 1, 1, bias: true, seed: 47);
        var x = Random<float>(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(53), 1, 4, 7, 7,
            MathHelper.GetNumericOperations<float>());
        Assert.True(ConvolutionBatchNormFolding<float>.TryForwardSequence(layers, 0, x, out var first, out _));
        AssertClose(Unfused(layers, x, 3), first, 2e-5, "first");

        // The convolution's weights, then the batch norm's gamma and beta, rewritten in place.
        foreach (var layer in new[] { layers[0], layers[1] })
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++) p[i] = p[i] * 0.5f + 0.125f;
            layer.SetParameters(p);
            Assert.True(ConvolutionBatchNormFolding<float>.TryForwardSequence(layers, 0, x, out var after, out _));
            AssertClose(Unfused(layers, x, 3), after, 2e-5, $"after updating {layer.GetType().Name}");
        }
    }

    [Fact]
    public void Training_mode_and_the_setting_turned_off_do_not_fold()
    {
        TestEngines.EnsureCpu();
        var layers = Block<float>(3, 4, 3, 1, 1, bias: true, seed: 59);
        var x = Random<float>(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(61), 1, 3, 6, 6,
            MathHelper.GetNumericOperations<float>());

        layers[0].SetTrainingMode(true);
        Assert.False(ConvolutionBatchNormFolding<float>.TryForwardSequence(layers, 0, x, out _, out _));
        layers[0].SetTrainingMode(false);

        bool previous = InferenceFusionSettings.FoldBatchNormIntoConvolution;
        InferenceFusionSettings.FoldBatchNormIntoConvolution = false;
        try
        {
            Assert.False(ConvolutionBatchNormFolding<float>.TryForwardSequence(layers, 0, x, out _, out _));
        }
        finally
        {
            InferenceFusionSettings.FoldBatchNormIntoConvolution = previous;
        }
    }

    // A convolution with its own activation computes act(conv), so the batch norm after it cannot fold.
    [Fact]
    public void A_convolution_with_its_own_activation_does_not_fold()
    {
        TestEngines.EnsureCpu();
        var conv = new ConvolutionalLayer<float>(4, 3, 1, 1, (IActivationFunction<float>)new ReLUActivation<float>());
        var norm = new BatchNormalizationLayer<float>();
        var x = Random<float>(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(67), 1, 3, 6, 6,
            MathHelper.GetNumericOperations<float>());
        conv.SetTrainingMode(false);
        norm.SetTrainingMode(false);
        norm.Forward(conv.Forward(x));
        Assert.False(ConvolutionBatchNormFolding<float>.TryForward(conv, norm, x, relu: false, out _));
    }

    // torchvision's vgg*_bn and VGG16BNBackbone: convolution -> batch norm -> ReLU, so the ReLU is not inside the conv.
    [Fact]
    public void Vgg_with_batch_norm_orders_convolution_batch_norm_relu()
    {
        var architecture = new NeuralNetworkArchitecture<float>(InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification, NetworkComplexity.Simple, inputHeight: 32, inputWidth: 32,
            inputDepth: 3, outputSize: 10);
        var withBn = LayerHelper<float>.CreateDefaultVGGLayers(architecture,
            new VGGConfiguration(VGGVariant.VGG11_BN, numClasses: 10, inputHeight: 32, inputWidth: 32)).ToList();
        int convs = 0;
        for (int i = 0; i < withBn.Count; i++)
        {
            if (withBn[i] is not ConvolutionalLayer<float> conv) continue;
            convs++;
            Assert.True(conv.ScalarActivation is IdentityActivation<float>, $"conv {convs} has an activation of its own");
            Assert.IsType<BatchNormalizationLayer<float>>(withBn[i + 1]);
            var relu = Assert.IsType<ActivationLayer<float>>(withBn[i + 2]);
            Assert.IsType<ReLUActivation<float>>(relu.ScalarActivation);
        }
        Assert.Equal(8, convs);

        var withoutBn = LayerHelper<float>.CreateDefaultVGGLayers(architecture,
            new VGGConfiguration(VGGVariant.VGG11, numClasses: 10, inputHeight: 32, inputWidth: 32)).ToList();
        foreach (var conv in withoutBn.OfType<ConvolutionalLayer<float>>())
            Assert.IsType<ReLUActivation<float>>(conv.ScalarActivation);
    }
}
