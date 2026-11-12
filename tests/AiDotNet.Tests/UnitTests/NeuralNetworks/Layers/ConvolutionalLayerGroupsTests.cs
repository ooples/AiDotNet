using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// General grouped 2-D convolution (PyTorch nn.Conv2d <c>groups</c> between 1 and C_in), which Shift-Net's RepConv
/// needs (groups = C/8): each output block reads only its own block of input channels, so the kernel's in-channel
/// axis is C_in / groups.
/// </summary>
public class ConvolutionalLayerGroupsTests
{
    private const int Batch = 2, InChannels = 6, OutChannels = 4, Size = 5, Kernel = 3, Groups = 2, Padding = 1;

    private static (ConvolutionalLayer<double> Layer, double[] Parameters, Tensor<double> Input) Build()
    {
        var layer = new ConvolutionalLayer<double>(OutChannels, Kernel, stride: 1, padding: Padding,
            groups: Groups, biasMode: BiasMode.Always);
        var rng = RandomHelper.CreateSeededRandom(11);
        var input = new Tensor<double>(new[] { Batch, InChannels, Size, Size });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;
        layer.Forward(input);
        var parameters = new double[OutChannels * (InChannels / Groups) * Kernel * Kernel + OutChannels];
        for (int i = 0; i < parameters.Length; i++) parameters[i] = rng.NextDouble() - 0.5;
        Assert.Equal(parameters.Length, layer.ParameterCount);
        layer.SetParameters(new Vector<double>(parameters));
        return (layer, parameters, input);
    }

    [Fact]
    public void GroupedForward_MatchesTheDirectDefinition()
    {
        var (layer, parameters, input) = Build();
        var output = layer.Forward(input);
        Assert.Equal(new[] { Batch, OutChannels, Size, Size }, output.Shape.ToArray());

        int inPerGroup = InChannels / Groups, outPerGroup = OutChannels / Groups;
        int biasOffset = OutChannels * inPerGroup * Kernel * Kernel;
        for (int b = 0; b < Batch; b++)
        for (int o = 0; o < OutChannels; o++)
        for (int y = 0; y < Size; y++)
        for (int x = 0; x < Size; x++)
        {
            double expected = parameters[biasOffset + o];
            int group = o / outPerGroup;
            for (int c = 0; c < inPerGroup; c++)
            for (int ky = 0; ky < Kernel; ky++)
            for (int kx = 0; kx < Kernel; kx++)
            {
                int sy = y + ky - Padding, sx = x + kx - Padding;
                if (sy < 0 || sy >= Size || sx < 0 || sx >= Size) continue;
                double weight = parameters[((o * inPerGroup + c) * Kernel + ky) * Kernel + kx];
                expected += weight * input[((b * InChannels + group * inPerGroup + c) * Size + sy) * Size + sx];
            }

            Assert.Equal(expected, output[((b * OutChannels + o) * Size + y) * Size + x], 10);
        }
    }

    [Fact]
    public void GroupedForward_IsDifferentiableThroughTheKernels()
    {
        var (layer, _, input) = Build();
        var kernels = layer.KernelsForFolding;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var output = layer.Forward(input);
            var loss = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.ReduceSum(output, new[] { 0, 1, 2, 3 }, false);
            grads = tape.ComputeGradients(loss, sources: new[] { kernels });
        }

        // d(sum)/dW[o, c, ky, kx] is the sum of the input channel the kernel reads, over the window it covers: the
        // gradient of an output block's kernels must come from its own group's input channels only.
        Assert.True(grads.TryGetValue(kernels, out var g));
        int inPerGroup = InChannels / Groups, outPerGroup = OutChannels / Groups;
        for (int o = 0; o < OutChannels; o++)
        for (int c = 0; c < inPerGroup; c++)
        {
            int channel = (o / outPerGroup) * inPerGroup + c;
            double expected = 0;
            for (int b = 0; b < Batch; b++)
            for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                // Centre tap (ky = kx = 1) reads every pixel exactly once.
                expected += input[((b * InChannels + channel) * Size + y) * Size + x];
            }
            Assert.Equal(expected, g[((o * inPerGroup + c) * Kernel + 1) * Kernel + 1], 10);
        }
    }
}
