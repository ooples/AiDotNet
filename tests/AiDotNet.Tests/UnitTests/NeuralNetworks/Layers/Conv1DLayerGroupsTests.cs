using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Grouped 1-D convolution (PyTorch nn.Conv1d <c>groups</c>), which Wav2Vec2's convolutional
/// positional embedding needs (kernel 128, 16 groups). Each output block may read only its own block
/// of input channels, so the kernel's in-channel axis is C_in / groups.
/// </summary>
public class Conv1DLayerGroupsTests
{
    private const int Batch = 2, InChannels = 4, OutChannels = 6, Length = 9, Kernel = 3, Groups = 2;
    private const int Stride = 2, Padding = 1, Dilation = 2;

    [Fact]
    public void GroupedForward_MatchesTheDirectDefinition()
    {
        var layer = new Conv1DLayer<double>(
            InChannels, OutChannels, Kernel, Dilation, Stride, Padding,
            (IActivationFunction<double>)new IdentityActivation<double>(), groups: Groups);
        var rng = RandomHelper.CreateSeededRandom(7);
        var parameters = new double[OutChannels * (InChannels / Groups) * Kernel + OutChannels];
        for (int i = 0; i < parameters.Length; i++) parameters[i] = rng.NextDouble() - 0.5;
        Assert.Equal(parameters.Length, layer.ParameterCount);
        layer.SetParameters(new Vector<double>(parameters));

        var input = new Tensor<double>(new[] { Batch, InChannels, Length });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;

        var output = layer.Forward(input);

        int outLength = (Length + 2 * Padding - Dilation * (Kernel - 1) - 1) / Stride + 1;
        Assert.Equal(new[] { Batch, OutChannels, outLength }, output.Shape.ToArray());

        int inPerGroup = InChannels / Groups, outPerGroup = OutChannels / Groups;
        int biasOffset = OutChannels * inPerGroup * Kernel;
        for (int b = 0; b < Batch; b++)
        for (int o = 0; o < OutChannels; o++)
        for (int t = 0; t < outLength; t++)
        {
            double expected = parameters[biasOffset + o];
            int group = o / outPerGroup;
            for (int c = 0; c < inPerGroup; c++)
            for (int k = 0; k < Kernel; k++)
            {
                int position = t * Stride + k * Dilation - Padding;
                if (position < 0 || position >= Length) continue;
                double weight = parameters[(o * inPerGroup + c) * Kernel + k];
                expected += weight * input[(b * InChannels + group * inPerGroup + c) * Length + position];
            }

            Assert.Equal(expected, output[(b * OutChannels + o) * outLength + t], 10);
        }
    }

    [Fact]
    public void GroupedLayer_HasTheBlockedParameterCount()
    {
        var dense = new Conv1DLayer<double>(InChannels, OutChannels, kernelSize: Kernel);
        var grouped = new Conv1DLayer<double>(InChannels, OutChannels, kernelSize: Kernel, groups: Groups);

        Assert.Equal(OutChannels * InChannels * Kernel + OutChannels, dense.ParameterCount);
        Assert.Equal(OutChannels * (InChannels / Groups) * Kernel + OutChannels, grouped.ParameterCount);
    }

    [Fact]
    public void LazyGroupedLayer_ResolvesTheBlockedKernelFromItsInput()
    {
        var layer = new Conv1DLayer<double>(OutChannels, Kernel, groups: Groups);
        layer.Forward(new Tensor<double>(new[] { 1, InChannels, Length }));

        Assert.Equal(OutChannels * (InChannels / Groups) * Kernel + OutChannels, layer.ParameterCount);
    }

    [Fact]
    public void Clone_PreservesGroupsAndOutput()
    {
        var layer = new Conv1DLayer<double>(InChannels, OutChannels, kernelSize: Kernel, groups: Groups);
        var input = new Tensor<double>(new[] { 1, InChannels, Length });
        var rng = RandomHelper.CreateSeededRandom(11);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
        var expected = layer.Forward(input);

        var clone = (Conv1DLayer<double>)((LayerBase<double>)layer).Clone();
        var actual = clone.Forward(input);

        Assert.Equal(layer.ParameterCount, clone.ParameterCount);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 12);
    }

    [Fact]
    public void Deserialize_RestoresGroupsFromMetadata_ForALazyLayer()
    {
        // The lazy constructor carries no [LayerState] for groups, so a saved model can only recover
        // them from the "Groups" metadata entry that serialization writes and DeserializationHelper reads.
        int blockedCount = OutChannels * (InChannels / Groups) * Kernel + OutChannels;
        var unresolved = new Conv1DLayer<double>(OutChannels, Kernel, groups: Groups);
        var rebuiltLazy = Rebuild(unresolved);
        rebuiltLazy.Forward(new Tensor<double>(new[] { 1, InChannels, Length }));
        Assert.Equal(blockedCount, rebuiltLazy.ParameterCount);

        // A resolved layer round-trips its weights too, and must compute the same grouped convolution.
        var layer = new Conv1DLayer<double>(OutChannels, Kernel, groups: Groups);
        var input = new Tensor<double>(new[] { 1, InChannels, Length });
        var rng = RandomHelper.CreateSeededRandom(13);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();
        var expected = layer.Forward(input);

        var restored = Rebuild(layer);
        restored.SetParameters(layer.GetParameters());
        var actual = restored.Forward(input);

        Assert.Equal(blockedCount, restored.ParameterCount);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 12);
    }

    private static ILayer<double> Rebuild(Conv1DLayer<double> source)
    {
        var metadata = source.GetMetadata().ToDictionary(entry => entry.Key, entry => (object)entry.Value);
        Assert.Equal(Groups.ToString(), metadata["Groups"]);
        return DeserializationHelper.CreateLayerFromType<double>(
            source.GetType().Name, source.GetInputShape(), source.GetOutputShape(), metadata);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void Groups_ThatDoNotDivideTheChannels_AreRejected(int groups)
    {
        // 4 input and 6 output channels: 3 does not divide 4, and 4 does not divide 6.
        Assert.Throws<ArgumentException>(() => new Conv1DLayer<double>(InChannels, OutChannels, kernelSize: Kernel, groups: groups));
    }
}