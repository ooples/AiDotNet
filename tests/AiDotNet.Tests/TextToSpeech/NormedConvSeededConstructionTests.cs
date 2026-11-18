using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class NormedConvSeededConstructionTests
{
    [Fact]
    public void SeededInternalInitializationPreservesThePublicConstructorSignature()
    {
        var constructor = Assert.Single(typeof(NormedConv1DLayer<double>).GetConstructors());
        var parameters = constructor.GetParameters();
        Assert.Equal(new[] { "inChannels", "outChannels", "kernelSize", "stride", "dilation", "groups", "padding",
            "transposed", "normalization", "useBias", "outputPadding" }, parameters.Select(p => p.Name));
        Assert.Equal(new[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int), typeof(int),
            typeof(bool), typeof(ConvolutionNormalization), typeof(bool), typeof(int) }, parameters.Select(p => p.ParameterType));
        Assert.All(parameters.Take(9), p => Assert.False(p.IsOptional));
        Assert.Equal(true, parameters[9].DefaultValue);
        Assert.Equal(0, parameters[10].DefaultValue);
    }

    [Theory]
    [InlineData(ConvolutionNormalization.Weight)]
    [InlineData(ConvolutionNormalization.Spectral)]
    public void SeededLayerFullClonePreservesParametersBuffersAndConstructionState(ConvolutionNormalization normalization)
    {
        using var source = new NormedConv1DLayer<double>(4, 6, 3, 2, 2, 2, 2, false,
            normalization, outputPadding: 0, useBias: true, initialization: new Random(1234));
        source.SetTrainingMode(false);
        using var full = (NormedConv1DLayer<double>)source.Clone(CloneOptions.Full);
        using var architecture = (NormedConv1DLayer<double>)source.Clone(CloneOptions.Architecture);
        foreach (string name in new[] { "_in", "_out", "_kernel", "_stride", "_padding", "_dilation", "_groups",
                     "_transposed", "_normalization", "_useBias", "_outputPadding" })
        {
            var field = typeof(NormedConv1DLayer<double>).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!;
            Assert.Equal(field.GetValue(source), field.GetValue(full));
            Assert.Equal(field.GetValue(source), field.GetValue(architecture));
        }
        var before = source.GetParameters().ToArray();
        Assert.Equal(before, full.GetParameters().ToArray());
        foreach (string name in new[] { "_u", "_v" })
        {
            var field = typeof(NormedConv1DLayer<double>).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!;
            Assert.Equal(((Tensor<double>)field.GetValue(source)!).ToArray(), ((Tensor<double>)field.GetValue(full)!).ToArray());
        }
        full.SetTrainingMode(false);
        using var input = new Tensor<double>(Enumerable.Range(0, 44).Select(i => Math.Sin(.13 * i)).ToArray(), new[] { 1, 4, 11 });
        Assert.Equal(source.Forward(input).ToArray(), full.Forward(input).ToArray());
        var changed = full.GetParameters();
        for (int i = 0; i < changed.Length; i++) changed[i] += .125;
        full.UpdateParameters(changed);
        Assert.Equal(before, source.GetParameters().ToArray());
        Assert.NotEqual(before, full.GetParameters().ToArray());
    }
}
