using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

[Collection("NonParallelIntegration")]
public sealed class UnmatchedGradientPublicationTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void InactiveLayerWithdrawsPublicationWithoutAllocatingAParameterSizedArray(int ownershipValue)
    {
        var previousEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            using var layer = new NormedConv1DLayer<float>(128, 128, 3, 1, 1, 1, 1, false,
                ConvolutionNormalization.Weight);
            var parameters = layer.GetTrainableParameters();
            Assert.True(parameters.Sum(parameter => parameter.Length) * sizeof(float) > 192 * 1024);
            var matching = parameters.ToDictionary(parameter => parameter,
                parameter => new Tensor<float>(Enumerable.Repeat(3f, parameter.Length).ToArray(), parameter.Shape.ToArray()));
            Assert.True(layer.ScatterParameterGradients(matching, GradientSourceOwnership.Plan) > 0);
            var published = layer.GetParameterGradients();
            var snapshot = published.ToArray();
            Assert.Contains(snapshot, value => value == 3f);

            var unrelated = new Tensor<float>(new[] { 1f }, new[] { 1 });
            var otherGroup = new Dictionary<Tensor<float>, Tensor<float>>
            {
                [unrelated] = new Tensor<float>(new[] { 7f }, new[] { 1 })
            };
            var ownership = ownershipValue == 0 ? GradientSourceOwnership.Caller
                : ownershipValue == 1 ? GradientSourceOwnership.Tape : GradientSourceOwnership.Plan;
            Assert.Equal(0, layer.ScatterParameterGradients(otherGroup, ownership)); // Warm metadata/JIT paths.
            Assert.Empty(layer.GetParameterGradients().ToArray());
#if NET5_0_OR_GREATER
            long before = GC.GetAllocatedBytesForCurrentThread();
#endif
            for (int repeat = 0; repeat < 4; repeat++) layer.ScatterParameterGradients(otherGroup, ownership);
#if NET5_0_OR_GREATER
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.True(allocated < 64 * 1024,
                $"An inactive layer allocated {allocated} bytes for four metadata-only publications.");
#endif
            Assert.Null(layer.ScatteredParameterGradients);
            Assert.Equal(snapshot, published.ToArray());
        }
        finally { AiDotNetEngine.Current = previousEngine; }
    }

    [Fact]
    public void MetadataPreflightRejectsWrongLengthsAndKeepsMatchedPlanGradientsLazy()
    {
        var previousEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            using var layer = new NormedConv1DLayer<float>(2, 3, 3, 1, 1, 1, 1, false,
                ConvolutionNormalization.Weight);
            var parameter = layer.GetTrainableParameters()[0];
            var wrongLength = new Dictionary<Tensor<float>, Tensor<float>>
            {
                [parameter] = new Tensor<float>(new[] { 9f }, new[] { 1 })
            };
            Assert.Equal(0, layer.ScatterParameterGradients(wrongLength, GradientSourceOwnership.Plan));
            Assert.Null(layer.ScatteredParameterGradients);
            wrongLength[parameter] = Tensor<float>.Empty();
            Assert.Equal(0, layer.ScatterParameterGradients(wrongLength, GradientSourceOwnership.Plan));
            Assert.Null(layer.ScatteredParameterGradients);

            var gradient = new Tensor<float>(Enumerable.Repeat(2f, parameter.Length).ToArray(), parameter.Shape.ToArray());
            var alias = gradient.Reshape(new[] { gradient.Length });
            var matching = new Dictionary<Tensor<float>, Tensor<float>> { [parameter] = alias };
            Assert.Equal(parameter.Length, layer.ScatterParameterGradients(matching, GradientSourceOwnership.Plan));
            gradient.AsWritableSpan().Fill(5f);
            var read = layer.GetParameterGradients();
            var snapshot = read.ToArray();
            Assert.Equal(parameter.Length, snapshot.Count(value => value == 5f));
            Assert.All(snapshot, value => Assert.True(value == 0f || value == 5f));
            gradient.AsWritableSpan().Fill(7f);
            Assert.Equal(snapshot, read.ToArray());
            Assert.Equal(parameter.Length, layer.ScatterParameterGradients(matching, GradientSourceOwnership.Plan));
            Assert.Equal(parameter.Length, layer.GetParameterGradients().ToArray().Count(value => value == 7f));
        }
        finally { AiDotNetEngine.Current = previousEngine; }
    }
}
