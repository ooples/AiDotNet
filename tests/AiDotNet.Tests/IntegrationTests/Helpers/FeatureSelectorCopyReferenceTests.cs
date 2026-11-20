using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.FeatureSelectors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

[Collection("EngineCurrentGlobalState")]
public sealed class FeatureSelectorCopyReferenceTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CopyFeature_PreservesTrailingValuesAndUntouchedDestinations(bool rankFour)
        => OnCpu(() =>
        {
            const int samples = 3, sourceFeatures = 4, destinationFeatures = 3;
            int[] sourceShape = rankFour ? new[] { samples, sourceFeatures, 2, 5 } : new[] { samples, sourceFeatures, 5 };
            int[] destinationShape = rankFour ? new[] { samples, destinationFeatures, 2, 5 } : new[] { samples, destinationFeatures, 5 };
            int trailing = rankFour ? 10 : 5;
            var sourceValues = Enumerable.Range(0, samples * sourceFeatures * trailing)
                .Select(index => 1000.0 + index * 7).ToArray();
            var source = new Tensor<double>((double[])sourceValues.Clone(), sourceShape);
            var destinationValues = Enumerable.Repeat(-123.0, samples * destinationFeatures * trailing).ToArray();
            var destination = new Tensor<double>(destinationValues, destinationShape);
            // Both feature indices and the sample index are nonzero. Compare every element,
            // not just the copied block: adjacent features and other samples must survive.
            const int sample = 2, sourceFeature = 3, destinationFeature = 1;
            FeatureSelectorHelper<double, Tensor<double>>.CopyFeature(source, destination, sample, sourceFeature, destinationFeature);
            var actual = destination.ToArray();
            for (int index = 0; index < actual.Length; index++)
            {
                int sampleAt = index / (destinationFeatures * trailing);
                int featureAt = index / trailing % destinationFeatures;
                double expected = sampleAt == sample && featureAt == destinationFeature
                    ? sourceValues[(sample * sourceFeatures + sourceFeature) * trailing + index % trailing]
                    : -123.0;
                Assert.Equal(expected, actual[index]);
            }
            Assert.Equal(sourceValues, source.ToArray());
        });

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FilteredTensor_ReordersAllFeatureValuesForEverySample(bool rankFour)
        => OnCpu(() =>
        {
            const int samples = 3, features = 4;
            int[] shape = rankFour ? new[] { samples, features, 2, 5 } : new[] { samples, features, 5 };
            int trailing = rankFour ? 10 : 5;
            var values = Enumerable.Range(0, samples * features * trailing).Select(index => index + .25).ToArray();
            var source = new Tensor<double>((double[])values.Clone(), shape);
            var selected = new List<int> { 3, 1 };
            var result = FeatureSelectorHelper<double, Tensor<double>>.CreateFilteredData(source, selected);
            int[] expectedShape = rankFour ? new[] { samples, selected.Count, 2, 5 } : new[] { samples, selected.Count, 5 };
            Assert.Equal(expectedShape, result.Shape.ToArray());
            var actual = result.ToArray();
            for (int sample = 0; sample < samples; sample++)
                for (int feature = 0; feature < selected.Count; feature++)
                    for (int element = 0; element < trailing; element++)
                        Assert.Equal(values[(sample * features + selected[feature]) * trailing + element],
                            actual[(sample * selected.Count + feature) * trailing + element]);
            Assert.Equal(values, source.ToArray());
        });

    private static void OnCpu(Action action)
    {
        var previous = AiDotNetEngine.Current;
        try { AiDotNetEngine.Current = new CpuEngine(); action(); }
        finally { AiDotNetEngine.Current = previous; }
    }
}
