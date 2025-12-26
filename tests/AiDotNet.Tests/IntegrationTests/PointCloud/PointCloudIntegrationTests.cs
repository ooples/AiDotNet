using System;
using AiDotNet.Models.Options;
using AiDotNet.PointCloud.Models;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.PointCloud;

/// <summary>
/// Integration tests for point cloud neural network models.
/// </summary>
public class PointCloudIntegrationTests
{
    [Fact]
    public void PointNet_Train_ProducesFiniteOutput()
    {
        var options = new PointNetOptions
        {
            NumClasses = 4,
            InputFeatureDim = 3,
            InputTransformDim = 3,
            InputMlpChannels = new[] { 16, 32 },
            FeatureMlpChannels = new[] { 32, 64 },
            ClassifierChannels = new[] { 32 },
            UseDropout = false
        };
        var model = new PointNet<double>(options);

        var input = CreatePointCloud(numPoints: 16, featureDim: 3, seed: 7);
        var expected = CreateOneHotExpected(numClasses: options.NumClasses, classIndex: 1);

        model.Train(input, expected);
        var output = model.Predict(input);

        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(options.NumClasses, output.Shape[1]);
        AssertAllFinite(output);
    }

    [Fact]
    public void PointNetPlusPlus_Train_ProducesFiniteOutput()
    {
        var options = new PointNetPlusPlusOptions
        {
            NumClasses = 4,
            InputFeatureDim = 3,
            SamplingRates = new[] { 16, 8 },
            SearchRadii = new[] { 0.2, 0.4 },
            NeighborSamples = new[] { 8, 8 },
            MlpDimensions = new[]
            {
                new[] { 16, 32 },
                new[] { 32, 64 }
            },
            ClassifierChannels = new[] { 64 },
            UseDropout = false
        };
        var model = new PointNetPlusPlus<double>(options);

        var input = CreatePointCloud(numPoints: 32, featureDim: 3, seed: 11);
        var expected = CreateOneHotExpected(numClasses: options.NumClasses, classIndex: 2);

        model.Train(input, expected);
        var output = model.Predict(input);

        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(options.NumClasses, output.Shape[1]);
        AssertAllFinite(output);
    }

    [Fact]
    public void DGCNN_Train_ProducesFiniteOutput()
    {
        var options = new DGCNNOptions
        {
            NumClasses = 4,
            InputFeatureDim = 3,
            KnnK = 8,
            EdgeConvChannels = new[] { 16, 32 },
            ClassifierChannels = new[] { 32 },
            UseDropout = false
        };
        var model = new DGCNN<double>(options);

        var input = CreatePointCloud(numPoints: 32, featureDim: 3, seed: 13);
        var expected = CreateOneHotExpected(numClasses: options.NumClasses, classIndex: 0);

        model.Train(input, expected);
        var output = model.Predict(input);

        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(options.NumClasses, output.Shape[1]);
        AssertAllFinite(output);
    }

    private static Tensor<double> CreatePointCloud(int numPoints, int featureDim, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var data = new double[numPoints * featureDim];

        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble() * 2.0 - 1.0;
        }

        return new Tensor<double>(data, new[] { numPoints, featureDim });
    }

    private static Tensor<double> CreateOneHotExpected(int numClasses, int classIndex)
    {
        if (classIndex < 0 || classIndex >= numClasses)
        {
            throw new ArgumentOutOfRangeException(nameof(classIndex));
        }

        var data = new double[numClasses];
        data[classIndex] = 1.0;
        return new Tensor<double>(data, new[] { 1, numClasses });
    }

    private static void AssertAllFinite(Tensor<double> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            double value = tensor.Data[i];
            Assert.False(double.IsNaN(value));
            Assert.False(double.IsInfinity(value));
        }
    }
}
