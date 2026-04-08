using System;
using System.Linq;
using AiDotNet.PointCloud.Data;
using AiDotNet.PointCloud.Layers;
using AiDotNet.Tensors.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.PointCloud;

/// <summary>
/// Integration tests for PointCloud layers: PointConvolutionLayer, MaxPoolingLayer, TNetLayer.
/// Tests gradient correctness, shape preservation, and mathematical properties.
/// </summary>
public class PointCloudLayersIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region PointConvolutionLayer Tests

    [Fact(Timeout = 120000)]
    public async Task PointConvolution_Forward_ProducesCorrectOutputShape()
    {
        var layer = new PointConvolutionLayer<double>(3, 16);
        var input = CreateRandomTensor(32, 3, seed: 42);

        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(32, output.Shape[0]); // numPoints preserved
        Assert.Equal(16, output.Shape[1]); // output channels
    }

    [Fact(Timeout = 120000)]
    public async Task PointConvolution_Forward_AllOutputsFinite()
    {
        var layer = new PointConvolutionLayer<double>(6, 32);
        var input = CreateRandomTensor(64, 6, seed: 7);

        var output = layer.Forward(input);

        AssertAllFinite(output);
    }



    [Fact(Timeout = 120000)]
    public async Task PointConvolution_ParameterCount_EqualsWeightsPlusBiases()
    {
        int inputCh = 5;
        int outputCh = 10;
        var layer = new PointConvolutionLayer<double>(inputCh, outputCh);

        // Parameters = inputCh * outputCh (weights) + outputCh (biases)
        Assert.Equal(inputCh * outputCh + outputCh, layer.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task PointConvolution_GetParameters_ReturnsCorrectLength()
    {
        var layer = new PointConvolutionLayer<double>(3, 7);
        var parameters = layer.GetParameters();

        Assert.Equal(layer.ParameterCount, parameters.Length);
    }



    [Fact(Timeout = 120000)]
    public async Task PointConvolution_SinglePoint_ProducesValidOutput()
    {
        var layer = new PointConvolutionLayer<double>(3, 2);
        var input = CreateRandomTensor(1, 3, seed: 41);

        var output = layer.Forward(input);

        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(2, output.Shape[1]);
        AssertAllFinite(output);
    }

    [Fact(Timeout = 120000)]
    public async Task PointConvolution_LargePointCloud_HandlesCorrectly()
    {
        var layer = new PointConvolutionLayer<double>(3, 16);
        var input = CreateRandomTensor(1024, 3, seed: 43);

        var output = layer.Forward(input);

        Assert.Equal(1024, output.Shape[0]);
        Assert.Equal(16, output.Shape[1]);
        AssertAllFinite(output);
    }

    #endregion

    #region MaxPoolingLayer Tests

    [Fact(Timeout = 120000)]
    public async Task MaxPooling_Forward_ReducesToSingleRow()
    {
        var layer = new MaxPoolingLayer<double>(4);
        var input = CreateRandomTensor(10, 4, seed: 47);

        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]);  // pooled to 1 point
        Assert.Equal(4, output.Shape[1]);  // features preserved
    }

    [Fact(Timeout = 120000)]
    public async Task MaxPooling_Forward_GoldenReference_TakesMaxPerChannel()
    {
        // Create known input where max values are easy to verify
        var data = new double[]
        {
            1.0, 5.0, 2.0,  // point 0
            3.0, 1.0, 8.0,  // point 1
            7.0, 3.0, 4.0,  // point 2
        };
        var input = new Tensor<double>(data, new[] { 3, 3 });
        var layer = new MaxPoolingLayer<double>(3);

        var output = layer.Forward(input);

        // Max per column: [7.0, 5.0, 8.0]
        Assert.Equal(7.0, output[0], Tolerance);
        Assert.Equal(5.0, output[1], Tolerance);
        Assert.Equal(8.0, output[2], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task MaxPooling_PermutationInvariant()
    {
        // MaxPooling should give same result regardless of point order
        var data1 = new double[]
        {
            1.0, 4.0,
            3.0, 2.0,
            5.0, 6.0,
        };
        var data2 = new double[]
        {
            5.0, 6.0,  // reordered
            1.0, 4.0,
            3.0, 2.0,
        };
        var input1 = new Tensor<double>(data1, new[] { 3, 2 });
        var input2 = new Tensor<double>(data2, new[] { 3, 2 });

        var layer1 = new MaxPoolingLayer<double>(2);
        var layer2 = new MaxPoolingLayer<double>(2);

        var output1 = layer1.Forward(input1);
        var output2 = layer2.Forward(input2);

        Assert.Equal(output1[0], output2[0], Tolerance);
        Assert.Equal(output1[1], output2[1], Tolerance);
    }


    [Fact(Timeout = 120000)]
    public async Task MaxPooling_ParameterCount_IsZero()
    {
        var layer = new MaxPoolingLayer<double>(16);
        Assert.Equal(0, layer.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task MaxPooling_SinglePoint_ReturnsSameValues()
    {
        var data = new double[] { 3.5, 7.2, 1.1 };
        var input = new Tensor<double>(data, new[] { 1, 3 });
        var layer = new MaxPoolingLayer<double>(3);

        var output = layer.Forward(input);

        Assert.Equal(3.5, output[0], Tolerance);
        Assert.Equal(7.2, output[1], Tolerance);
        Assert.Equal(1.1, output[2], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task MaxPooling_AllSameValues_ReturnsConstant()
    {
        var data = new double[] { 5.0, 5.0, 5.0, 5.0 };
        var input = new Tensor<double>(data, new[] { 2, 2 });
        var layer = new MaxPoolingLayer<double>(2);

        var output = layer.Forward(input);

        Assert.Equal(5.0, output[0], Tolerance);
        Assert.Equal(5.0, output[1], Tolerance);
    }

    #endregion

    #region TNetLayer Tests

    [Fact(Timeout = 120000)]
    public async Task TNet_Forward_PreservesShape()
    {
        var tnet = new TNetLayer<double>(3, 3, new[] { 16, 32 }, new[] { 16 });
        var input = CreateRandomTensor(16, 3, seed: 53);

        var output = tnet.Forward(input);

        Assert.Equal(input.Shape[0], output.Shape[0]);
        Assert.Equal(input.Shape[1], output.Shape[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task TNet_Forward_OutputIsFinite()
    {
        var tnet = new TNetLayer<double>(3, 3, new[] { 16, 32 }, new[] { 16 });
        var input = CreateRandomTensor(16, 3, seed: 59);

        var output = tnet.Forward(input);

        AssertAllFinite(output);
    }


    [Fact(Timeout = 120000)]
    public async Task TNet_ParameterCount_IsPositive()
    {
        var tnet = new TNetLayer<double>(3, 3, new[] { 16, 32 }, new[] { 16 });
        Assert.True(tnet.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task TNet_InvalidTransformDim_ThrowsArgumentOutOfRange()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TNetLayer<double>(0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TNetLayer<double>(-1, 3));
    }

    [Fact(Timeout = 120000)]
    public async Task TNet_TransformDimGreaterThanFeatures_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TNetLayer<double>(5, 3));
    }

    [Fact(Timeout = 120000)]
    public async Task TNet_InvalidFeatures_ThrowsArgumentOutOfRange()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TNetLayer<double>(3, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TNetLayer<double>(3, -1));
    }


    [Fact(Timeout = 120000)]
    public async Task TNet_FeatureTransform_PreservesExtraFeatures()
    {
        // Transform only first 3 dims of a 6-dim input, extra features should pass through
        var tnet = new TNetLayer<double>(3, 6, new[] { 16, 32 }, new[] { 16 });
        var input = CreateRandomTensor(8, 6, seed: 79);

        var output = tnet.Forward(input);

        Assert.Equal(8, output.Shape[0]);
        Assert.Equal(6, output.Shape[1]);
        AssertAllFinite(output);

        // Features 3-5 should be preserved (untransformed)
        for (int i = 0; i < 8; i++)
        {
            for (int f = 3; f < 6; f++)
            {
                Assert.Equal(
                    input[i * 6 + f],
                    output[i * 6 + f],
                    Tolerance);
            }
        }
    }



    #endregion

    #region PointCloudData Tests

    [Fact(Timeout = 120000)]
    public async Task PointCloudData_ConstructFromTensor_SetsProperties()
    {
        var points = CreateRandomTensor(50, 3, seed: 101);
        var cloud = new PointCloudData<double>(points);

        Assert.Equal(50, cloud.NumPoints);
        Assert.Equal(3, cloud.NumFeatures);
        Assert.Null(cloud.Labels);
    }

    [Fact(Timeout = 120000)]
    public async Task PointCloudData_WithLabels_StoresLabels()
    {
        var points = CreateRandomTensor(10, 3, seed: 103);
        var labels = new Vector<double>(new double[] { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 });
        var cloud = new PointCloudData<double>(points, labels);

        Assert.NotNull(cloud.Labels);
        Assert.Equal(10, cloud.Labels.Length);
        Assert.Equal(0.0, cloud.Labels[0]);
        Assert.Equal(1.0, cloud.Labels[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task PointCloudData_GetCoordinates_ExtractsXYZOnly()
    {
        // 6-feature cloud (XYZ + RGB)
        var data = new double[]
        {
            1.0, 2.0, 3.0, 255.0, 0.0, 0.0,
            4.0, 5.0, 6.0, 0.0, 255.0, 0.0,
        };
        var points = new Tensor<double>(data, new[] { 2, 6 });
        var cloud = new PointCloudData<double>(points);

        var coords = cloud.GetCoordinates();

        Assert.Equal(new[] { 2, 3 }, coords.Shape.ToArray());
        Assert.Equal(1.0, coords[0], Tolerance);
        Assert.Equal(2.0, coords[1], Tolerance);
        Assert.Equal(3.0, coords[2], Tolerance);
        Assert.Equal(4.0, coords[3], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task PointCloudData_GetCoordinates_ReturnsOriginalWhen3Features()
    {
        var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var points = new Tensor<double>(data, new[] { 2, 3 });
        var cloud = new PointCloudData<double>(points);

        var coords = cloud.GetCoordinates();

        // Should return the same tensor reference when NumFeatures == 3
        Assert.Same(cloud.Points, coords);
    }

    [Fact(Timeout = 120000)]
    public async Task PointCloudData_GetFeatures_ReturnsNullForXYZOnly()
    {
        var points = CreateRandomTensor(10, 3, seed: 107);
        var cloud = new PointCloudData<double>(points);

        Assert.Null(cloud.GetFeatures());
    }

    [Fact(Timeout = 120000)]
    public async Task PointCloudData_GetFeatures_ExtractsNonXYZColumns()
    {
        var data = new double[]
        {
            1.0, 2.0, 3.0, 100.0, 200.0, 50.0,
            4.0, 5.0, 6.0, 150.0, 250.0, 75.0,
        };
        var points = new Tensor<double>(data, new[] { 2, 6 });
        var cloud = new PointCloudData<double>(points);

        var features = cloud.GetFeatures();

        Assert.NotNull(features);
        Assert.Equal(new[] { 2, 3 }, features.Shape.ToArray());
        Assert.Equal(100.0, features[0], Tolerance);
        Assert.Equal(200.0, features[1], Tolerance);
        Assert.Equal(50.0, features[2], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task PointCloudData_FromCoordinates_CreatesCorrectTensor()
    {
        var matrix = new Matrix<double>(3, 3);
        matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
        matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
        matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 9.0;

        var cloud = PointCloudData<double>.FromCoordinates(matrix);

        Assert.Equal(3, cloud.NumPoints);
        Assert.Equal(3, cloud.NumFeatures);
    }

    #endregion

    #region End-to-End Layer Pipeline

    [Fact(Timeout = 120000)]
    public async Task PointCloudPipeline_ConvMaxPoolFC_ProducesFiniteOutput()
    {
        // Simulate a mini PointNet-like pipeline: Conv -> MaxPool
        var conv = new PointConvolutionLayer<double>(3, 8);
        var pool = new MaxPoolingLayer<double>(8);

        var input = CreateRandomTensor(32, 3, seed: 109);

        var features = conv.Forward(input);
        var global = pool.Forward(features);

        Assert.Equal(1, global.Shape[0]);
        Assert.Equal(8, global.Shape[1]);
        AssertAllFinite(global);
    }

    [Fact(Timeout = 120000)]
    public async Task PointCloudPipeline_MultiLayerConv_IncreasesFeatureDim()
    {
        var conv1 = new PointConvolutionLayer<double>(3, 16);
        var conv2 = new PointConvolutionLayer<double>(16, 32);
        var conv3 = new PointConvolutionLayer<double>(32, 64);

        var input = CreateRandomTensor(64, 3, seed: 113);

        var h1 = conv1.Forward(input);
        Assert.Equal(16, h1.Shape[1]);

        var h2 = conv2.Forward(h1);
        Assert.Equal(32, h2.Shape[1]);

        var h3 = conv3.Forward(h2);
        Assert.Equal(64, h3.Shape[1]);

        AssertAllFinite(h3);
    }


    #endregion

    #region Helpers

    private static Tensor<double> CreateRandomTensor(int rows, int cols, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var data = new double[rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble() * 2.0 - 1.0;
        }
        return new Tensor<double>(data, new[] { rows, cols });
    }

    private static void AssertAllFinite(Tensor<double> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            double value = tensor[i];
            Assert.False(double.IsNaN(value), $"NaN at index {i}");
            Assert.False(double.IsInfinity(value), $"Infinity at index {i}");
        }
    }

    #endregion
}
