using System;
using System.Linq;
using AiDotNet.PointCloud.Data;
using AiDotNet.PointCloud.Layers;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.PointCloud;

/// <summary>
/// Integration tests for PointCloud layers: PointConvolutionLayer, MaxPoolingLayer, TNetLayer.
/// Tests gradient correctness, shape preservation, and mathematical properties.
/// </summary>
public class PointCloudLayersIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region PointConvolutionLayer Tests

    [Fact]
    public void PointConvolution_Forward_ProducesCorrectOutputShape()
    {
        var layer = new PointConvolutionLayer<double>(3, 16);
        var input = CreateRandomTensor(32, 3, seed: 42);

        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(32, output.Shape[0]); // numPoints preserved
        Assert.Equal(16, output.Shape[1]); // output channels
    }

    [Fact]
    public void PointConvolution_Forward_AllOutputsFinite()
    {
        var layer = new PointConvolutionLayer<double>(6, 32);
        var input = CreateRandomTensor(64, 6, seed: 7);

        var output = layer.Forward(input);

        AssertAllFinite(output);
    }

    [Fact]
    public void PointConvolution_Backward_ProducesCorrectGradientShape()
    {
        var layer = new PointConvolutionLayer<double>(3, 8);
        var input = CreateRandomTensor(16, 3, seed: 11);
        layer.Forward(input);

        var outputGrad = CreateRandomTensor(16, 8, seed: 13);
        var inputGrad = layer.Backward(outputGrad);

        Assert.Equal(2, inputGrad.Shape.Length);
        Assert.Equal(16, inputGrad.Shape[0]); // same as input numPoints
        Assert.Equal(3, inputGrad.Shape[1]);  // same as input channels
    }

    [Fact]
    public void PointConvolution_Backward_GradientsAreFinite()
    {
        var layer = new PointConvolutionLayer<double>(4, 12);
        var input = CreateRandomTensor(20, 4, seed: 17);
        layer.Forward(input);

        var outputGrad = CreateRandomTensor(20, 12, seed: 19);
        var inputGrad = layer.Backward(outputGrad);

        AssertAllFinite(inputGrad);
    }

    [Fact]
    public void PointConvolution_ParameterCount_EqualsWeightsPlusBiases()
    {
        int inputCh = 5;
        int outputCh = 10;
        var layer = new PointConvolutionLayer<double>(inputCh, outputCh);

        // Parameters = inputCh * outputCh (weights) + outputCh (biases)
        Assert.Equal(inputCh * outputCh + outputCh, layer.ParameterCount);
    }

    [Fact]
    public void PointConvolution_GetParameters_ReturnsCorrectLength()
    {
        var layer = new PointConvolutionLayer<double>(3, 7);
        var parameters = layer.GetParameters();

        Assert.Equal(layer.ParameterCount, parameters.Length);
    }

    [Fact]
    public void PointConvolution_UpdateParameters_ChangesOutput()
    {
        var layer = new PointConvolutionLayer<double>(3, 4);
        var input = CreateRandomTensor(8, 3, seed: 23);

        var output1 = layer.Forward(input);
        var val1 = output1[0];

        // Update with learning rate
        layer.Forward(input);
        var grad = CreateRandomTensor(8, 4, seed: 29);
        layer.Backward(grad);
        layer.UpdateParameters(0.01);

        var output2 = layer.Forward(input);
        var val2 = output2[0];

        // Output should change after parameter update
        Assert.NotEqual(val1, val2);
    }

    [Fact]
    public void PointConvolution_ClearGradients_ResetsAccumulation()
    {
        var layer = new PointConvolutionLayer<double>(3, 4);
        var input = CreateRandomTensor(8, 3, seed: 31);
        layer.Forward(input);

        var grad = CreateRandomTensor(8, 4, seed: 37);
        layer.Backward(grad);

        // Clear and do another forward/backward
        layer.ClearGradients();
        layer.Forward(input);
        layer.Backward(grad);

        // Should not throw and should produce valid gradients
        var inputGrad = layer.Backward(grad);
        AssertAllFinite(inputGrad);
    }

    [Fact]
    public void PointConvolution_SinglePoint_ProducesValidOutput()
    {
        var layer = new PointConvolutionLayer<double>(3, 2);
        var input = CreateRandomTensor(1, 3, seed: 41);

        var output = layer.Forward(input);

        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(2, output.Shape[1]);
        AssertAllFinite(output);
    }

    [Fact]
    public void PointConvolution_LargePointCloud_HandlesCorrectly()
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

    [Fact]
    public void MaxPooling_Forward_ReducesToSingleRow()
    {
        var layer = new MaxPoolingLayer<double>(4);
        var input = CreateRandomTensor(10, 4, seed: 47);

        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]);  // pooled to 1 point
        Assert.Equal(4, output.Shape[1]);  // features preserved
    }

    [Fact]
    public void MaxPooling_Forward_GoldenReference_TakesMaxPerChannel()
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

    [Fact]
    public void MaxPooling_PermutationInvariant()
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

    [Fact]
    public void MaxPooling_Backward_RoutesGradientToMaxElements()
    {
        var data = new double[]
        {
            1.0, 5.0,
            3.0, 2.0,
        };
        var input = new Tensor<double>(data, new[] { 2, 2 });
        var layer = new MaxPoolingLayer<double>(2);
        layer.Forward(input);

        var outGrad = new Tensor<double>(new[] { 10.0, 20.0 }, new[] { 1, 2 });
        var inputGrad = layer.Backward(outGrad);

        Assert.Equal(2, inputGrad.Shape[0]);
        Assert.Equal(2, inputGrad.Shape[1]);
        AssertAllFinite(inputGrad);

        // Gradient should flow to max elements:
        // Channel 0: max at index 1 (value 3.0), so grad[0,0]=0, grad[1,0]=10
        // Channel 1: max at index 0 (value 5.0), so grad[0,1]=20, grad[1,1]=0
        Assert.Equal(0.0, inputGrad[0], Tolerance);  // point 0, ch 0 (not max)
        Assert.Equal(20.0, inputGrad[1], Tolerance);  // point 0, ch 1 (max)
        Assert.Equal(10.0, inputGrad[2], Tolerance);  // point 1, ch 0 (max)
        Assert.Equal(0.0, inputGrad[3], Tolerance);   // point 1, ch 1 (not max)
    }

    [Fact]
    public void MaxPooling_ParameterCount_IsZero()
    {
        var layer = new MaxPoolingLayer<double>(16);
        Assert.Equal(0, layer.ParameterCount);
    }

    [Fact]
    public void MaxPooling_SinglePoint_ReturnsSameValues()
    {
        var data = new double[] { 3.5, 7.2, 1.1 };
        var input = new Tensor<double>(data, new[] { 1, 3 });
        var layer = new MaxPoolingLayer<double>(3);

        var output = layer.Forward(input);

        Assert.Equal(3.5, output[0], Tolerance);
        Assert.Equal(7.2, output[1], Tolerance);
        Assert.Equal(1.1, output[2], Tolerance);
    }

    [Fact]
    public void MaxPooling_AllSameValues_ReturnsConstant()
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

    [Fact]
    public void TNet_Forward_PreservesShape()
    {
        var tnet = new TNetLayer<double>(3, 3, new[] { 16, 32 }, new[] { 16 });
        var input = CreateRandomTensor(16, 3, seed: 53);

        var output = tnet.Forward(input);

        Assert.Equal(input.Shape[0], output.Shape[0]);
        Assert.Equal(input.Shape[1], output.Shape[1]);
    }

    [Fact]
    public void TNet_Forward_OutputIsFinite()
    {
        var tnet = new TNetLayer<double>(3, 3, new[] { 16, 32 }, new[] { 16 });
        var input = CreateRandomTensor(16, 3, seed: 59);

        var output = tnet.Forward(input);

        AssertAllFinite(output);
    }

    [Fact]
    public void TNet_Backward_ProducesCorrectShape()
    {
        var tnet = new TNetLayer<double>(3, 3, new[] { 16, 32 }, new[] { 16 });
        var input = CreateRandomTensor(16, 3, seed: 61);
        tnet.Forward(input);

        var grad = CreateRandomTensor(16, 3, seed: 67);
        var inputGrad = tnet.Backward(grad);

        Assert.Equal(16, inputGrad.Shape[0]);
        Assert.Equal(3, inputGrad.Shape[1]);
        AssertAllFinite(inputGrad);
    }

    [Fact]
    public void TNet_ParameterCount_IsPositive()
    {
        var tnet = new TNetLayer<double>(3, 3, new[] { 16, 32 }, new[] { 16 });
        Assert.True(tnet.ParameterCount > 0);
    }

    [Fact]
    public void TNet_InvalidTransformDim_ThrowsArgumentOutOfRange()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TNetLayer<double>(0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TNetLayer<double>(-1, 3));
    }

    [Fact]
    public void TNet_TransformDimGreaterThanFeatures_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TNetLayer<double>(5, 3));
    }

    [Fact]
    public void TNet_InvalidFeatures_ThrowsArgumentOutOfRange()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TNetLayer<double>(3, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TNetLayer<double>(3, -1));
    }

    [Fact]
    public void TNet_UpdateParameters_DoesNotThrow()
    {
        var tnet = new TNetLayer<double>(3, 3, new[] { 8, 16 }, new[] { 8 });
        var input = CreateRandomTensor(8, 3, seed: 71);
        tnet.Forward(input);

        var grad = CreateRandomTensor(8, 3, seed: 73);
        tnet.Backward(grad);
        tnet.UpdateParameters(0.001);
    }

    [Fact]
    public void TNet_FeatureTransform_PreservesExtraFeatures()
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

    [Fact]
    public void TNet_ResetState_ClearsCachedData()
    {
        var tnet = new TNetLayer<double>(3, 3, new[] { 16, 32 }, new[] { 16 });
        var input = CreateRandomTensor(8, 3, seed: 83);
        tnet.Forward(input);

        tnet.ResetState();

        // Backward should fail after reset since no forward was done
        var grad = CreateRandomTensor(8, 3, seed: 89);
        Assert.Throws<InvalidOperationException>(() => tnet.Backward(grad));
    }

    [Fact]
    public void TNet_BackwardWithoutForward_Throws()
    {
        var tnet = new TNetLayer<double>(3, 3, new[] { 16, 32 }, new[] { 16 });
        var grad = CreateRandomTensor(8, 3, seed: 97);

        Assert.Throws<InvalidOperationException>(() => tnet.Backward(grad));
    }

    #endregion

    #region PointCloudData Tests

    [Fact]
    public void PointCloudData_ConstructFromTensor_SetsProperties()
    {
        var points = CreateRandomTensor(50, 3, seed: 101);
        var cloud = new PointCloudData<double>(points);

        Assert.Equal(50, cloud.NumPoints);
        Assert.Equal(3, cloud.NumFeatures);
        Assert.Null(cloud.Labels);
    }

    [Fact]
    public void PointCloudData_WithLabels_StoresLabels()
    {
        var points = CreateRandomTensor(10, 3, seed: 103);
        var labels = new Vector<double>(new double[] { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 });
        var cloud = new PointCloudData<double>(points, labels);

        Assert.NotNull(cloud.Labels);
        Assert.Equal(10, cloud.Labels.Length);
        Assert.Equal(0.0, cloud.Labels[0]);
        Assert.Equal(1.0, cloud.Labels[1]);
    }

    [Fact]
    public void PointCloudData_GetCoordinates_ExtractsXYZOnly()
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

        Assert.Equal(new[] { 2, 3 }, coords.Shape);
        Assert.Equal(1.0, coords[0], Tolerance);
        Assert.Equal(2.0, coords[1], Tolerance);
        Assert.Equal(3.0, coords[2], Tolerance);
        Assert.Equal(4.0, coords[3], Tolerance);
    }

    [Fact]
    public void PointCloudData_GetCoordinates_ReturnsOriginalWhen3Features()
    {
        var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var points = new Tensor<double>(data, new[] { 2, 3 });
        var cloud = new PointCloudData<double>(points);

        var coords = cloud.GetCoordinates();

        // Should return the same tensor reference when NumFeatures == 3
        Assert.Same(cloud.Points, coords);
    }

    [Fact]
    public void PointCloudData_GetFeatures_ReturnsNullForXYZOnly()
    {
        var points = CreateRandomTensor(10, 3, seed: 107);
        var cloud = new PointCloudData<double>(points);

        Assert.Null(cloud.GetFeatures());
    }

    [Fact]
    public void PointCloudData_GetFeatures_ExtractsNonXYZColumns()
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
        Assert.Equal(new[] { 2, 3 }, features.Shape);
        Assert.Equal(100.0, features[0], Tolerance);
        Assert.Equal(200.0, features[1], Tolerance);
        Assert.Equal(50.0, features[2], Tolerance);
    }

    [Fact]
    public void PointCloudData_FromCoordinates_CreatesCorrectTensor()
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

    [Fact]
    public void PointCloudPipeline_ConvMaxPoolFC_ProducesFiniteOutput()
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

    [Fact]
    public void PointCloudPipeline_MultiLayerConv_IncreasesFeatureDim()
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

    [Fact]
    public void PointCloudPipeline_BackwardPropagates_ThroughConvLayers()
    {
        var conv1 = new PointConvolutionLayer<double>(3, 8);
        var conv2 = new PointConvolutionLayer<double>(8, 4);

        var input = CreateRandomTensor(16, 3, seed: 127);
        var h1 = conv1.Forward(input);
        var h2 = conv2.Forward(h1);

        var outGrad = CreateRandomTensor(16, 4, seed: 131);
        var grad1 = conv2.Backward(outGrad);
        var grad0 = conv1.Backward(grad1);

        Assert.Equal(16, grad0.Shape[0]);
        Assert.Equal(3, grad0.Shape[1]);
        AssertAllFinite(grad0);
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
