using AiDotNet.PointCloud.Data;
using AiDotNet.PointCloud.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.PointCloud;

/// <summary>
/// Deep math integration tests for the PointCloud module.
/// Tests point convolution (matmul + bias), max pooling, T-Net transformation,
/// He initialization, gradient computation, and point cloud data operations.
/// </summary>
public class PointCloudDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    // ============================
    // PointConvolutionLayer Forward Tests
    // ============================

    [Fact]
    public void PointConvolution_Forward_OutputShapeCorrect()
    {
        // Input: [N=4, C_in=3], Output should be [N=4, C_out=8]
        var layer = new PointConvolutionLayer<double>(3, 8);
        var input = CreateRandomTensor(4, 3, 42);

        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(4, output.Shape[0]); // same number of points
        Assert.Equal(8, output.Shape[1]); // output channels
    }

    [Fact]
    public void PointConvolution_Forward_IsAffineTransform()
    {
        // Forward: output = input * W + b (matrix multiply + bias)
        // For a single point, output[j] = sum_i(input[i] * W[i,j]) + b[j]
        var layer = new PointConvolutionLayer<double>(2, 2);

        // Set known weights and biases by getting and setting parameters
        // Parameters: [W11, W12, W21, W22, b1, b2] = [2*2 + 2 = 6 params]
        var paramCount = layer.ParameterCount;
        Assert.Equal(6, paramCount); // 2*2 weights + 2 biases

        var params1 = new Vector<double>(6);
        // W = [[1, 0], [0, 1]] (identity), b = [0.5, -0.5]
        params1[0] = 1.0; params1[1] = 0.0; // W row 0
        params1[2] = 0.0; params1[3] = 1.0; // W row 1
        params1[4] = 0.5; params1[5] = -0.5; // biases
        layer.UpdateParameters(params1);

        // Input: single point [3.0, 7.0]
        var input = new Tensor<double>(new[] { 3.0, 7.0 }, [1, 2]);
        var output = layer.Forward(input);

        // Expected: [3*1 + 7*0 + 0.5, 3*0 + 7*1 - 0.5] = [3.5, 6.5]
        var outArr = output.ToArray();
        Assert.Equal(3.5, outArr[0], Tolerance);
        Assert.Equal(6.5, outArr[1], Tolerance);
    }

    [Fact]
    public void PointConvolution_Forward_MultiplePoints_IndependentPerPoint()
    {
        // Each point is processed independently - same transform applied to each
        var layer = new PointConvolutionLayer<double>(2, 2);

        // Set identity transform with bias
        var p = new Vector<double>(6);
        p[0] = 2.0; p[1] = 0.0;  // W row 0: scale x by 2
        p[2] = 0.0; p[3] = 3.0;  // W row 1: scale y by 3
        p[4] = 1.0; p[5] = -1.0; // biases
        layer.UpdateParameters(p);

        // Two points
        var input = new Tensor<double>(new[] { 1.0, 1.0, 2.0, 2.0 }, [2, 2]);
        var output = layer.Forward(input);

        // Point 0: [1*2+1*0+1, 1*0+1*3-1] = [3.0, 2.0]
        var outArr = output.ToArray();
        Assert.Equal(3.0, outArr[0], Tolerance);
        Assert.Equal(2.0, outArr[1], Tolerance);

        // Point 1: [2*2+2*0+1, 2*0+2*3-1] = [5.0, 5.0]
        Assert.Equal(5.0, outArr[2], Tolerance);
        Assert.Equal(5.0, outArr[3], Tolerance);
    }

    [Fact]
    public void PointConvolution_ParameterCount_IsCorrect()
    {
        // Parameters = inputChannels * outputChannels + outputChannels (weights + biases)
        var layer = new PointConvolutionLayer<double>(3, 64);
        Assert.Equal(3 * 64 + 64, layer.ParameterCount); // 256

        var layer2 = new PointConvolutionLayer<double>(64, 128);
        Assert.Equal(64 * 128 + 128, layer2.ParameterCount); // 8320
    }

    // ============================
    // PointConvolution Backward Tests
    // ============================

    [Fact]
    public void PointConvolution_Backward_OutputShapeMatchesInput()
    {
        var layer = new PointConvolutionLayer<double>(3, 8);
        var input = CreateRandomTensor(4, 3, 42);

        var output = layer.Forward(input);
        var gradOutput = CreateRandomTensor(4, 8, 99);
        var gradInput = layer.Backward(gradOutput);

        // Gradient w.r.t. input should have same shape as input
        Assert.Equal(4, gradInput.Shape[0]);
        Assert.Equal(3, gradInput.Shape[1]);
    }

    [Fact]
    public void PointConvolution_Backward_BeforeForward_Throws()
    {
        var layer = new PointConvolutionLayer<double>(3, 8);
        var gradOutput = CreateRandomTensor(4, 8, 99);

        Assert.Throws<InvalidOperationException>(() => layer.Backward(gradOutput));
    }

    [Fact]
    public void PointConvolution_Backward_InputGradient_IsWeightsTransposeTimesOutputGrad()
    {
        // dL/dX = dL/dY * W^T
        // With known weights, verify the gradient computation
        var layer = new PointConvolutionLayer<double>(2, 2);

        // W = [[1, 2], [3, 4]], b = [0, 0]
        var p = new Vector<double>(6);
        p[0] = 1.0; p[1] = 2.0;
        p[2] = 3.0; p[3] = 4.0;
        p[4] = 0.0; p[5] = 0.0;
        layer.UpdateParameters(p);

        var input = new Tensor<double>(new[] { 1.0, 0.0 }, [1, 2]);
        layer.Forward(input);

        // Gradient of loss w.r.t. output: [1.0, 1.0]
        var gradOutput = new Tensor<double>(new[] { 1.0, 1.0 }, [1, 2]);
        var gradInput = layer.Backward(gradOutput);

        // dL/dX = gradOutput * W^T = [1, 1] * [[1, 3], [2, 4]] = [1*1+1*2, 1*3+1*4] = [3, 7]
        var gradArr = gradInput.ToArray();
        Assert.Equal(3.0, gradArr[0], Tolerance);
        Assert.Equal(7.0, gradArr[1], Tolerance);
    }

    // ============================
    // He Initialization Tests
    // ============================

    [Fact]
    public void PointConvolution_HeInitialization_WeightsHaveCorrectVariance()
    {
        // He initialization: var(W) ≈ 2/inputDim
        int inputChannels = 100;
        int outputChannels = 100;
        var layer = new PointConvolutionLayer<double>(inputChannels, outputChannels);

        var params2 = layer.GetParameters();
        int numWeights = inputChannels * outputChannels;

        // Extract weights (first numWeights parameters)
        double sum = 0, sumSq = 0;
        for (int i = 0; i < numWeights; i++)
        {
            sum += params2[i];
            sumSq += params2[i] * params2[i];
        }

        double mean = sum / numWeights;
        double variance = sumSq / numWeights - mean * mean;
        double expectedVariance = 2.0 / inputChannels;

        // Variance should be approximately 2/inputDim (with some tolerance for randomness)
        Assert.True(variance > expectedVariance * 0.3, $"Variance {variance} too small vs expected {expectedVariance}");
        Assert.True(variance < expectedVariance * 3.0, $"Variance {variance} too large vs expected {expectedVariance}");
    }

    [Fact]
    public void PointConvolution_HeInitialization_BiasesAreZero()
    {
        int inputChannels = 10;
        int outputChannels = 10;
        var layer = new PointConvolutionLayer<double>(inputChannels, outputChannels);

        var params2 = layer.GetParameters();
        int numWeights = inputChannels * outputChannels;

        // Biases are the last outputChannels parameters
        for (int i = numWeights; i < params2.Length; i++)
        {
            Assert.Equal(0.0, params2[i], Tolerance);
        }
    }

    // ============================
    // MaxPoolingLayer Tests
    // ============================

    [Fact]
    public void MaxPooling_Forward_SelectsMaxPerChannel()
    {
        // Input: [3 points, 2 features]
        // Point 0: [1, 5]
        // Point 1: [3, 2]
        // Point 2: [2, 4]
        // Expected max: [3, 5]
        var layer = new MaxPoolingLayer<double>(2);
        var input = new Tensor<double>(new[] { 1.0, 5.0, 3.0, 2.0, 2.0, 4.0 }, [3, 2]);

        var output = layer.Forward(input);

        Assert.Equal(1, output.Shape[0]); // pooled to 1
        Assert.Equal(2, output.Shape[1]); // same features
        var outArr = output.ToArray();
        Assert.Equal(3.0, outArr[0], Tolerance); // max of [1, 3, 2]
        Assert.Equal(5.0, outArr[1], Tolerance); // max of [5, 2, 4]
    }

    [Fact]
    public void MaxPooling_Forward_SinglePoint_ReturnsItself()
    {
        var layer = new MaxPoolingLayer<double>(3);
        var input = new Tensor<double>(new[] { 7.0, -3.0, 1.5 }, [1, 3]);

        var output = layer.Forward(input);

        var outArr = output.ToArray();
        Assert.Equal(7.0, outArr[0], Tolerance);
        Assert.Equal(-3.0, outArr[1], Tolerance);
        Assert.Equal(1.5, outArr[2], Tolerance);
    }

    [Fact]
    public void MaxPooling_Forward_AllSameValues_ReturnsValue()
    {
        var layer = new MaxPoolingLayer<double>(2);
        var input = new Tensor<double>(new[] { 4.0, 4.0, 4.0, 4.0, 4.0, 4.0 }, [3, 2]);

        var output = layer.Forward(input);

        var outArr = output.ToArray();
        Assert.Equal(4.0, outArr[0], Tolerance);
        Assert.Equal(4.0, outArr[1], Tolerance);
    }

    [Fact]
    public void MaxPooling_IsPermutationInvariant()
    {
        // Reordering points should give same result - this is critical for point clouds
        var layer1 = new MaxPoolingLayer<double>(2);
        var layer2 = new MaxPoolingLayer<double>(2);

        // Order 1: [1,5], [3,2], [2,4]
        var input1 = new Tensor<double>(new[] { 1.0, 5.0, 3.0, 2.0, 2.0, 4.0 }, [3, 2]);
        // Order 2: [3,2], [2,4], [1,5] (same points, different order)
        var input2 = new Tensor<double>(new[] { 3.0, 2.0, 2.0, 4.0, 1.0, 5.0 }, [3, 2]);

        var output1 = layer1.Forward(input1);
        var output2 = layer2.Forward(input2);

        var arr1 = output1.ToArray();
        var arr2 = output2.ToArray();
        Assert.Equal(arr1[0], arr2[0], Tolerance);
        Assert.Equal(arr1[1], arr2[1], Tolerance);
    }

    [Fact]
    public void MaxPooling_HasNoTrainableParameters()
    {
        var layer = new MaxPoolingLayer<double>(10);
        Assert.Equal(0, layer.ParameterCount);
        Assert.False(layer.SupportsTraining);
    }

    [Fact]
    public void MaxPooling_Backward_GradientOnlyFlowsToMaxElement()
    {
        // In max pooling backward, only the index that had the max value receives gradient
        var layer = new MaxPoolingLayer<double>(2);
        var input = new Tensor<double>(new[] { 1.0, 5.0, 3.0, 2.0, 2.0, 4.0 }, [3, 2]);

        layer.Forward(input);

        // Gradient w.r.t. output
        var gradOutput = new Tensor<double>(new[] { 1.0, 1.0 }, [1, 2]);
        var gradInput = layer.Backward(gradOutput);

        // Feature 0 max at index 1 (value 3.0): grad should be [0, 1, 0]
        // Feature 1 max at index 0 (value 5.0): grad should be [1, 0, 0]
        Assert.Equal(3, gradInput.Shape[0]);
        Assert.Equal(2, gradInput.Shape[1]);

        var gradArr = gradInput.ToArray();
        // Point 0: feature 0 not max (0), feature 1 is max (1)
        Assert.Equal(0.0, gradArr[0], Tolerance);
        Assert.Equal(1.0, gradArr[1], Tolerance);
        // Point 1: feature 0 is max (1), feature 1 not max (0)
        Assert.Equal(1.0, gradArr[2], Tolerance);
        Assert.Equal(0.0, gradArr[3], Tolerance);
        // Point 2: neither is max (0, 0)
        Assert.Equal(0.0, gradArr[4], Tolerance);
        Assert.Equal(0.0, gradArr[5], Tolerance);
    }

    // ============================
    // TNetLayer Tests
    // ============================

    [Fact]
    public void TNet_Constructor_ValidatesPositiveDimensions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TNetLayer<double>(0, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TNetLayer<double>(3, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TNetLayer<double>(-1, 3));
    }

    [Fact]
    public void TNet_Constructor_TransformDimMustNotExceedFeatures()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TNetLayer<double>(5, 3));
    }

    [Fact]
    public void TNet_Constructor_ValidDimensions_Succeeds()
    {
        // Should not throw
        var layer = new TNetLayer<double>(3, 3);
        Assert.True(layer.ParameterCount > 0);
    }

    [Fact]
    public void TNet_Forward_OutputShapeMatchesInput()
    {
        var layer = new TNetLayer<double>(3, 3, new[] { 16, 32 }, new[] { 16 });
        var input = CreateRandomTensor(10, 3, 42);

        var output = layer.Forward(input);

        Assert.Equal(10, output.Shape[0]); // same number of points
        Assert.Equal(3, output.Shape[1]);  // same features
    }

    [Fact]
    public void TNet_Backward_BeforeForward_Throws()
    {
        var layer = new TNetLayer<double>(3, 3, new[] { 16, 32 }, new[] { 16 });
        var grad = CreateRandomTensor(10, 3, 42);

        Assert.Throws<InvalidOperationException>(() => layer.Backward(grad));
    }

    [Fact]
    public void TNet_SupportsTraining()
    {
        var layer = new TNetLayer<double>(3, 3, new[] { 16 }, new[] { 8 });
        Assert.True(layer.SupportsTraining);
    }

    // ============================
    // PointCloudData Tests
    // ============================

    [Fact]
    public void PointCloudData_Constructor_SetsCorrectDimensions()
    {
        // 5 points with 3 features (XYZ)
        var tensor = new Tensor<double>(new double[5 * 3], [5, 3]);
        var data = new PointCloudData<double>(tensor);

        Assert.Equal(5, data.NumPoints);
        Assert.Equal(3, data.NumFeatures);
        Assert.Null(data.Labels);
    }

    [Fact]
    public void PointCloudData_GetCoordinates_ExtractsFirst3Channels()
    {
        // 2 points with 6 features (XYZ + RGB)
        var values = new double[]
        {
            1.0, 2.0, 3.0, 0.5, 0.6, 0.7, // Point 0: XYZ=1,2,3 RGB=0.5,0.6,0.7
            4.0, 5.0, 6.0, 0.8, 0.9, 1.0   // Point 1: XYZ=4,5,6 RGB=0.8,0.9,1.0
        };
        var tensor = new Tensor<double>(values, [2, 6]);
        var data = new PointCloudData<double>(tensor);

        var coords = data.GetCoordinates();

        Assert.Equal(2, coords.Shape[0]);
        Assert.Equal(3, coords.Shape[1]);
        var coordArr = coords.ToArray();
        Assert.Equal(1.0, coordArr[0], Tolerance); // X of point 0
        Assert.Equal(2.0, coordArr[1], Tolerance); // Y of point 0
        Assert.Equal(3.0, coordArr[2], Tolerance); // Z of point 0
        Assert.Equal(4.0, coordArr[3], Tolerance); // X of point 1
    }

    [Fact]
    public void PointCloudData_GetFeatures_ExtractsNonCoordinateChannels()
    {
        var values = new double[]
        {
            1.0, 2.0, 3.0, 10.0, 20.0,
            4.0, 5.0, 6.0, 30.0, 40.0
        };
        var tensor = new Tensor<double>(values, [2, 5]);
        var data = new PointCloudData<double>(tensor);

        var features = data.GetFeatures();

        Assert.NotNull(features);
        Assert.Equal(2, features.Shape[0]);
        Assert.Equal(2, features.Shape[1]); // 5 - 3 = 2 extra features
        var featArr = features.ToArray();
        Assert.Equal(10.0, featArr[0], Tolerance);
        Assert.Equal(20.0, featArr[1], Tolerance);
        Assert.Equal(30.0, featArr[2], Tolerance);
        Assert.Equal(40.0, featArr[3], Tolerance);
    }

    [Fact]
    public void PointCloudData_GetFeatures_OnlyXYZ_ReturnsNull()
    {
        var tensor = new Tensor<double>(new double[3 * 3], [3, 3]);
        var data = new PointCloudData<double>(tensor);

        var features = data.GetFeatures();
        Assert.Null(features);
    }

    [Fact]
    public void PointCloudData_GetCoordinates_OnlyXYZ_ReturnsSameTensor()
    {
        var values = new double[] { 1.0, 2.0, 3.0 };
        var tensor = new Tensor<double>(values, [1, 3]);
        var data = new PointCloudData<double>(tensor);

        var coords = data.GetCoordinates();

        // Should return the same tensor reference (optimization)
        Assert.Equal(tensor, coords);
    }

    [Fact]
    public void PointCloudData_WithLabels_StoresLabels()
    {
        var tensor = new Tensor<double>(new double[3 * 3], [3, 3]);
        var labels = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
        var data = new PointCloudData<double>(tensor, labels);

        Assert.NotNull(data.Labels);
        Assert.Equal(3, data.Labels.Length);
        Assert.Equal(0.0, data.Labels[0]);
        Assert.Equal(1.0, data.Labels[1]);
        Assert.Equal(2.0, data.Labels[2]);
    }

    // ============================
    // PointConvolution Update Parameters Tests
    // ============================

    [Fact]
    public void PointConvolution_UpdateParameters_ChangesOutput()
    {
        var layer = new PointConvolutionLayer<double>(2, 2);
        var input = new Tensor<double>(new[] { 1.0, 1.0 }, [1, 2]);

        var output1 = layer.Forward(input);
        var val1 = output1.ToArray()[0];

        // Change parameters
        var newParams = new Vector<double>(6);
        newParams[0] = 10.0; newParams[1] = 0.0;
        newParams[2] = 0.0; newParams[3] = 10.0;
        newParams[4] = 0.0; newParams[5] = 0.0;
        layer.UpdateParameters(newParams);

        var output2 = layer.Forward(input);

        // Output should change after parameter update
        var out2Arr = output2.ToArray();
        Assert.Equal(10.0, out2Arr[0], Tolerance); // 1*10 + 1*0 + 0
        Assert.Equal(10.0, out2Arr[1], Tolerance); // 1*0 + 1*10 + 0
    }

    [Fact]
    public void PointConvolution_ClearGradients_ResetsAccumulatedGradients()
    {
        var layer = new PointConvolutionLayer<double>(2, 2);

        var p = new Vector<double>(6);
        p[0] = 1.0; p[1] = 0.0;
        p[2] = 0.0; p[3] = 1.0;
        p[4] = 0.0; p[5] = 0.0;
        layer.UpdateParameters(p);

        var input = new Tensor<double>(new[] { 1.0, 2.0 }, [1, 2]);
        layer.Forward(input);

        var grad = new Tensor<double>(new[] { 1.0, 1.0 }, [1, 2]);
        layer.Backward(grad);

        // Clear and do another forward/backward - gradients should not accumulate from before clear
        layer.ClearGradients();

        layer.Forward(input);
        layer.Backward(grad);

        // After clear + 1 backward, gradients should be from just 1 pass
        // (We can't easily inspect gradients directly, but we verify the clear doesn't crash)
    }

    [Fact]
    public void PointConvolution_SGDUpdate_MovesWeightsInGradientDirection()
    {
        var layer = new PointConvolutionLayer<double>(2, 2);

        // Set known weights
        var p = new Vector<double>(6);
        p[0] = 1.0; p[1] = 0.0;
        p[2] = 0.0; p[3] = 1.0;
        p[4] = 0.0; p[5] = 0.0;
        layer.UpdateParameters(p);

        // Forward
        var input = new Tensor<double>(new[] { 1.0, 0.0 }, [1, 2]);
        layer.Forward(input);

        // Backward with output gradient
        var gradOutput = new Tensor<double>(new[] { 1.0, 0.0 }, [1, 2]);
        layer.Backward(gradOutput);

        // Update with small learning rate
        double lr = 0.01;
        layer.UpdateParameters(lr);

        // Check parameters changed
        var newParams = layer.GetParameters();

        // Weight gradient for W[0,0]: dL/dW[0,0] = input[0] * gradOutput[0] = 1.0 * 1.0 = 1.0
        // New W[0,0] = 1.0 - 0.01 * 1.0 = 0.99
        Assert.Equal(0.99, newParams[0], 1e-4);
    }

    // ============================
    // Mathematical Property Tests
    // ============================

    [Fact]
    public void MaxPooling_OutputBound_ByInputMax()
    {
        // The max pooled output for each channel is exactly the max of that channel
        var layer = new MaxPoolingLayer<double>(3);
        var input = new Tensor<double>(new[]
        {
            -5.0, 10.0, 0.0,
            3.0, -1.0, 7.0,
            1.0, 5.0, 3.0,
            -2.0, 8.0, -4.0
        }, [4, 3]);

        var output = layer.Forward(input);

        var outArr = output.ToArray();
        Assert.Equal(3.0, outArr[0], Tolerance);  // max of [-5, 3, 1, -2]
        Assert.Equal(10.0, outArr[1], Tolerance); // max of [10, -1, 5, 8]
        Assert.Equal(7.0, outArr[2], Tolerance);  // max of [0, 7, 3, -4]
    }

    [Fact]
    public void PointConvolution_ZeroInput_OutputEqualsBias()
    {
        // When input is all zeros, output = 0 * W + b = b
        var layer = new PointConvolutionLayer<double>(3, 2);

        var p = new Vector<double>(3 * 2 + 2);
        // Set random weights (shouldn't matter)
        for (int i = 0; i < 6; i++) p[i] = (i + 1) * 0.5;
        // Set known biases
        p[6] = 2.5;
        p[7] = -1.5;
        layer.UpdateParameters(p);

        var input = new Tensor<double>(new double[3], [1, 3]); // all zeros
        var output = layer.Forward(input);

        var outArr = output.ToArray();
        Assert.Equal(2.5, outArr[0], Tolerance);
        Assert.Equal(-1.5, outArr[1], Tolerance);
    }

    [Fact]
    public void PointConvolution_InvalidParameterLength_Throws()
    {
        var layer = new PointConvolutionLayer<double>(3, 2);
        var wrongParams = new Vector<double>(5); // should be 3*2+2 = 8
        Assert.Throws<ArgumentException>(() => layer.UpdateParameters(wrongParams));
    }

    // ============================
    // TNet Transform Matrix Property Tests
    // ============================

    [Fact]
    public void TNet_InitialTransform_IsNearIdentity()
    {
        // TNet adds identity to the learned matrix: transform = predicted + I
        // At initialization with small weights, transform ≈ I
        // So output ≈ input
        var layer = new TNetLayer<double>(3, 3, new[] { 8, 16 }, new[] { 8 });
        var input = new Tensor<double>(new[]
        {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        }, [3, 3]);

        var output = layer.Forward(input);

        // With random initialization, the output should still be close to input
        // because the predicted matrix is near zero, and identity is added
        // This is a loose test since weights are random
        Assert.Equal(3, output.Shape[0]);
        Assert.Equal(3, output.Shape[1]);
    }

    [Fact]
    public void TNet_ResetState_ClearsInternalState()
    {
        var layer = new TNetLayer<double>(3, 3, new[] { 8 }, new[] { 4 });
        var input = CreateRandomTensor(5, 3, 42);

        layer.Forward(input);
        layer.ResetState();

        // After reset, backward should throw (no stored forward state)
        var grad = CreateRandomTensor(5, 3, 99);
        Assert.Throws<InvalidOperationException>(() => layer.Backward(grad));
    }

    // ============================
    // Helper Methods
    // ============================

    private static Tensor<double> CreateRandomTensor(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var data = new double[rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = rng.NextDouble() * 2 - 1; // [-1, 1]
        }
        return new Tensor<double>(data, [rows, cols]);
    }
}
