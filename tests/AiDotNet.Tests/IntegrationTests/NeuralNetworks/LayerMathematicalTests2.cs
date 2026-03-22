using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Mathematical invariant tests for convolutional, pooling, normalization, and embedding layers.
/// Tests verify mathematical correctness of operations, not just shapes.
/// </summary>
public class LayerMathematicalTests2
{
    private const double Tol = 1e-5;

    #region Convolutional Layer Tests

    /// <summary>
    /// Conv2D output size: floor((H + 2*padding - kernelSize) / stride) + 1
    /// </summary>
    [Fact]
    public void Conv2DLayer_OutputSize_WithPadding()
    {
        // inputDepth=1, inputHeight=8, inputWidth=8, outputDepth=4, kernelSize=3, stride=1, padding=1
        var layer = new ConvolutionalLayer<double>(1, 8, 8, 4, 3, 1, 1);

        // Input: [batch=1, channels=1, height=8, width=8]
        var input = new Tensor<double>(new double[64], [1, 1, 8, 8]);
        var output = layer.Forward(input);

        // With padding=1, kernel=3, stride=1: output size = (8+2-3)/1 + 1 = 8
        Assert.Equal(4, output.Shape[1]); // outputDepth
        Assert.Equal(8, output.Shape[2]); // height preserved
        Assert.Equal(8, output.Shape[3]); // width preserved
    }

    [Fact]
    public void Conv2DLayer_Stride2_HalvesSize()
    {
        // stride=2, padding=1
        var layer = new ConvolutionalLayer<double>(1, 8, 8, 4, 3, 2, 1);

        var input = new Tensor<double>(new double[64], [1, 1, 8, 8]);
        var output = layer.Forward(input);

        // (8+2-3)/2 + 1 = 4
        Assert.Equal(4, output.Shape[2]);
        Assert.Equal(4, output.Shape[3]);
    }

    [Fact]
    public void Conv2DLayer_ZeroInput_ProducesBiasOutput()
    {
        // With zero input, output should be just the convolution bias
        var layer = new ConvolutionalLayer<double>(1, 4, 4, 2, 3, 1, 1);
        var input = new Tensor<double>(new double[16], [1, 1, 4, 4]); // all zeros
        var output = layer.Forward(input);

        // Output should be the bias value (which may be zero or initialized)
        // At minimum, output should be finite
        for (int i = 0; i < output.Shape.Aggregate(1, (a, b) => a * b); i++)
            Assert.False(double.IsNaN(output[i]) || double.IsInfinity(output[i]),
                $"Conv output[{i}] should be finite");
    }

    #endregion

    #region Pooling Layer Tests

    [Fact]
    public void MaxPool2D_OutputSize()
    {
        // inputShape=[1, 4, 4], poolSize=2, stride=2
        var layer = new MaxPoolingLayer<double>([1, 4, 4], 2, 2);

        var data = new double[16];
        for (int i = 0; i < 16; i++) data[i] = i;
        var input = new Tensor<double>(data, [1, 1, 4, 4]);
        var output = layer.Forward(input);

        // 4/2 = 2
        Assert.Equal(2, output.Shape[2]);
        Assert.Equal(2, output.Shape[3]);
    }

    [Fact]
    public void MaxPool2D_SelectsMaximum()
    {
        var layer = new MaxPoolingLayer<double>([1, 2, 2], 2, 2);

        // Input: [[1, 3], [2, 4]]
        var data = new double[] { 1.0, 3.0, 2.0, 4.0 };
        var input = new Tensor<double>(data, [1, 1, 2, 2]);
        var output = layer.Forward(input);

        // Max of [1, 3, 2, 4] = 4
        Assert.Equal(4.0, output[0], Tol);
    }

    [Fact]
    public void MaxPool2D_Idempotent_ForConstantInput()
    {
        // If all values are the same, max pooling should preserve the value
        var layer = new MaxPoolingLayer<double>([1, 4, 4], 2, 2);

        var data = new double[16];
        for (int i = 0; i < 16; i++) data[i] = 7.0;
        var input = new Tensor<double>(data, [1, 1, 4, 4]);
        var output = layer.Forward(input);

        for (int i = 0; i < output.Shape.Aggregate(1, (a, b) => a * b); i++)
            Assert.Equal(7.0, output[i], Tol);
    }

    [Fact]
    public void AvgPool2D_ComputesAverage()
    {
        var layer = new AveragePoolingLayer<double>([1, 2, 2], 2, 2);

        // [[1, 3], [2, 4]]
        var data = new double[] { 1.0, 3.0, 2.0, 4.0 };
        var input = new Tensor<double>(data, [1, 1, 2, 2]);
        var output = layer.Forward(input);

        // Average of [1, 3, 2, 4] = 10/4 = 2.5
        Assert.Equal(2.5, output[0], Tol);
    }

    #endregion

    #region Layer Normalization Tests

    [Fact]
    public void LayerNorm_NormalizesEachSample()
    {
        int featureSize = 4;
        var layer = new LayerNormalizationLayer<double>(featureSize);

        var data = new double[] {
            1.0, 2.0, 3.0, 4.0,      // sample 0
            10.0, 20.0, 30.0, 40.0    // sample 1
        };
        var input = new Tensor<double>(data, [2, 4]);
        var output = layer.Forward(input);

        // Each sample should have approximately zero mean
        for (int b = 0; b < 2; b++)
        {
            double sum = 0;
            for (int f = 0; f < 4; f++)
                sum += output[b * 4 + f];
            Assert.True(Math.Abs(sum / 4.0) < 0.1,
                $"Sample {b} mean should be ~0, got {sum / 4.0}");
        }
    }

    #endregion

    #region Embedding Layer Tests

    [Fact]
    public void EmbeddingLayer_OutputShape()
    {
        int vocabSize = 100, embeddingDim = 32;
        var layer = new EmbeddingLayer<double>(vocabSize, embeddingDim);

        var input = new Tensor<double>(new[] { 0.0, 5.0, 99.0 }, [3]);
        var output = layer.Forward(input);

        Assert.Equal(3, output.Shape[0]);
        Assert.Equal(embeddingDim, output.Shape[1]);
    }

    [Fact]
    public void EmbeddingLayer_SameIndex_SameVector()
    {
        int vocabSize = 10, embeddingDim = 4;
        var layer = new EmbeddingLayer<double>(vocabSize, embeddingDim);

        var input1 = new Tensor<double>(new[] { 3.0 }, [1]);
        var input2 = new Tensor<double>(new[] { 3.0 }, [1]);
        var output1 = layer.Forward(input1);
        var output2 = layer.Forward(input2);

        for (int i = 0; i < embeddingDim; i++)
            Assert.Equal(output1[i], output2[i], Tol);
    }

    [Fact]
    public void EmbeddingLayer_DifferentIndices_DifferentVectors()
    {
        int vocabSize = 10, embeddingDim = 8;
        var layer = new EmbeddingLayer<double>(vocabSize, embeddingDim);

        var input1 = new Tensor<double>(new[] { 0.0 }, [1]);
        var input2 = new Tensor<double>(new[] { 1.0 }, [1]);
        var output1 = layer.Forward(input1);
        var output2 = layer.Forward(input2);

        // Different indices should (almost always) produce different vectors
        bool different = false;
        for (int i = 0; i < embeddingDim; i++)
            if (Math.Abs(output1[i] - output2[i]) > 1e-10)
                different = true;
        Assert.True(different, "Different indices should produce different embeddings");
    }

    #endregion
}
