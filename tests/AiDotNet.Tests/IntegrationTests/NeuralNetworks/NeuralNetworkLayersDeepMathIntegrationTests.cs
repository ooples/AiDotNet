using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Deep math-correctness integration tests for neural network layers.
/// Each test hand-computes expected outputs and verifies the code matches.
/// </summary>
public class NeuralNetworkLayersDeepMathIntegrationTests
{
    private const double Tol = 1e-6;

    // ========================================================================
    // PositionalEncoding - Sinusoidal formula verification
    // ========================================================================

    [Fact]
    public void PositionalEncoding_Position0_AllSinAreZero_AllCosAreOne()
    {
        // PE(0, 2i) = sin(0 / 10000^(2i/d)) = sin(0) = 0
        // PE(0, 2i+1) = cos(0 / 10000^(2i/d)) = cos(0) = 1
        int embeddingSize = 8;
        var layer = new PositionalEncodingLayer<double>(maxSequenceLength: 10, embeddingSize: embeddingSize);

        // Create zero input (1 position, embeddingSize dims)
        var input = new Tensor<double>(new[] { 1, embeddingSize });
        var output = layer.Forward(input);

        for (int i = 0; i < embeddingSize / 2; i++)
        {
            // Even indices: sin(0) = 0
            Assert.Equal(0.0, NumOps<double>.ToDouble(output[0, 2 * i]), Tol);
            // Odd indices: cos(0) = 1
            Assert.Equal(1.0, NumOps<double>.ToDouble(output[0, 2 * i + 1]), Tol);
        }
    }

    [Fact]
    public void PositionalEncoding_Position1_HandComputed()
    {
        // For position 1, embedding_size=4:
        // PE(1, 0) = sin(1 / 10000^(0/4)) = sin(1 / 1) = sin(1) ≈ 0.8415
        // PE(1, 1) = cos(1 / 10000^(0/4)) = cos(1 / 1) = cos(1) ≈ 0.5403
        // PE(1, 2) = sin(1 / 10000^(2/4)) = sin(1 / 100) = sin(0.01) ≈ 0.01
        // PE(1, 3) = cos(1 / 10000^(2/4)) = cos(1 / 100) = cos(0.01) ≈ 1.0
        int embeddingSize = 4;
        var layer = new PositionalEncodingLayer<double>(maxSequenceLength: 10, embeddingSize: embeddingSize);

        var input = new Tensor<double>(new[] { 2, embeddingSize }); // 2 positions
        var output = layer.Forward(input);

        // Position 1, dim 0: sin(1 / 10000^(0/4)) = sin(1)
        double expected_1_0 = Math.Sin(1.0);
        Assert.Equal(expected_1_0, NumOps<double>.ToDouble(output[1, 0]), 1e-4);

        // Position 1, dim 1: cos(1 / 10000^(0/4)) = cos(1)
        double expected_1_1 = Math.Cos(1.0);
        Assert.Equal(expected_1_1, NumOps<double>.ToDouble(output[1, 1]), 1e-4);

        // Position 1, dim 2: sin(1 / 10000^(2/4)) = sin(1/100)
        double expected_1_2 = Math.Sin(1.0 / Math.Pow(10000, 2.0 / 4.0));
        Assert.Equal(expected_1_2, NumOps<double>.ToDouble(output[1, 2]), 1e-4);

        // Position 1, dim 3: cos(1 / 10000^(2/4)) = cos(1/100)
        double expected_1_3 = Math.Cos(1.0 / Math.Pow(10000, 2.0 / 4.0));
        Assert.Equal(expected_1_3, NumOps<double>.ToDouble(output[1, 3]), 1e-4);
    }

    [Fact]
    public void PositionalEncoding_DifferentPositions_AreDistinct()
    {
        // Different positions should produce different encoding vectors
        int embeddingSize = 16;
        var layer = new PositionalEncodingLayer<double>(maxSequenceLength: 100, embeddingSize: embeddingSize);

        var input = new Tensor<double>(new[] { 5, embeddingSize });
        var output = layer.Forward(input);

        // Check that position 0 and position 1 encodings differ
        double sumDiff01 = 0;
        double sumDiff12 = 0;
        for (int d = 0; d < embeddingSize; d++)
        {
            double v0 = NumOps<double>.ToDouble(output[0, d]);
            double v1 = NumOps<double>.ToDouble(output[1, d]);
            double v2 = NumOps<double>.ToDouble(output[2, d]);
            sumDiff01 += Math.Abs(v0 - v1);
            sumDiff12 += Math.Abs(v1 - v2);
        }
        Assert.True(sumDiff01 > 0.1, "Position 0 and 1 encodings should differ");
        Assert.True(sumDiff12 > 0.1, "Position 1 and 2 encodings should differ");
    }

    [Fact]
    public void PositionalEncoding_AddsToInput_NotReplaces()
    {
        // The forward pass should ADD encodings to the input, not replace it
        int embeddingSize = 4;
        var layer = new PositionalEncodingLayer<double>(maxSequenceLength: 10, embeddingSize: embeddingSize);

        // Create input with known value
        var input = new Tensor<double>(new[] { 1, embeddingSize });
        input[0, 0] = 5.0;
        input[0, 1] = 5.0;
        input[0, 2] = 5.0;
        input[0, 3] = 5.0;

        var output = layer.Forward(input);

        // Position 0: sin dims are 0, cos dims are 1
        // So output should be input + encoding: 5+0=5 for even, 5+1=6 for odd
        Assert.Equal(5.0, NumOps<double>.ToDouble(output[0, 0]), Tol); // 5 + sin(0) = 5
        Assert.Equal(6.0, NumOps<double>.ToDouble(output[0, 1]), Tol); // 5 + cos(0) = 6
        Assert.Equal(5.0, NumOps<double>.ToDouble(output[0, 2]), Tol); // 5 + sin(0) = 5
        Assert.Equal(6.0, NumOps<double>.ToDouble(output[0, 3]), Tol); // 5 + cos(0) = 6
    }

    [Fact]
    public void PositionalEncoding_HigherDimensions_LowerFrequency()
    {
        // The encoding uses sin(pos / 10000^(2i/d))
        // Higher dimension index i → smaller divisor → lower frequency → smaller values for small pos
        // At position 1: dim 0 has freq 1/1, dim 2 has freq 1/100 (for d=4)
        // So |PE(1,0)| > |PE(1,2)|
        int embeddingSize = 4;
        var layer = new PositionalEncodingLayer<double>(maxSequenceLength: 10, embeddingSize: embeddingSize);

        var input = new Tensor<double>(new[] { 2, embeddingSize });
        var output = layer.Forward(input);

        double pe_1_0 = Math.Abs(NumOps<double>.ToDouble(output[1, 0])); // sin(1)
        double pe_1_2 = Math.Abs(NumOps<double>.ToDouble(output[1, 2])); // sin(0.01)

        Assert.True(pe_1_0 > pe_1_2,
            $"Low-dimension sin should have higher magnitude: |sin(1)|={pe_1_0} vs |sin(0.01)|={pe_1_2}");
    }

    [Fact]
    public void PositionalEncoding_Backward_PassesGradientThrough()
    {
        // Since positional encoding just adds a constant, backward should pass gradient through unchanged
        int embeddingSize = 4;
        var layer = new PositionalEncodingLayer<double>(maxSequenceLength: 10, embeddingSize: embeddingSize);

        var input = new Tensor<double>(new[] { 2, embeddingSize });
        layer.Forward(input);

        var grad = new Tensor<double>(new[] { 2, embeddingSize });
        grad[0, 0] = 1.0;
        grad[0, 1] = 2.0;
        grad[1, 2] = 3.0;

        var inputGrad = layer.Backward(grad);

        // Input gradient should equal output gradient (addition of constant)
        Assert.Equal(1.0, NumOps<double>.ToDouble(inputGrad[0, 0]), Tol);
        Assert.Equal(2.0, NumOps<double>.ToDouble(inputGrad[0, 1]), Tol);
        Assert.Equal(3.0, NumOps<double>.ToDouble(inputGrad[1, 2]), Tol);
    }

    // ========================================================================
    // LayerNormalization - Forward pass math verification
    // ========================================================================

    [Fact]
    public void LayerNorm_UniformInput_OutputIsZero()
    {
        // If all features are the same value, mean=value, variance=0
        // normalized = (value - value) / sqrt(0 + eps) = 0
        // output = gamma * 0 + beta = 0 + 0 = 0 (gamma=1, beta=0 initially)
        int featureSize = 4;
        var layer = new LayerNormalizationLayer<double>(featureSize);

        var input = new Tensor<double>(new[] { 1, featureSize });
        for (int i = 0; i < featureSize; i++) input[0, i] = 5.0;

        var output = layer.Forward(input);

        for (int i = 0; i < featureSize; i++)
        {
            Assert.Equal(0.0, NumOps<double>.ToDouble(output[0, i]), 1e-4);
        }
    }

    [Fact]
    public void LayerNorm_HandComputed_SingleSample()
    {
        // Input: [2, 4, 6, 8]
        // mean = (2+4+6+8)/4 = 5
        // variance = ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2)/4
        //          = (9 + 1 + 1 + 9)/4 = 20/4 = 5
        // normalized[i] = (x[i] - 5) / sqrt(5 + eps)
        // With gamma=1, beta=0:
        // output[0] = (2-5)/sqrt(5+eps) = -3/sqrt(5) ≈ -1.3416
        // output[1] = (4-5)/sqrt(5+eps) = -1/sqrt(5) ≈ -0.4472
        // output[2] = (6-5)/sqrt(5+eps) = 1/sqrt(5) ≈ 0.4472
        // output[3] = (8-5)/sqrt(5+eps) = 3/sqrt(5) ≈ 1.3416
        int featureSize = 4;
        var layer = new LayerNormalizationLayer<double>(featureSize);

        var input = new Tensor<double>(new[] { 1, featureSize });
        input[0, 0] = 2.0;
        input[0, 1] = 4.0;
        input[0, 2] = 6.0;
        input[0, 3] = 8.0;

        var output = layer.Forward(input);

        double eps = 1e-5; // default epsilon
        double mean = 5.0;
        double variance = 5.0;
        double invStd = 1.0 / Math.Sqrt(variance + eps);

        Assert.Equal((2.0 - mean) * invStd, NumOps<double>.ToDouble(output[0, 0]), 1e-3);
        Assert.Equal((4.0 - mean) * invStd, NumOps<double>.ToDouble(output[0, 1]), 1e-3);
        Assert.Equal((6.0 - mean) * invStd, NumOps<double>.ToDouble(output[0, 2]), 1e-3);
        Assert.Equal((8.0 - mean) * invStd, NumOps<double>.ToDouble(output[0, 3]), 1e-3);
    }

    [Fact]
    public void LayerNorm_OutputMeanIsZero()
    {
        // Layer normalization should produce output with mean ≈ 0 (when beta=0)
        int featureSize = 5;
        var layer = new LayerNormalizationLayer<double>(featureSize);

        var input = new Tensor<double>(new[] { 1, featureSize });
        input[0, 0] = 1.0;
        input[0, 1] = 3.0;
        input[0, 2] = 7.0;
        input[0, 3] = 2.0;
        input[0, 4] = 5.0;

        var output = layer.Forward(input);

        double sum = 0;
        for (int i = 0; i < featureSize; i++)
            sum += NumOps<double>.ToDouble(output[0, i]);

        Assert.Equal(0.0, sum / featureSize, 1e-5);
    }

    [Fact]
    public void LayerNorm_OutputVarianceIsOne()
    {
        // Layer normalization should produce output with variance ≈ 1 (when gamma=1)
        int featureSize = 5;
        var layer = new LayerNormalizationLayer<double>(featureSize);

        var input = new Tensor<double>(new[] { 1, featureSize });
        input[0, 0] = 1.0;
        input[0, 1] = 3.0;
        input[0, 2] = 7.0;
        input[0, 3] = 2.0;
        input[0, 4] = 5.0;

        var output = layer.Forward(input);

        // Compute variance of output
        double sum = 0, sumSq = 0;
        for (int i = 0; i < featureSize; i++)
        {
            double v = NumOps<double>.ToDouble(output[0, i]);
            sum += v;
            sumSq += v * v;
        }
        double mean = sum / featureSize;
        double variance = sumSq / featureSize - mean * mean;

        Assert.Equal(1.0, variance, 1e-3);
    }

    [Fact]
    public void LayerNorm_BatchIndependence()
    {
        // Layer norm normalizes each sample independently
        // So changing one sample should not affect another's output
        int featureSize = 3;
        var layer = new LayerNormalizationLayer<double>(featureSize);

        var input1 = new Tensor<double>(new[] { 2, featureSize });
        input1[0, 0] = 1.0; input1[0, 1] = 2.0; input1[0, 2] = 3.0;
        input1[1, 0] = 4.0; input1[1, 1] = 5.0; input1[1, 2] = 6.0;

        var output1 = layer.Forward(input1);
        double sample0_dim0_v1 = NumOps<double>.ToDouble(output1[0, 0]);

        // Change sample 1, keep sample 0 same
        var input2 = new Tensor<double>(new[] { 2, featureSize });
        input2[0, 0] = 1.0; input2[0, 1] = 2.0; input2[0, 2] = 3.0;
        input2[1, 0] = 100.0; input2[1, 1] = 200.0; input2[1, 2] = 300.0;

        var output2 = layer.Forward(input2);
        double sample0_dim0_v2 = NumOps<double>.ToDouble(output2[0, 0]);

        // Sample 0's output should be the same regardless of sample 1
        Assert.Equal(sample0_dim0_v1, sample0_dim0_v2, 1e-5);
    }

    [Fact]
    public void LayerNorm_WithCustomGammaBeta()
    {
        // Set gamma=2, beta=1, then output = 2 * normalized + 1
        int featureSize = 4;
        var layer = new LayerNormalizationLayer<double>(featureSize);

        // Set gamma=2, beta=1 for all features
        var gammaValues = new double[featureSize];
        var betaValues = new double[featureSize];
        for (int i = 0; i < featureSize; i++)
        {
            gammaValues[i] = 2.0;
            betaValues[i] = 1.0;
        }

        // Parameters: [gamma_0..gamma_n, beta_0..beta_n]
        var parameters = new Vector<double>(featureSize * 2);
        for (int i = 0; i < featureSize; i++)
        {
            parameters[i] = gammaValues[i];
            parameters[featureSize + i] = betaValues[i];
        }
        layer.SetParameters(parameters);

        var input = new Tensor<double>(new[] { 1, featureSize });
        input[0, 0] = 2.0;
        input[0, 1] = 4.0;
        input[0, 2] = 6.0;
        input[0, 3] = 8.0;

        var output = layer.Forward(input);

        double eps = 1e-5;
        double mean = 5.0;
        double variance = 5.0;
        double invStd = 1.0 / Math.Sqrt(variance + eps);

        // output = 2 * normalized + 1
        for (int i = 0; i < featureSize; i++)
        {
            double x = NumOps<double>.ToDouble(input[0, i]);
            double normalized = (x - mean) * invStd;
            double expected = 2.0 * normalized + 1.0;
            Assert.Equal(expected, NumOps<double>.ToDouble(output[0, i]), 1e-3);
        }
    }

    // ========================================================================
    // FullyConnectedLayer - Forward pass math verification
    // ========================================================================

    [Fact]
    public void FullyConnectedLayer_Forward_MatMulPlusBias()
    {
        // FC layer: output = input * W^T + bias
        // With inputSize=2, outputSize=3
        // Input: [1, 2] (batch=1)
        // W: [3, 2] (outputSize x inputSize)
        // bias: [3]
        var layer = new FullyConnectedLayer<double>(2, 3, (IActivationFunction<double>?)null);

        // Set known weights and biases
        // Parameters: [weights (3*2=6), biases (3)] = 9 total
        var parameters = new Vector<double>(9);
        // Weights [3x2]: w[0,0]=1, w[0,1]=2, w[1,0]=3, w[1,1]=4, w[2,0]=5, w[2,1]=6
        parameters[0] = 1.0; parameters[1] = 2.0;  // output neuron 0
        parameters[2] = 3.0; parameters[3] = 4.0;  // output neuron 1
        parameters[4] = 5.0; parameters[5] = 6.0;  // output neuron 2
        // Biases: b[0]=0.1, b[1]=0.2, b[2]=0.3
        parameters[6] = 0.1;
        parameters[7] = 0.2;
        parameters[8] = 0.3;
        layer.SetParameters(parameters);

        var input = new Tensor<double>(new[] { 1, 2 });
        input[0, 0] = 1.0;
        input[0, 1] = 2.0;

        var output = layer.Forward(input);

        // output[0] = 1*1 + 2*2 + 0.1 = 1 + 4 + 0.1 = 5.1
        // output[1] = 3*1 + 4*2 + 0.2 = 3 + 8 + 0.2 = 11.2
        // output[2] = 5*1 + 6*2 + 0.3 = 5 + 12 + 0.3 = 17.3
        Assert.Equal(5.1, NumOps<double>.ToDouble(output[0, 0]), 1e-4);
        Assert.Equal(11.2, NumOps<double>.ToDouble(output[0, 1]), 1e-4);
        Assert.Equal(17.3, NumOps<double>.ToDouble(output[0, 2]), 1e-4);
    }

    [Fact]
    public void FullyConnectedLayer_Forward_BatchProcessing()
    {
        // Test with batch size 2
        var layer = new FullyConnectedLayer<double>(2, 2, (IActivationFunction<double>?)null);

        var parameters = new Vector<double>(6);
        // Weights [2x2]: identity matrix
        parameters[0] = 1.0; parameters[1] = 0.0; // neuron 0
        parameters[2] = 0.0; parameters[3] = 1.0; // neuron 1
        // Biases: [0, 0]
        parameters[4] = 0.0;
        parameters[5] = 0.0;
        layer.SetParameters(parameters);

        var input = new Tensor<double>(new[] { 2, 2 });
        input[0, 0] = 3.0; input[0, 1] = 4.0;
        input[1, 0] = 5.0; input[1, 1] = 6.0;

        var output = layer.Forward(input);

        // Identity weights + zero bias → output = input
        Assert.Equal(3.0, NumOps<double>.ToDouble(output[0, 0]), 1e-5);
        Assert.Equal(4.0, NumOps<double>.ToDouble(output[0, 1]), 1e-5);
        Assert.Equal(5.0, NumOps<double>.ToDouble(output[1, 0]), 1e-5);
        Assert.Equal(6.0, NumOps<double>.ToDouble(output[1, 1]), 1e-5);
    }

    [Fact]
    public void FullyConnectedLayer_ParameterCount()
    {
        // FC layer with inputSize=3, outputSize=2 should have 3*2 + 2 = 8 parameters
        var layer = new FullyConnectedLayer<double>(3, 2, (IActivationFunction<double>?)null);
        Assert.Equal(8, layer.ParameterCount);
    }

    // ========================================================================
    // DropoutLayer - Inference mode passes through
    // ========================================================================

    [Fact]
    public void DropoutLayer_InferenceMode_PassesThrough()
    {
        // In inference mode (not training), dropout should pass all values through
        var layer = new DropoutLayer<double>(dropoutRate: 0.5);
        layer.SetTrainingMode(false); // inference mode

        var input = new Tensor<double>(new[] { 1, 4 });
        input[0, 0] = 1.0;
        input[0, 1] = 2.0;
        input[0, 2] = 3.0;
        input[0, 3] = 4.0;

        var output = layer.Forward(input);

        Assert.Equal(1.0, NumOps<double>.ToDouble(output[0, 0]), Tol);
        Assert.Equal(2.0, NumOps<double>.ToDouble(output[0, 1]), Tol);
        Assert.Equal(3.0, NumOps<double>.ToDouble(output[0, 2]), Tol);
        Assert.Equal(4.0, NumOps<double>.ToDouble(output[0, 3]), Tol);
    }

    [Fact]
    public void DropoutLayer_TrainingMode_ScalesOutput()
    {
        // In training mode, dropout drops some values and scales remaining by 1/(1-rate)
        // With rate=0.5, surviving values should be scaled by 2.0
        var layer = new DropoutLayer<double>(dropoutRate: 0.5);
        layer.SetTrainingMode(true);

        var input = new Tensor<double>(new[] { 1, 100 });
        for (int i = 0; i < 100; i++) input[0, i] = 1.0;

        var output = layer.Forward(input);

        // Check: each output value is either 0 (dropped) or 2.0 (scaled by 1/0.5)
        int zeros = 0;
        int scaled = 0;
        for (int i = 0; i < 100; i++)
        {
            double v = NumOps<double>.ToDouble(output[0, i]);
            if (Math.Abs(v) < 1e-10)
                zeros++;
            else if (Math.Abs(v - 2.0) < 1e-10)
                scaled++;
        }

        // Roughly half should be zero, half should be 2.0
        Assert.True(zeros + scaled == 100, "All outputs should be either 0 or 2.0");
        Assert.True(zeros > 10, $"Expected some zeros, got {zeros}");
        Assert.True(scaled > 10, $"Expected some scaled values, got {scaled}");
    }

    [Fact]
    public void DropoutLayer_Rate0_NothingDropped()
    {
        // With rate=0, nothing should be dropped
        var layer = new DropoutLayer<double>(dropoutRate: 0.0);
        layer.SetTrainingMode(true);

        var input = new Tensor<double>(new[] { 1, 10 });
        for (int i = 0; i < 10; i++) input[0, i] = (double)(i + 1);

        var output = layer.Forward(input);

        for (int i = 0; i < 10; i++)
        {
            Assert.Equal((double)(i + 1), NumOps<double>.ToDouble(output[0, i]), Tol);
        }
    }

    // ========================================================================
    // FlattenLayer - Shape transformation
    // ========================================================================

    [Fact]
    public void FlattenLayer_PreservesValues()
    {
        // FlattenLayer should reshape [batch, h, w] -> [batch, h*w]
        var layer = new FlattenLayer<double>(new[] { 2, 3 }); // 2x3 input

        var input = new Tensor<double>(new[] { 1, 2, 3 });
        input[0, 0, 0] = 1.0; input[0, 0, 1] = 2.0; input[0, 0, 2] = 3.0;
        input[0, 1, 0] = 4.0; input[0, 1, 1] = 5.0; input[0, 1, 2] = 6.0;

        var output = layer.Forward(input);

        // Output should be [1, 6] preserving all values
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(6, output.Shape[1]);

        Assert.Equal(1.0, NumOps<double>.ToDouble(output[0, 0]), Tol);
        Assert.Equal(2.0, NumOps<double>.ToDouble(output[0, 1]), Tol);
        Assert.Equal(3.0, NumOps<double>.ToDouble(output[0, 2]), Tol);
        Assert.Equal(4.0, NumOps<double>.ToDouble(output[0, 3]), Tol);
        Assert.Equal(5.0, NumOps<double>.ToDouble(output[0, 4]), Tol);
        Assert.Equal(6.0, NumOps<double>.ToDouble(output[0, 5]), Tol);
    }

    // ========================================================================
    // EmbeddingLayer - Lookup verification
    // ========================================================================

    [Fact]
    public void EmbeddingLayer_Lookup_CorrectRowsReturned()
    {
        // EmbeddingLayer maps indices to embedding vectors
        int vocabSize = 5;
        int embeddingDim = 3;
        var layer = new EmbeddingLayer<double>(vocabSize, embeddingDim);

        // Set known embeddings: row i has values [i*10, i*10+1, i*10+2]
        var parameters = new Vector<double>(vocabSize * embeddingDim);
        for (int i = 0; i < vocabSize; i++)
        {
            for (int j = 0; j < embeddingDim; j++)
            {
                parameters[i * embeddingDim + j] = i * 10.0 + j;
            }
        }
        layer.SetParameters(parameters);

        // Lookup index 2 and 4 using 1D input [seqLen=2]
        var input = new Tensor<double>(new[] { 2 });
        input[0] = 2.0;
        input[1] = 4.0;

        var output = layer.Forward(input);
        // 1D input [2] -> output [2, embeddingDim=3]

        // Row 0 should be embedding for index 2: [20, 21, 22]
        Assert.Equal(20.0, NumOps<double>.ToDouble(output[0, 0]), Tol);
        Assert.Equal(21.0, NumOps<double>.ToDouble(output[0, 1]), Tol);
        Assert.Equal(22.0, NumOps<double>.ToDouble(output[0, 2]), Tol);

        // Row 1 should be embedding for index 4: [40, 41, 42]
        Assert.Equal(40.0, NumOps<double>.ToDouble(output[1, 0]), Tol);
        Assert.Equal(41.0, NumOps<double>.ToDouble(output[1, 1]), Tol);
        Assert.Equal(42.0, NumOps<double>.ToDouble(output[1, 2]), Tol);
    }

    // ========================================================================
    // Helper to access NumOps for assertions
    // ========================================================================

    private static class NumOps<TNum>
    {
        private static readonly INumericOperations<TNum> Ops =
            AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<TNum>();

        public static double ToDouble(TNum value) => Ops.ToDouble(value);
    }
}
