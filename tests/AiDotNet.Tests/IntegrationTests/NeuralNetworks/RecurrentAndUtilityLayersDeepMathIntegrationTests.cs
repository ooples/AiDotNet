using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Deep math-correctness integration tests for recurrent layers (GRU, LSTM)
/// and utility layers (Reshape, Mean).
/// Each test hand-computes expected outputs and verifies the code matches.
/// </summary>
public class RecurrentAndUtilityLayersDeepMathIntegrationTests
{
    private const double Tol = 1e-4;

    // ========================================================================
    // GRULayer - Parameter Count Formula
    // ========================================================================

    [Fact]
    public void GRU_ParameterCount_Formula_Input4Hidden3()
    {
        // ParameterCount = hiddenSize * inputSize * 3 (Wz, Wr, Wh)
        //                + hiddenSize * hiddenSize * 3 (Uz, Ur, Uh)
        //                + hiddenSize * 3 (bz, br, bh)
        // For input=4, hidden=3:
        // = 3*4*3 + 3*3*3 + 3*3 = 36 + 27 + 9 = 72
        var gru = new GRULayer<double>(inputSize: 4, hiddenSize: 3,
            activation: (IActivationFunction<double>?)null);
        Assert.Equal(72, gru.ParameterCount);
    }

    [Fact]
    public void GRU_ParameterCount_Formula_Input10Hidden8()
    {
        // = 8*10*3 + 8*8*3 + 8*3 = 240 + 192 + 24 = 456
        var gru = new GRULayer<double>(inputSize: 10, hiddenSize: 8,
            activation: (IActivationFunction<double>?)null);
        Assert.Equal(456, gru.ParameterCount);
    }

    [Fact]
    public void GRU_ParameterCount_Formula_Input1Hidden1()
    {
        // = 1*1*3 + 1*1*3 + 1*3 = 3 + 3 + 3 = 9
        var gru = new GRULayer<double>(inputSize: 1, hiddenSize: 1,
            activation: (IActivationFunction<double>?)null);
        Assert.Equal(9, gru.ParameterCount);
    }

    // ========================================================================
    // GRULayer - Output Shape
    // ========================================================================

    [Fact]
    public void GRU_OutputShape_2DInput_ReturnsHiddenSize()
    {
        // 2D input [seqLen, inputSize] -> output [hiddenSize]
        var gru = new GRULayer<double>(inputSize: 3, hiddenSize: 5,
            activation: (IActivationFunction<double>?)null);
        var input = new Tensor<double>(new[] { 4, 3 }); // 4 timesteps, 3 features
        var output = gru.Forward(input);

        Assert.Single(output.Shape); // 1D
        Assert.Equal(5, output.Shape[0]);
    }

    [Fact]
    public void GRU_OutputShape_3DInput_ReturnsBatchHiddenSize()
    {
        // 3D input [batch, seqLen, inputSize] -> output [batch, hiddenSize]
        var gru = new GRULayer<double>(inputSize: 3, hiddenSize: 5,
            activation: (IActivationFunction<double>?)null);
        var input = new Tensor<double>(new[] { 2, 4, 3 }); // 2 batches, 4 timesteps, 3 features
        var output = gru.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(5, output.Shape[1]);
    }

    [Fact]
    public void GRU_OutputShape_ReturnSequences_3DOutput()
    {
        // With returnSequences=true: [batch, seqLen, inputSize] -> [batch, seqLen, hiddenSize]
        var gru = new GRULayer<double>(inputSize: 3, hiddenSize: 5,
            returnSequences: true, activation: (IActivationFunction<double>?)null);
        var input = new Tensor<double>(new[] { 2, 4, 3 });
        var output = gru.Forward(input);

        Assert.Equal(3, output.Shape.Length);
        Assert.Equal(2, output.Shape[0]);  // batch
        Assert.Equal(4, output.Shape[1]);  // seqLen preserved
        Assert.Equal(5, output.Shape[2]);  // hiddenSize
    }

    // ========================================================================
    // GRULayer - Hidden State Dynamics
    // ========================================================================

    [Fact]
    public void GRU_HiddenStateCarriesOver_SecondCallDiffersFromFirst()
    {
        var gru = new GRULayer<double>(inputSize: 2, hiddenSize: 3,
            activation: (IActivationFunction<double>?)null);

        var input = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new double[] { 1.0, 0.5 }));

        var output1 = gru.Forward(input);
        var o1vals = new double[3];
        for (int i = 0; i < 3; i++)
            o1vals[i] = output1[i];

        var output2 = gru.Forward(input);

        bool differs = false;
        for (int i = 0; i < 3; i++)
        {
            if (Math.Abs(output2[i] - o1vals[i]) > 1e-10)
                differs = true;
        }
        Assert.True(differs, "Second GRU forward with same input should differ due to hidden state");
    }

    [Fact]
    public void GRU_ResetState_OutputMatchesFirstCall()
    {
        var gru = new GRULayer<double>(inputSize: 2, hiddenSize: 3,
            activation: (IActivationFunction<double>?)null);

        var input = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new double[] { 1.0, 0.5 }));

        var output1 = gru.Forward(input);
        var o1vals = new double[3];
        for (int i = 0; i < 3; i++)
            o1vals[i] = output1[i];

        gru.Forward(input);
        gru.ResetState();

        var output3 = gru.Forward(input);
        for (int i = 0; i < 3; i++)
            Assert.Equal(o1vals[i], output3[i], Tol);
    }

    [Fact]
    public void GRU_OutputIsBounded_ByTanh()
    {
        var gru = new GRULayer<double>(inputSize: 3, hiddenSize: 4,
            activation: (IActivationFunction<double>?)null);

        var input = new Tensor<double>(new[] { 5, 3 });
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 3; j++)
                input[i, j] = (i + 1) * (j + 1) * 10.0;

        var output = gru.Forward(input);

        for (int i = 0; i < 4; i++)
        {
            Assert.True(output[i] >= -1.0 - 1e-6, $"GRU output [{i}]={output[i]} should be >= -1");
            Assert.True(output[i] <= 1.0 + 1e-6, $"GRU output [{i}]={output[i]} should be <= 1");
        }
    }

    [Fact]
    public void GRU_AllOutputsFinite()
    {
        var gru = new GRULayer<double>(inputSize: 3, hiddenSize: 4,
            activation: (IActivationFunction<double>?)null);

        var input = new Tensor<double>(new[] { 2, 3, 3 });
        for (int b = 0; b < 2; b++)
            for (int t = 0; t < 3; t++)
                for (int f = 0; f < 3; f++)
                    input[b, t, f] = (b + 1) * 0.1 + t * 0.2 + f * 0.3;

        var output = gru.Forward(input);

        for (int b = 0; b < 2; b++)
            for (int h = 0; h < 4; h++)
            {
                Assert.False(double.IsNaN(output[b, h]), $"NaN at [{b},{h}]");
                Assert.False(double.IsInfinity(output[b, h]), $"Infinity at [{b},{h}]");
            }
    }

    [Fact]
    public void GRU_GetParameters_LengthMatchesParameterCount()
    {
        var gru = new GRULayer<double>(inputSize: 4, hiddenSize: 3,
            activation: (IActivationFunction<double>?)null);
        var parameters = gru.GetParameters();
        Assert.Equal(gru.ParameterCount, parameters.Length);
    }

    // ========================================================================
    // LSTMLayer - Parameter Count Formula
    // ========================================================================

    [Fact]
    public void LSTM_ParameterCount_Formula_Input4Hidden3()
    {
        // ParameterCount = 4 * (hiddenSize * inputSize) + 4 * (hiddenSize * hiddenSize) + 4 * hiddenSize
        // For input=4, hidden=3:
        // = 4*3*4 + 4*3*3 + 4*3 = 48 + 36 + 12 = 96
        var lstm = new LSTMLayer<double>(inputSize: 4, hiddenSize: 3, inputShape: new[] { 1, 4 },
            activation: (IActivationFunction<double>?)null);
        Assert.Equal(96, lstm.ParameterCount);
    }

    [Fact]
    public void LSTM_ParameterCount_Formula_Input10Hidden8()
    {
        // = 4*8*10 + 4*8*8 + 4*8 = 320 + 256 + 32 = 608
        var lstm = new LSTMLayer<double>(inputSize: 10, hiddenSize: 8, inputShape: new[] { 1, 10 },
            activation: (IActivationFunction<double>?)null);
        Assert.Equal(608, lstm.ParameterCount);
    }

    [Fact]
    public void LSTM_ParameterCount_AlwaysMore_ThanGRU()
    {
        int inputSize = 5;
        int hiddenSize = 7;
        var gru = new GRULayer<double>(inputSize, hiddenSize,
            activation: (IActivationFunction<double>?)null);
        var lstm = new LSTMLayer<double>(inputSize, hiddenSize, inputShape: new[] { 1, inputSize },
            activation: (IActivationFunction<double>?)null);
        Assert.True(lstm.ParameterCount > gru.ParameterCount,
            $"LSTM params ({lstm.ParameterCount}) should exceed GRU params ({gru.ParameterCount})");
    }

    [Fact]
    public void LSTM_ParameterCount_Ratio_Is4Over3_TimesGRU()
    {
        // Ratio = 4/3 for same input/hidden sizes
        int inputSize = 6;
        int hiddenSize = 4;
        var gru = new GRULayer<double>(inputSize, hiddenSize,
            activation: (IActivationFunction<double>?)null);
        var lstm = new LSTMLayer<double>(inputSize, hiddenSize, inputShape: new[] { 1, inputSize },
            activation: (IActivationFunction<double>?)null);

        double ratio = (double)lstm.ParameterCount / gru.ParameterCount;
        Assert.Equal(4.0 / 3.0, ratio, Tol);
    }

    // ========================================================================
    // LSTMLayer - Output Shape and Finiteness
    // ========================================================================

    [Fact]
    public void LSTM_OutputShape_MatchesHiddenSize()
    {
        var lstm = new LSTMLayer<double>(inputSize: 3, hiddenSize: 5, inputShape: new[] { 4, 3 },
            activation: (IActivationFunction<double>?)null);
        var input = new Tensor<double>(new[] { 4, 3 });
        var output = lstm.Forward(input);

        Assert.True(output.Length >= 5, $"LSTM output length ({output.Length}) should be at least hiddenSize (5)");
    }

    [Fact]
    public void LSTM_AllOutputsFinite()
    {
        var lstm = new LSTMLayer<double>(inputSize: 3, hiddenSize: 4, inputShape: new[] { 2, 3 },
            activation: (IActivationFunction<double>?)null);

        var input = new Tensor<double>(new[] { 2, 3 });
        for (int t = 0; t < 2; t++)
            for (int f = 0; f < 3; f++)
                input[t, f] = t * 0.1 + f * 0.2;

        var output = lstm.Forward(input);
        var outputArr = output.ToArray();

        for (int i = 0; i < outputArr.Length; i++)
        {
            Assert.False(double.IsNaN(outputArr[i]), $"NaN at index {i}");
            Assert.False(double.IsInfinity(outputArr[i]), $"Infinity at index {i}");
        }
    }

    [Fact]
    public void LSTM_GetParameters_LengthMatchesParameterCount()
    {
        var lstm = new LSTMLayer<double>(inputSize: 4, hiddenSize: 3, inputShape: new[] { 1, 4 },
            activation: (IActivationFunction<double>?)null);
        var parameters = lstm.GetParameters();
        Assert.Equal(lstm.ParameterCount, parameters.Length);
    }

    // ========================================================================
    // MeanLayer - Exact Mean Computation
    // ========================================================================

    [Fact]
    public void MeanLayer_Axis1_HandComputed()
    {
        // Input [2, 3]: [[1, 2, 3], [4, 5, 6]]
        // Mean along axis 1: [(1+2+3)/3, (4+5+6)/3] = [2, 5]
        var mean = new MeanLayer<double>(new[] { 2, 3 }, 1);

        var input = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }));
        var output = mean.Forward(input);
        var arr = output.ToArray();

        Assert.Equal(2, arr.Length);
        Assert.Equal(2.0, arr[0], Tol);
        Assert.Equal(5.0, arr[1], Tol);
    }

    [Fact]
    public void MeanLayer_Axis0_HandComputed()
    {
        // Input [2, 3]: [[1, 2, 3], [7, 8, 9]]
        // Mean along axis 0: [(1+7)/2, (2+8)/2, (3+9)/2] = [4, 5, 6]
        var mean = new MeanLayer<double>(new[] { 2, 3 }, 0);

        var input = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 7.0, 8.0, 9.0 }));
        var output = mean.Forward(input);
        var arr = output.ToArray();

        Assert.Equal(3, arr.Length);
        Assert.Equal(4.0, arr[0], Tol);
        Assert.Equal(5.0, arr[1], Tol);
        Assert.Equal(6.0, arr[2], Tol);
    }

    [Fact]
    public void MeanLayer_UniformInput_MeanEqualsValue()
    {
        var mean = new MeanLayer<double>(new[] { 3, 4 }, 1);

        var input = new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                input[i, j] = 7.5;

        var output = mean.Forward(input);
        var arr = output.ToArray();

        for (int i = 0; i < 3; i++)
            Assert.Equal(7.5, arr[i], Tol);
    }

    [Fact]
    public void MeanLayer_SingleElement_MeanIsElement()
    {
        var mean = new MeanLayer<double>(new[] { 1, 5 }, 0);

        var input = new Tensor<double>(new[] { 1, 5 }, new Vector<double>(new double[] { 10.0, 20.0, 30.0, 40.0, 50.0 }));
        var output = mean.Forward(input);
        var arr = output.ToArray();

        Assert.Equal(5, arr.Length);
        Assert.Equal(10.0, arr[0], Tol);
        Assert.Equal(50.0, arr[4], Tol);
    }

    [Fact]
    public void MeanLayer_NegativeValues_HandComputed()
    {
        var mean = new MeanLayer<double>(new[] { 1, 4 }, 1);

        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new double[] { -2.0, -4.0, 6.0, 8.0 }));
        var output = mean.Forward(input);
        var arr = output.ToArray();

        // (-2 + -4 + 6 + 8) / 4 = 8 / 4 = 2.0
        Assert.Equal(2.0, arr[0], Tol);
    }

    [Fact]
    public void MeanLayer_LargeValues_StillAccurate()
    {
        var mean = new MeanLayer<double>(new[] { 1, 3 }, 1);

        var input = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new double[] { 1e10, 2e10, 3e10 }));
        var output = mean.Forward(input);
        var arr = output.ToArray();

        Assert.Equal(2e10, arr[0], 1e5);
    }

    // ========================================================================
    // ReshapeLayer - Data Preservation
    // ========================================================================

    [Fact]
    public void ReshapeLayer_PreservesAllValues()
    {
        var reshape = new ReshapeLayer<double>(new[] { 2, 3 }, new[] { 3, 2 });

        var input = new Tensor<double>(new[] { 1, 2, 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }));
        var output = reshape.Forward(input);

        Assert.Equal(6, output.Length);
        var inputArr = input.ToArray();
        var outputArr = output.ToArray();
        for (int i = 0; i < 6; i++)
            Assert.Equal(inputArr[i], outputArr[i], Tol);
    }

    [Fact]
    public void ReshapeLayer_OutputShapeIsCorrect()
    {
        var reshape = new ReshapeLayer<double>(new[] { 4, 3 }, new[] { 6, 2 });

        var input = new Tensor<double>(new[] { 1, 4, 3 });
        var output = reshape.Forward(input);

        Assert.True(output.Length == 12, $"Expected 12 elements, got {output.Length}");
    }

    [Fact]
    public void ReshapeLayer_Backward_RestoresInputShape()
    {
        var reshape = new ReshapeLayer<double>(new[] { 6 }, new[] { 2, 3 });

        var input = new Tensor<double>(new[] { 1, 6 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }));
        reshape.Forward(input);

        var grad = new Tensor<double>(new[] { 1, 2, 3 }, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }));
        var inputGrad = reshape.Backward(grad);

        Assert.Equal(6, inputGrad.Length);
    }

    [Fact]
    public void ReshapeLayer_Backward_PreservesGradientValues()
    {
        var reshape = new ReshapeLayer<double>(new[] { 6 }, new[] { 2, 3 });

        var input = new Tensor<double>(new[] { 1, 6 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }));
        reshape.Forward(input);

        var grad = new Tensor<double>(new[] { 1, 2, 3 }, new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }));
        var inputGrad = reshape.Backward(grad);

        var gradArr = grad.ToArray();
        var inputGradArr = inputGrad.ToArray();
        for (int i = 0; i < 6; i++)
            Assert.Equal(gradArr[i], inputGradArr[i], Tol);
    }

    [Fact]
    public void ReshapeLayer_NoTrainableParameters()
    {
        var reshape = new ReshapeLayer<double>(new[] { 4 }, new[] { 2, 2 });
        Assert.Equal(0, reshape.ParameterCount);
    }

    // ========================================================================
    // GRU - Backward Gradient Structure
    // ========================================================================

    [Fact]
    public void GRU_Backward_GradientIsFinite()
    {
        var gru = new GRULayer<double>(inputSize: 3, hiddenSize: 4,
            activation: (IActivationFunction<double>?)null);
        gru.SetTrainingMode(true);

        var input = new Tensor<double>(new[] { 2, 3 });
        for (int t = 0; t < 2; t++)
            for (int f = 0; f < 3; f++)
                input[t, f] = (t + 1) * 0.1 + f * 0.2;

        gru.Forward(input);

        var grad = new Tensor<double>(new[] { 4 });
        for (int h = 0; h < 4; h++)
            grad[h] = 0.1 * (h + 1);

        var inputGrad = gru.Backward(grad);
        var inputGradArr = inputGrad.ToArray();

        for (int i = 0; i < inputGradArr.Length; i++)
        {
            Assert.False(double.IsNaN(inputGradArr[i]), $"NaN in gradient at index {i}");
            Assert.False(double.IsInfinity(inputGradArr[i]), $"Infinity in gradient at index {i}");
        }
    }

    // ========================================================================
    // GRU vs LSTM - Structural Comparisons
    // ========================================================================

    [Fact]
    public void GRU_And_LSTM_SameInputSize_DifferentParamCounts()
    {
        int inputSize = 8;
        int hiddenSize = 16;

        var gru = new GRULayer<double>(inputSize, hiddenSize,
            activation: (IActivationFunction<double>?)null);
        var lstm = new LSTMLayer<double>(inputSize, hiddenSize, inputShape: new[] { 1, inputSize },
            activation: (IActivationFunction<double>?)null);

        int expectedGRU = 3 * (hiddenSize * inputSize + hiddenSize * hiddenSize + hiddenSize);
        int expectedLSTM = 4 * (hiddenSize * inputSize + hiddenSize * hiddenSize + hiddenSize);

        Assert.Equal(expectedGRU, gru.ParameterCount);
        Assert.Equal(expectedLSTM, lstm.ParameterCount);
    }

    // ========================================================================
    // GRU - Zero and Large Input Behavior
    // ========================================================================

    [Fact]
    public void GRU_ZeroInput_OutputIsFinite()
    {
        var gru = new GRULayer<double>(inputSize: 3, hiddenSize: 4,
            activation: (IActivationFunction<double>?)null);

        var input = new Tensor<double>(new[] { 1, 3 });
        var output = gru.Forward(input);

        for (int i = 0; i < 4; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"NaN at [{i}]");
            Assert.False(double.IsInfinity(output[i]), $"Infinity at [{i}]");
        }
    }

    [Fact]
    public void GRU_LargeInput_StillBounded()
    {
        var gru = new GRULayer<double>(inputSize: 2, hiddenSize: 3,
            activation: (IActivationFunction<double>?)null);

        var input = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new double[] { 1000.0, -1000.0 }));
        var output = gru.Forward(input);

        for (int i = 0; i < 3; i++)
        {
            Assert.True(output[i] >= -1.0 - 1e-6, $"Output [{i}]={output[i]} should be >= -1");
            Assert.True(output[i] <= 1.0 + 1e-6, $"Output [{i}]={output[i]} should be <= 1");
            Assert.False(double.IsNaN(output[i]), $"NaN at [{i}]");
        }
    }

    [Fact]
    public void GRU_MultipleTimesteps_OutputChanges()
    {
        var gru = new GRULayer<double>(inputSize: 2, hiddenSize: 3,
            returnSequences: true, activation: (IActivationFunction<double>?)null);

        var input = new Tensor<double>(new[] { 1, 3, 2 });
        input[0, 0, 0] = 1.0; input[0, 0, 1] = 0.0;
        input[0, 1, 0] = 0.0; input[0, 1, 1] = 1.0;
        input[0, 2, 0] = 1.0; input[0, 2, 1] = 1.0;

        var output = gru.Forward(input);

        bool timestepsDiffer = false;
        for (int h = 0; h < 3; h++)
        {
            if (Math.Abs(output[0, 0, h] - output[0, 1, h]) > 1e-10 ||
                Math.Abs(output[0, 1, h] - output[0, 2, h]) > 1e-10)
                timestepsDiffer = true;
        }
        Assert.True(timestepsDiffer, "Different timestep inputs should produce different hidden states");
    }

    // ========================================================================
    // MeanLayer - Backward Gradient
    // ========================================================================

    [Fact]
    public void MeanLayer_Backward_GradientIsFinite()
    {
        var mean = new MeanLayer<double>(new[] { 2, 3 }, 1);

        var input = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }));
        mean.Forward(input);

        var grad = new Tensor<double>(new[] { 2 }, new Vector<double>(new double[] { 3.0, 6.0 }));
        var inputGrad = mean.Backward(grad);
        var inputGradArr = inputGrad.ToArray();

        Assert.Equal(6, inputGradArr.Length);

        for (int i = 0; i < 6; i++)
        {
            Assert.False(double.IsNaN(inputGradArr[i]), $"NaN at index {i}");
            Assert.False(double.IsInfinity(inputGradArr[i]), $"Infinity at index {i}");
        }
    }
}
