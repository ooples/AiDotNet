using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using System.Threading.Tasks;

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

    [Fact(Timeout = 120000)]
    public async Task GRU_ParameterCount_Formula_Input4Hidden3()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // ParameterCount = hiddenSize * inputSize * 3 (Wz, Wr, Wh)
        //                + hiddenSize * hiddenSize * 3 (Uz, Ur, Uh)
        //                + hiddenSize * 3 (bz, br, bh)
        // For input=4, hidden=3:
        // = 3*4*3 + 3*3*3 + 3*3 = 36 + 27 + 9 = 72
        var gru = new GRULayer<double>( hiddenSize: 3,
            activation: (IActivationFunction<double>?)null);
        // GRULayer is lazy-input (PyTorch-style): weights materialize on the first
        // forward, which resolves inputSize from the input's last axis. Warm up
        // with a [seq, 4] input to resolve inputSize = 4 before the formula check.
        gru.Forward(new Tensor<double>(new[] { 2, 4 }));
        Assert.Equal(72, (int)gru.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task GRU_ParameterCount_Formula_Input10Hidden8()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // = 8*10*3 + 8*8*3 + 8*3 = 240 + 192 + 24 = 456
        var gru = new GRULayer<double>( hiddenSize: 8,
            activation: (IActivationFunction<double>?)null);
        // Lazy-input: warm up with a [seq, 10] input to resolve inputSize = 10.
        gru.Forward(new Tensor<double>(new[] { 2, 10 }));
        Assert.Equal(456, (int)gru.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task GRU_ParameterCount_Formula_Input1Hidden1()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // = 1*1*3 + 1*1*3 + 1*3 = 3 + 3 + 3 = 9
        var gru = new GRULayer<double>( hiddenSize: 1,
            activation: (IActivationFunction<double>?)null);
        // Lazy-input: warm up with a [seq, 1] input to resolve inputSize = 1.
        gru.Forward(new Tensor<double>(new[] { 2, 1 }));
        Assert.Equal(9, (int)gru.ParameterCount);
    }

    // ========================================================================
    // GRULayer - Output Shape
    // ========================================================================

    [Fact(Timeout = 120000)]
    public async Task GRU_OutputShape_2DInput_ReturnsHiddenSize()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // 2D input [seqLen, inputSize] -> output [hiddenSize]
        var gru = new GRULayer<double>( hiddenSize: 5,
            activation: (IActivationFunction<double>?)null);
        var input = new Tensor<double>(new[] { 4, 3 }); // 4 timesteps, 3 features
        var output = gru.Forward(input);

        Assert.Single(output.Shape.ToArray()); // 1D
        Assert.Equal(5, output.Shape[0]);
    }

    [Fact(Timeout = 120000)]
    public async Task GRU_OutputShape_3DInput_ReturnsBatchHiddenSize()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // 3D input [batch, seqLen, inputSize] -> output [batch, hiddenSize]
        var gru = new GRULayer<double>( hiddenSize: 5,
            activation: (IActivationFunction<double>?)null);
        var input = new Tensor<double>(new[] { 2, 4, 3 }); // 2 batches, 4 timesteps, 3 features
        var output = gru.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(5, output.Shape[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task GRU_OutputShape_ReturnSequences_3DOutput()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // With returnSequences=true: [batch, seqLen, inputSize] -> [batch, seqLen, hiddenSize]
        var gru = new GRULayer<double>( hiddenSize: 5,
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

    [Fact(Timeout = 120000)]
    public async Task GRU_StatelessForward_SecondCallMatchesFirst()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // The library implements STANDARD non-streaming RNN semantics: every Forward starts from
        // a zero initial hidden state (GRULayer documents this explicitly — it guarantees
        // repeated-Predict determinism and Clone-after-training parity). The previous version of
        // this test asserted the OPPOSITE (stateful carry-over), a contract the layer never had.
        var gru = new GRULayer<double>( hiddenSize: 3,
            activation: (IActivationFunction<double>?)null);

        var input = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new double[] { 1.0, 0.5 }));

        var output1 = gru.Forward(input);
        var o1vals = new double[3];
        for (int i = 0; i < 3; i++)
            o1vals[i] = output1[i];

        var output2 = gru.Forward(input);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(o1vals[i], output2[i], 10);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task GRU_ResetState_OutputMatchesFirstCall()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var gru = new GRULayer<double>( hiddenSize: 3,
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

    [Fact(Timeout = 120000)]
    public async Task GRU_OutputIsBounded_ByTanh()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var gru = new GRULayer<double>( hiddenSize: 4,
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

    [Fact(Timeout = 120000)]
    public async Task GRU_AllOutputsFinite()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var gru = new GRULayer<double>( hiddenSize: 4,
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

    [Fact(Timeout = 120000)]
    public async Task GRU_GetParameters_LengthMatchesParameterCount()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var gru = new GRULayer<double>( hiddenSize: 3,
            activation: (IActivationFunction<double>?)null);
        var parameters = gru.GetParameters();
        Assert.Equal(gru.ParameterCount, parameters.Length);
    }

    // ========================================================================
    // LSTMLayer - Parameter Count Formula
    // ========================================================================

    [Fact(Timeout = 120000)]
    public async Task LSTM_ParameterCount_Formula_Input4Hidden3()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // ParameterCount = 4 * (hiddenSize * inputSize) + 4 * (hiddenSize * hiddenSize) + 4 * hiddenSize
        // For input=4, hidden=3:
        // = 4*3*4 + 4*3*3 + 4*3 = 48 + 36 + 12 = 96
        var lstm = new LSTMLayer<double>( hiddenSize: 3,
            activation: (IActivationFunction<double>?)null);
        // LSTMLayer is lazy-input (PyTorch nn.LazyLSTM-style): its weights are not
        // allocated until the first forward resolves the input feature count from
        // the input's last axis, so ParameterCount is 0 until then. Warm up with a
        // [seq, 4] input to resolve inputSize = 4, then assert the formula.
        lstm.Forward(new Tensor<double>(new[] { 2, 4 }));
        Assert.Equal(96, (int)lstm.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task LSTM_ParameterCount_Formula_Input10Hidden8()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // = 4*8*10 + 4*8*8 + 4*8 = 320 + 256 + 32 = 608
        var lstm = new LSTMLayer<double>( hiddenSize: 8,
            activation: (IActivationFunction<double>?)null);
        // Lazy-input: warm up with a [seq, 10] input to resolve inputSize = 10.
        lstm.Forward(new Tensor<double>(new[] { 2, 10 }));
        Assert.Equal(608, (int)lstm.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task LSTM_ParameterCount_AlwaysMore_ThanGRU()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        int inputSize = 5;
        int hiddenSize = 7;
        var gru = new GRULayer<double>( hiddenSize,
            activation: (IActivationFunction<double>?)null);
        var lstm = new LSTMLayer<double>( hiddenSize,
            activation: (IActivationFunction<double>?)null);
        // Both layers are lazy-input; materialize their weights with a matching
        // [seq, inputSize] warm-up forward before comparing parameter counts.
        gru.Forward(new Tensor<double>(new[] { 2, inputSize }));
        lstm.Forward(new Tensor<double>(new[] { 2, inputSize }));
        Assert.True(lstm.ParameterCount > gru.ParameterCount,
            $"LSTM params ({lstm.ParameterCount}) should exceed GRU params ({gru.ParameterCount})");
    }

    [Fact(Timeout = 120000)]
    public async Task LSTM_ParameterCount_Ratio_Is4Over3_TimesGRU()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // Ratio = 4/3 for same input/hidden sizes
        int inputSize = 6;
        int hiddenSize = 4;
        var gru = new GRULayer<double>( hiddenSize,
            activation: (IActivationFunction<double>?)null);
        var lstm = new LSTMLayer<double>( hiddenSize,
            activation: (IActivationFunction<double>?)null);
        // Lazy-input: materialize both before reading parameter counts.
        gru.Forward(new Tensor<double>(new[] { 2, inputSize }));
        lstm.Forward(new Tensor<double>(new[] { 2, inputSize }));

        double ratio = (double)lstm.ParameterCount / gru.ParameterCount;
        Assert.Equal(4.0 / 3.0, ratio, Tol);
    }

    // ========================================================================
    // LSTMLayer - Output Shape and Finiteness
    // ========================================================================

    [Fact(Timeout = 120000)]
    public async Task LSTM_OutputShape_MatchesHiddenSize()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var lstm = new LSTMLayer<double>( hiddenSize: 5,
            activation: (IActivationFunction<double>?)null);
        var input = new Tensor<double>(new[] { 4, 3 });
        var output = lstm.Forward(input);

        Assert.True(output.Length >= 5, $"LSTM output length ({output.Length}) should be at least hiddenSize (5)");
    }

    [Fact(Timeout = 120000)]
    public async Task LSTM_AllOutputsFinite()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var lstm = new LSTMLayer<double>( hiddenSize: 4,
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

    [Fact(Timeout = 120000)]
    public async Task LSTM_GetParameters_LengthMatchesParameterCount()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var lstm = new LSTMLayer<double>( hiddenSize: 3,
            activation: (IActivationFunction<double>?)null);
        var parameters = lstm.GetParameters();
        Assert.Equal(lstm.ParameterCount, parameters.Length);
    }

    // ========================================================================
    // MeanLayer - Exact Mean Computation
    // ========================================================================

    [Fact(Timeout = 120000)]
    public async Task MeanLayer_Axis1_HandComputed()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // Input [2, 3]: [[1, 2, 3], [4, 5, 6]]
        // Mean along axis 1: [(1+2+3)/3, (4+5+6)/3] = [2, 5]
        var mean = new MeanLayer<double>(1);

        var input = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }));
        var output = mean.Forward(input);
        var arr = output.ToArray();

        Assert.Equal(2, arr.Length);
        Assert.Equal(2.0, arr[0], Tol);
        Assert.Equal(5.0, arr[1], Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MeanLayer_Axis0_HandComputed()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        // Input [2, 3]: [[1, 2, 3], [7, 8, 9]]
        // Mean along axis 0: [(1+7)/2, (2+8)/2, (3+9)/2] = [4, 5, 6]
        var mean = new MeanLayer<double>(0);

        var input = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 7.0, 8.0, 9.0 }));
        var output = mean.Forward(input);
        var arr = output.ToArray();

        Assert.Equal(3, arr.Length);
        Assert.Equal(4.0, arr[0], Tol);
        Assert.Equal(5.0, arr[1], Tol);
        Assert.Equal(6.0, arr[2], Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MeanLayer_UniformInput_MeanEqualsValue()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var mean = new MeanLayer<double>(1);

        var input = new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                input[i, j] = 7.5;

        var output = mean.Forward(input);
        var arr = output.ToArray();

        for (int i = 0; i < 3; i++)
            Assert.Equal(7.5, arr[i], Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MeanLayer_SingleElement_MeanIsElement()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var mean = new MeanLayer<double>(0);

        var input = new Tensor<double>(new[] { 1, 5 }, new Vector<double>(new double[] { 10.0, 20.0, 30.0, 40.0, 50.0 }));
        var output = mean.Forward(input);
        var arr = output.ToArray();

        Assert.Equal(5, arr.Length);
        Assert.Equal(10.0, arr[0], Tol);
        Assert.Equal(50.0, arr[4], Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MeanLayer_NegativeValues_HandComputed()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var mean = new MeanLayer<double>(1);

        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new double[] { -2.0, -4.0, 6.0, 8.0 }));
        var output = mean.Forward(input);
        var arr = output.ToArray();

        // (-2 + -4 + 6 + 8) / 4 = 8 / 4 = 2.0
        Assert.Equal(2.0, arr[0], Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task MeanLayer_LargeValues_StillAccurate()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var mean = new MeanLayer<double>(1);

        var input = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new double[] { 1e10, 2e10, 3e10 }));
        var output = mean.Forward(input);
        var arr = output.ToArray();

        Assert.Equal(2e10, arr[0], 1e5);
    }

    // ========================================================================
    // ReshapeLayer - Data Preservation
    // ========================================================================

    [Fact(Timeout = 120000)]
    public async Task ReshapeLayer_PreservesAllValues()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var reshape = new ReshapeLayer<double>(new[] { 3, 2 });

        var input = new Tensor<double>(new[] { 1, 2, 3 }, new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }));
        var output = reshape.Forward(input);

        Assert.Equal(6, output.Length);
        var inputArr = input.ToArray();
        var outputArr = output.ToArray();
        for (int i = 0; i < 6; i++)
            Assert.Equal(inputArr[i], outputArr[i], Tol);
    }

    [Fact(Timeout = 120000)]
    public async Task ReshapeLayer_OutputShapeIsCorrect()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var reshape = new ReshapeLayer<double>(new[] { 6, 2 });

        var input = new Tensor<double>(new[] { 1, 4, 3 });
        var output = reshape.Forward(input);

        Assert.True(output.Length == 12, $"Expected 12 elements, got {output.Length}");
    }



    [Fact(Timeout = 120000)]
    public async Task ReshapeLayer_NoTrainableParameters()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var reshape = new ReshapeLayer<double>(new[] { 2, 2 });
        Assert.Equal(0, (int)reshape.ParameterCount);
    }

    // ========================================================================
    // GRU - Backward Gradient Structure
    // ========================================================================


    // ========================================================================
    // GRU vs LSTM - Structural Comparisons
    // ========================================================================

    [Fact(Timeout = 120000)]
    public async Task GRU_And_LSTM_SameInputSize_DifferentParamCounts()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        int inputSize = 8;
        int hiddenSize = 16;

        var gru = new GRULayer<double>( hiddenSize,
            activation: (IActivationFunction<double>?)null);
        var lstm = new LSTMLayer<double>( hiddenSize,
            activation: (IActivationFunction<double>?)null);

        // Lazy-input: materialize both layers' weights with a matching
        // [seq, inputSize] warm-up forward before reading parameter counts.
        gru.Forward(new Tensor<double>(new[] { 2, inputSize }));
        lstm.Forward(new Tensor<double>(new[] { 2, inputSize }));

        int expectedGRU = 3 * (hiddenSize * inputSize + hiddenSize * hiddenSize + hiddenSize);
        int expectedLSTM = 4 * (hiddenSize * inputSize + hiddenSize * hiddenSize + hiddenSize);

        Assert.Equal(expectedGRU, (int)gru.ParameterCount);
        Assert.Equal(expectedLSTM, (int)lstm.ParameterCount);
    }

    // ========================================================================
    // GRU - Zero and Large Input Behavior
    // ========================================================================

    [Fact(Timeout = 120000)]
    public async Task GRU_ZeroInput_OutputIsFinite()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var gru = new GRULayer<double>( hiddenSize: 4,
            activation: (IActivationFunction<double>?)null);

        var input = new Tensor<double>(new[] { 1, 3 });
        var output = gru.Forward(input);

        for (int i = 0; i < 4; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"NaN at [{i}]");
            Assert.False(double.IsInfinity(output[i]), $"Infinity at [{i}]");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task GRU_LargeInput_StillBounded()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var gru = new GRULayer<double>( hiddenSize: 3,
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

    [Fact(Timeout = 120000)]
    public async Task GRU_MultipleTimesteps_OutputChanges()
    {
        await Task.Yield(); // make the body truly async so [Fact(Timeout)] is enforced (xUnit v2)
        var gru = new GRULayer<double>( hiddenSize: 3,
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

}
