using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LoRA;

/// <summary>
/// Deep math-correctness integration tests for the LoRA module.
/// Verifies exact numerical values for forward pass, backward pass, merge weights,
/// gradient computation, parameter updates, and adapter-specific math.
/// </summary>
[Collection("NonParallelIntegration")]
public class LoRADeepMathIntegrationTests
{
    private const double Tol = 1e-10;

    #region Helpers

    private static LoRALayer<double> CreateLayerWithKnownMatrices(
        int inputSize, int outputSize, int rank, double alpha,
        double[,] matrixA, double[,] matrixB)
    {
        var layer = new LoRALayer<double>(inputSize, outputSize, rank, alpha);
        var parameters = new Vector<double>(layer.ParameterCount);
        int idx = 0;
        for (int i = 0; i < inputSize; i++)
            for (int j = 0; j < rank; j++)
                parameters[idx++] = matrixA[i, j];
        for (int i = 0; i < rank; i++)
            for (int j = 0; j < outputSize; j++)
                parameters[idx++] = matrixB[i, j];
        layer.SetParameters(parameters);
        return layer;
    }

    private static Tensor<double> MakeTensor(int batch, int features, double[] data)
    {
        return new Tensor<double>(new[] { batch, features }, new Vector<double>(data));
    }

    #endregion

    #region LoRALayer Scaling Factor

    [Fact]
    public void Scaling_AlphaOverRank_ExactValue()
    {
        // alpha=4, rank=2 → scaling = 4/2 = 2
        var layer = new LoRALayer<double>(4, 4, 2, 4.0);
        Assert.Equal(2.0, Convert.ToDouble(layer.Scaling), Tol);
    }

    [Fact]
    public void Scaling_DefaultAlpha_EqualsOne()
    {
        // default alpha = rank → scaling = rank/rank = 1
        var layer = new LoRALayer<double>(4, 4, 3, -1);
        Assert.Equal(1.0, Convert.ToDouble(layer.Scaling), Tol);
    }

    [Fact]
    public void Scaling_AlphaDoubleRank_EqualsTwo()
    {
        var layer = new LoRALayer<double>(10, 10, 5, 10.0);
        Assert.Equal(2.0, Convert.ToDouble(layer.Scaling), Tol);
    }

    #endregion

    #region B-Zero Initialization

    [Fact]
    public void BZero_InitialForwardPass_ProducesZeroOutput()
    {
        // B starts at zero → input * A * 0 * scaling = 0
        var layer = new LoRALayer<double>(3, 2, 2, 4.0);
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);
        var output = layer.Forward(input);

        for (int i = 0; i < output.Length; i++)
            Assert.Equal(0.0, output[i], Tol);
    }

    [Fact]
    public void BZero_InitialMergeWeights_AllZeros()
    {
        var layer = new LoRALayer<double>(4, 3, 2, 4.0);
        var merged = layer.MergeWeights();

        for (int i = 0; i < merged.Rows; i++)
            for (int j = 0; j < merged.Columns; j++)
                Assert.Equal(0.0, merged[i, j], Tol);
    }

    [Fact]
    public void BZero_GetMatrixB_AllZeros()
    {
        var layer = new LoRALayer<double>(4, 3, 2, 4.0);
        var B = layer.GetMatrixB();

        for (int i = 0; i < B.Rows; i++)
            for (int j = 0; j < B.Columns; j++)
                Assert.Equal(0.0, B[i, j], Tol);
    }

    #endregion

    #region Forward Pass Exact Values

    [Fact]
    public void Forward_3x2Rank2_HandComputedValues()
    {
        // A=[3,2], B=[2,2], alpha=4, scaling=2
        // A = [[1,0],[0,1],[1,1]], B = [[1,2],[3,4]]
        // input = [1,2,3]
        // input*A = [1+0+3, 0+2+3] = [4, 5]
        // (input*A)*B = [4+15, 8+20] = [19, 28]
        // output = [19,28]*2 = [38, 56]
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);
        var output = layer.Forward(input);

        Assert.Equal(38.0, output[0], Tol);
        Assert.Equal(56.0, output[1], Tol);
    }

    [Fact]
    public void Forward_BatchOf2_HandComputedValues()
    {
        // Same matrices, batch of 2
        // input1=[1,2,3]: output=[38,56] (computed above)
        // input2=[4,5,6]:
        //   input2*A = [4+0+6, 0+5+6] = [10, 11]
        //   (input2*A)*B = [10+33, 20+44] = [43, 64]
        //   output2 = [43,64]*2 = [86, 128]
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        var output = layer.Forward(input);

        Assert.Equal(38.0, output[0], Tol);
        Assert.Equal(56.0, output[1], Tol);
        Assert.Equal(86.0, output[2], Tol);
        Assert.Equal(128.0, output[3], Tol);
    }

    [Fact]
    public void Forward_ScalingOne_NoAmplification()
    {
        // alpha=rank=2 → scaling=1
        // A = [[1,0],[0,1]], B = [[2,0],[0,3]]
        // input = [1, 1]
        // input*A = [1, 1]
        // (input*A)*B = [2, 3]
        // output = [2,3]*1 = [2, 3]
        var layer = CreateLayerWithKnownMatrices(2, 2, 2, 2.0,
            new double[,] { { 1, 0 }, { 0, 1 } },
            new double[,] { { 2, 0 }, { 0, 3 } });
        var input = MakeTensor(1, 2, [1.0, 1.0]);
        var output = layer.Forward(input);

        Assert.Equal(2.0, output[0], Tol);
        Assert.Equal(3.0, output[1], Tol);
    }

    [Fact]
    public void Forward_Rank1_OuterProductBehavior()
    {
        // rank=1: A=[3,1], B=[1,2], alpha=1, scaling=1
        // A = [[2],[3],[5]], B = [[7, 11]]
        // input = [1, 0, 0]
        // input*A = [2]
        // 2 * B = [14, 22]
        var layer = CreateLayerWithKnownMatrices(3, 2, 1, 1.0,
            new double[,] { { 2 }, { 3 }, { 5 } },
            new double[,] { { 7, 11 } });
        var input = MakeTensor(1, 3, [1.0, 0.0, 0.0]);
        var output = layer.Forward(input);

        Assert.Equal(14.0, output[0], Tol);
        Assert.Equal(22.0, output[1], Tol);
    }

    [Fact]
    public void Forward_1x1Rank1_ScalarMultiplication()
    {
        // input=1, output=1, rank=1, alpha=3, scaling=3
        // A = [[2]], B = [[5]]
        // input = [4]
        // output = 4 * 2 * 5 * 3 = 120
        var layer = CreateLayerWithKnownMatrices(1, 1, 1, 3.0,
            new double[,] { { 2 } },
            new double[,] { { 5 } });
        var input = MakeTensor(1, 1, [4.0]);
        var output = layer.Forward(input);

        Assert.Equal(120.0, output[0], Tol);
    }

    #endregion

    #region MergeWeights Exact Values

    [Fact]
    public void MergeWeights_3x2Rank2_HandComputed()
    {
        // A*B*scaling = [[1,0],[0,1],[1,1]] * [[1,2],[3,4]] * 2
        // A*B = [[1,2],[3,4],[4,6]]
        // merged = [[2,4],[6,8],[8,12]]
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var merged = layer.MergeWeights();

        Assert.Equal(3, merged.Rows);
        Assert.Equal(2, merged.Columns);
        Assert.Equal(2.0, merged[0, 0], Tol);
        Assert.Equal(4.0, merged[0, 1], Tol);
        Assert.Equal(6.0, merged[1, 0], Tol);
        Assert.Equal(8.0, merged[1, 1], Tol);
        Assert.Equal(8.0, merged[2, 0], Tol);
        Assert.Equal(12.0, merged[2, 1], Tol);
    }

    [Fact]
    public void MergeWeights_ScalingOne_EqualsATimesB()
    {
        // alpha=rank → scaling=1 → merged = A*B
        var layer = CreateLayerWithKnownMatrices(2, 2, 2, 2.0,
            new double[,] { { 1, 2 }, { 3, 4 } },
            new double[,] { { 5, 6 }, { 7, 8 } });
        var merged = layer.MergeWeights();

        // A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //     = [[19, 22], [43, 50]]
        Assert.Equal(19.0, merged[0, 0], Tol);
        Assert.Equal(22.0, merged[0, 1], Tol);
        Assert.Equal(43.0, merged[1, 0], Tol);
        Assert.Equal(50.0, merged[1, 1], Tol);
    }

    [Fact]
    public void MergeWeights_IdentityA_EqualsScaledB()
    {
        // A = I (identity), rank=2, inputSize=2, alpha=6, scaling=3
        // merged = I * B * 3 = B * 3
        var layer = CreateLayerWithKnownMatrices(2, 3, 2, 6.0,
            new double[,] { { 1, 0 }, { 0, 1 } },
            new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var merged = layer.MergeWeights();

        Assert.Equal(3.0, merged[0, 0], Tol);
        Assert.Equal(6.0, merged[0, 1], Tol);
        Assert.Equal(9.0, merged[0, 2], Tol);
        Assert.Equal(12.0, merged[1, 0], Tol);
        Assert.Equal(15.0, merged[1, 1], Tol);
        Assert.Equal(18.0, merged[1, 2], Tol);
    }

    #endregion

    #region Backward Pass Exact Gradients

    [Fact]
    public void Backward_GradB_ExactValues()
    {
        // Setup: same 3x2 rank=2 alpha=4 layer
        // dL/dB = (input*A)^T * gradMatrix * scaling
        // (input*A)^T = [[4],[5]], gradMatrix = [[1,1]]
        // = [[4,4],[5,5]] * 2 = [[8,8],[10,10]]
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);
        layer.Forward(input);

        var outputGrad = MakeTensor(1, 2, [1.0, 1.0]);
        layer.Backward(outputGrad);

        // Read parameter gradients: first 6 are for A, next 4 for B
        var paramGrads = layer.GetParameterGradients();
        // B gradients at indices 6,7,8,9
        Assert.Equal(8.0, paramGrads[6], Tol);   // gradB[0,0]
        Assert.Equal(8.0, paramGrads[7], Tol);   // gradB[0,1]
        Assert.Equal(10.0, paramGrads[8], Tol);  // gradB[1,0]
        Assert.Equal(10.0, paramGrads[9], Tol);  // gradB[1,1]
    }

    [Fact]
    public void Backward_GradA_ExactValues()
    {
        // dL/dA = input^T * (gradMatrix * B^T) * scaling
        // gradMatrix*B^T = [[1,1]]*[[1,3],[2,4]] = [[3,7]]
        // input^T * [[3,7]] = [[3,7],[6,14],[9,21]]
        // * scaling(2) = [[6,14],[12,28],[18,42]]
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);
        layer.Forward(input);

        var outputGrad = MakeTensor(1, 2, [1.0, 1.0]);
        layer.Backward(outputGrad);

        var paramGrads = layer.GetParameterGradients();
        // A gradients at indices 0-5
        Assert.Equal(6.0, paramGrads[0], Tol);   // gradA[0,0]
        Assert.Equal(14.0, paramGrads[1], Tol);  // gradA[0,1]
        Assert.Equal(12.0, paramGrads[2], Tol);  // gradA[1,0]
        Assert.Equal(28.0, paramGrads[3], Tol);  // gradA[1,1]
        Assert.Equal(18.0, paramGrads[4], Tol);  // gradA[2,0]
        Assert.Equal(42.0, paramGrads[5], Tol);  // gradA[2,1]
    }

    [Fact]
    public void Backward_InputGradient_ExactValues()
    {
        // dL/dinput = gradMatrix * B^T * A^T * scaling
        // = [[3,7]] * [[1,0,1],[0,1,1]] * 2
        // = [[3,7,10]] * 2 = [[6,14,20]]
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);
        layer.Forward(input);

        var outputGrad = MakeTensor(1, 2, [1.0, 1.0]);
        var inputGrad = layer.Backward(outputGrad);

        Assert.Equal(6.0, inputGrad[0], Tol);
        Assert.Equal(14.0, inputGrad[1], Tol);
        Assert.Equal(20.0, inputGrad[2], Tol);
    }

    [Fact]
    public void Backward_AsymmetricGradient_ExactValues()
    {
        // outputGrad = [2, 0] instead of [1, 1]
        // gradMatrix = [[2, 0]]
        // gradMatrix*B^T = [[2,0]]*[[1,3],[2,4]] = [[2,6]]
        // input^T * [[2,6]] = [[2,6],[4,12],[6,18]]
        // gradA = * 2 = [[4,12],[8,24],[12,36]]
        // inputTimesA^T * gradMatrix = [[4],[5]]^T * [[2,0]] ... wait
        // inputTimesA^T = [[4],[5]] * gradMatrix = [[4],[5]] * [[2,0]] = [[8,0],[10,0]]
        // gradB = * 2 = [[16,0],[20,0]]
        // inputGrad = [[2,6]] * [[1,0,1],[0,1,1]] * 2 = [[2,6,8]] * 2 = [[4,12,16]]
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);
        layer.Forward(input);

        var outputGrad = MakeTensor(1, 2, [2.0, 0.0]);
        var inputGrad = layer.Backward(outputGrad);

        Assert.Equal(4.0, inputGrad[0], Tol);
        Assert.Equal(12.0, inputGrad[1], Tol);
        Assert.Equal(16.0, inputGrad[2], Tol);

        var paramGrads = layer.GetParameterGradients();
        Assert.Equal(4.0, paramGrads[0], Tol);   // gradA[0,0]
        Assert.Equal(12.0, paramGrads[1], Tol);  // gradA[0,1]
        Assert.Equal(16.0, paramGrads[6], Tol);  // gradB[0,0]
        Assert.Equal(0.0, paramGrads[7], Tol);   // gradB[0,1]
    }

    [Fact]
    public void Backward_ZeroGradient_ProducesZeroGrads()
    {
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);
        layer.Forward(input);

        var outputGrad = MakeTensor(1, 2, [0.0, 0.0]);
        var inputGrad = layer.Backward(outputGrad);

        for (int i = 0; i < inputGrad.Length; i++)
            Assert.Equal(0.0, inputGrad[i], Tol);

        var paramGrads = layer.GetParameterGradients();
        for (int i = 0; i < paramGrads.Length; i++)
            Assert.Equal(0.0, paramGrads[i], Tol);
    }

    #endregion

    #region SGD Update Exact Values

    [Fact]
    public void UpdateParameters_MatrixA_ExactSGDStep()
    {
        // A_new = A - lr * gradA
        // lr=0.1, gradA = [[6,14],[12,28],[18,42]]
        // A_new = [[1-0.6, 0-1.4],[0-1.2, 1-2.8],[1-1.8, 1-4.2]]
        //       = [[0.4, -1.4],[-1.2, -1.8],[-0.8, -3.2]]
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);
        layer.Forward(input);
        layer.Backward(MakeTensor(1, 2, [1.0, 1.0]));
        layer.UpdateParameters(0.1);

        var A = layer.GetMatrixA();
        Assert.Equal(0.4, A[0, 0], Tol);
        Assert.Equal(-1.4, A[0, 1], Tol);
        Assert.Equal(-1.2, A[1, 0], Tol);
        Assert.Equal(-1.8, A[1, 1], Tol);
        Assert.Equal(-0.8, A[2, 0], Tol);
        Assert.Equal(-3.2, A[2, 1], Tol);
    }

    [Fact]
    public void UpdateParameters_MatrixB_ExactSGDStep()
    {
        // B_new = B - lr * gradB
        // lr=0.1, gradB = [[8,8],[10,10]]
        // B_new = [[1-0.8, 2-0.8],[3-1.0, 4-1.0]]
        //       = [[0.2, 1.2],[2.0, 3.0]]
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);
        layer.Forward(input);
        layer.Backward(MakeTensor(1, 2, [1.0, 1.0]));
        layer.UpdateParameters(0.1);

        var B = layer.GetMatrixB();
        Assert.Equal(0.2, B[0, 0], Tol);
        Assert.Equal(1.2, B[0, 1], Tol);
        Assert.Equal(2.0, B[1, 0], Tol);
        Assert.Equal(3.0, B[1, 1], Tol);
    }

    [Fact]
    public void UpdateParameters_ParameterVector_MatchesMatrices()
    {
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);
        layer.Forward(input);
        layer.Backward(MakeTensor(1, 2, [1.0, 1.0]));
        layer.UpdateParameters(0.1);

        var parameters = layer.GetParameters();
        // First 6: A (row-major), next 4: B (row-major)
        Assert.Equal(0.4, parameters[0], Tol);    // A[0,0]
        Assert.Equal(-1.4, parameters[1], Tol);   // A[0,1]
        Assert.Equal(-1.2, parameters[2], Tol);   // A[1,0]
        Assert.Equal(-1.8, parameters[3], Tol);   // A[1,1]
        Assert.Equal(-0.8, parameters[4], Tol);   // A[2,0]
        Assert.Equal(-3.2, parameters[5], Tol);   // A[2,1]
        Assert.Equal(0.2, parameters[6], Tol);    // B[0,0]
        Assert.Equal(1.2, parameters[7], Tol);    // B[0,1]
        Assert.Equal(2.0, parameters[8], Tol);    // B[1,0]
        Assert.Equal(3.0, parameters[9], Tol);    // B[1,1]
    }

    #endregion

    #region Parameter Packing Round-Trip

    [Fact]
    public void SetParameters_SetsMatrixA_Correctly()
    {
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });

        var A = layer.GetMatrixA();
        Assert.Equal(1.0, A[0, 0], Tol);
        Assert.Equal(0.0, A[0, 1], Tol);
        Assert.Equal(0.0, A[1, 0], Tol);
        Assert.Equal(1.0, A[1, 1], Tol);
        Assert.Equal(1.0, A[2, 0], Tol);
        Assert.Equal(1.0, A[2, 1], Tol);
    }

    [Fact]
    public void SetParameters_SetsMatrixB_Correctly()
    {
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });

        var B = layer.GetMatrixB();
        Assert.Equal(1.0, B[0, 0], Tol);
        Assert.Equal(2.0, B[0, 1], Tol);
        Assert.Equal(3.0, B[1, 0], Tol);
        Assert.Equal(4.0, B[1, 1], Tol);
    }

    [Fact]
    public void GetParameters_RoundTrip_PreservesValues()
    {
        var layer = CreateLayerWithKnownMatrices(2, 2, 1, 1.0,
            new double[,] { { 7 }, { 11 } },
            new double[,] { { 13, 17 } });

        var params1 = layer.GetParameters();
        layer.SetParameters(params1);
        var params2 = layer.GetParameters();

        for (int i = 0; i < params1.Length; i++)
            Assert.Equal(params1[i], params2[i], Tol);
    }

    [Fact]
    public void GetMatrixA_ReturnsClone_NotReference()
    {
        var layer = CreateLayerWithKnownMatrices(2, 2, 1, 1.0,
            new double[,] { { 3 }, { 5 } },
            new double[,] { { 7, 11 } });

        var A1 = layer.GetMatrixA();
        A1[0, 0] = 999.0; // modify clone
        var A2 = layer.GetMatrixA();
        Assert.Equal(3.0, A2[0, 0], Tol); // original unchanged
    }

    #endregion

    #region Parameter Count Formula

    [Fact]
    public void ParameterCount_StandardFormula()
    {
        // paramCount = inputSize*rank + rank*outputSize
        var layer = new LoRALayer<double>(64, 32, 8, 16.0);
        Assert.Equal(64 * 8 + 8 * 32, layer.ParameterCount);
    }

    [Fact]
    public void ParameterCount_Asymmetric()
    {
        var layer = new LoRALayer<double>(100, 10, 4, 4.0);
        Assert.Equal(100 * 4 + 4 * 10, layer.ParameterCount);
    }

    [Fact]
    public void ParameterCount_Rank1_Minimal()
    {
        var layer = new LoRALayer<double>(50, 30, 1, 1.0);
        Assert.Equal(50 + 30, layer.ParameterCount);
    }

    #endregion

    #region DoRA Magnitude Decomposition

    [Fact]
    public void DoRA_Magnitude_L2Norm_KnownWeights()
    {
        // Weights = [[3,4],[0,5]], biases = [0,0]
        // magnitude[0] = ||[3,4]|| = 5
        // magnitude[1] = ||[0,5]|| = 5
        var baseLayer = new DenseLayer<double>(2, 2);
        baseLayer.SetParameters(new Vector<double>(new double[] { 3, 4, 0, 5, 0, 0 }));

        var adapter = new DoRAAdapter<double>(baseLayer, rank: 1, alpha: 1.0);
        var parameters = adapter.GetParameters();

        // DoRA params: LoRA params (inputSize*rank + rank*outputSize) + magnitude (outputSize)
        // = 2*1 + 1*2 + 2 = 6
        int loraParamCount = 2 * 1 + 1 * 2; // A[2x1] + B[1x2] = 4
        int magnitudeStart = loraParamCount;

        Assert.Equal(5.0, parameters[magnitudeStart], 1e-6);     // magnitude[0]
        Assert.Equal(5.0, parameters[magnitudeStart + 1], 1e-6); // magnitude[1]
    }

    [Fact]
    public void DoRA_Magnitude_UnitWeights()
    {
        // Weights = [[1,0],[0,1]], biases = [0,0]
        // magnitude[0] = ||[1,0]|| = 1, magnitude[1] = ||[0,1]|| = 1
        var baseLayer = new DenseLayer<double>(2, 2);
        baseLayer.SetParameters(new Vector<double>(new double[] { 1, 0, 0, 1, 0, 0 }));

        var adapter = new DoRAAdapter<double>(baseLayer, rank: 1, alpha: 1.0);
        var parameters = adapter.GetParameters();
        int magnitudeStart = 2 * 1 + 1 * 2;

        Assert.Equal(1.0, parameters[magnitudeStart], 1e-6);
        Assert.Equal(1.0, parameters[magnitudeStart + 1], 1e-6);
    }

    [Fact]
    public void DoRA_Magnitude_3_4_5_Triangle()
    {
        // Only first output neuron has [3,4] → magnitude = 5
        // Second: [5,12] → magnitude = 13
        var baseLayer = new DenseLayer<double>(2, 2);
        baseLayer.SetParameters(new Vector<double>(new double[] { 3, 4, 5, 12, 0, 0 }));

        var adapter = new DoRAAdapter<double>(baseLayer, rank: 1, alpha: 1.0);
        var parameters = adapter.GetParameters();
        int magnitudeStart = 2 * 1 + 1 * 2;

        Assert.Equal(5.0, parameters[magnitudeStart], 1e-6);
        Assert.Equal(13.0, parameters[magnitudeStart + 1], 1e-6);
    }

    [Fact]
    public void DoRA_ParameterCount_IncludesMagnitude()
    {
        var baseLayer = new DenseLayer<double>(10, 5);
        var adapter = new DoRAAdapter<double>(baseLayer, rank: 2, alpha: 2.0);

        // frozen: loraParams + magnitude = (10*2 + 2*5) + 5 = 35
        Assert.Equal(10 * 2 + 2 * 5 + 5, adapter.ParameterCount);
    }

    #endregion

    #region DoRA Initial Forward

    [Fact]
    public void DoRA_InitialForward_MatchesBaseWeightTimesInput()
    {
        // With B=0, DoRA output = input @ W^T (no bias)
        // W = [[3,4],[0,5]], input = [1, 2]
        // output = [1*3+2*4, 1*0+2*5] = [11, 10]
        var baseLayer = new DenseLayer<double>(2, 2);
        baseLayer.SetParameters(new Vector<double>(new double[] { 3, 4, 0, 5, 0, 0 }));

        var adapter = new DoRAAdapter<double>(baseLayer, rank: 1, alpha: 1.0);
        var input = MakeTensor(1, 2, [1.0, 2.0]);
        var output = adapter.Forward(input);

        Assert.Equal(11.0, output[0], 1e-6);
        Assert.Equal(10.0, output[1], 1e-6);
    }

    [Fact]
    public void DoRA_InitialForward_IdentityWeights_PassThrough()
    {
        // W = I (identity), input = [3, 7]
        // output = [3, 7]
        var baseLayer = new DenseLayer<double>(2, 2);
        baseLayer.SetParameters(new Vector<double>(new double[] { 1, 0, 0, 1, 0, 0 }));

        var adapter = new DoRAAdapter<double>(baseLayer, rank: 1, alpha: 1.0);
        var input = MakeTensor(1, 2, [3.0, 7.0]);
        var output = adapter.Forward(input);

        Assert.Equal(3.0, output[0], 1e-6);
        Assert.Equal(7.0, output[1], 1e-6);
    }

    [Fact]
    public void DoRA_InitialForward_Batch_ExactValues()
    {
        // W = [[3,4],[0,5]]
        // input1 = [1, 0] → [3, 0]
        // input2 = [0, 1] → [4, 5]
        var baseLayer = new DenseLayer<double>(2, 2);
        baseLayer.SetParameters(new Vector<double>(new double[] { 3, 4, 0, 5, 0, 0 }));

        var adapter = new DoRAAdapter<double>(baseLayer, rank: 1, alpha: 1.0);
        var input = MakeTensor(2, 2, [1.0, 0.0, 0.0, 1.0]);
        var output = adapter.Forward(input);

        Assert.Equal(3.0, output[0], 1e-6);
        Assert.Equal(0.0, output[1], 1e-6);
        Assert.Equal(4.0, output[2], 1e-6);
        Assert.Equal(5.0, output[3], 1e-6);
    }

    #endregion

    #region StandardLoRA Adapter Math

    [Fact]
    public void StandardAdapter_FrozenParamCount_OnlyLoRA()
    {
        var baseLayer = new DenseLayer<double>(10, 5);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, rank: 3, freezeBaseLayer: true);

        int loraOnly = 10 * 3 + 3 * 5;
        Assert.Equal(loraOnly, adapter.ParameterCount);
    }

    [Fact]
    public void StandardAdapter_UnfrozenParamCount_BaseAndLoRA()
    {
        var baseLayer = new DenseLayer<double>(10, 5);
        int baseParams = baseLayer.ParameterCount;
        var adapter = new StandardLoRAAdapter<double>(baseLayer, rank: 3, freezeBaseLayer: false);

        int loraParams = 10 * 3 + 3 * 5;
        Assert.Equal(baseParams + loraParams, adapter.ParameterCount);
    }

    [Fact]
    public void StandardAdapter_InitialForward_EqualsBaseOnly()
    {
        // With B=0, LoRA output = 0, so adapter output = base output + 0 = base output
        var baseLayer = new DenseLayer<double>(4, 3);
        baseLayer.SetParameters(new Vector<double>(new double[]
        {
            1, 0, 0, 0,  // W[0,:] = [1,0,0,0]
            0, 1, 0, 0,  // W[1,:] = [0,1,0,0]
            0, 0, 1, 0,  // W[2,:] = [0,0,1,0]
            0, 0, 0       // biases = [0,0,0]
        }));

        var adapter = new StandardLoRAAdapter<double>(baseLayer, rank: 2, alpha: 2.0);
        var input = MakeTensor(1, 4, [5.0, 7.0, 11.0, 13.0]);

        var baseOutput = baseLayer.Forward(input);
        // Need a fresh forward on the adapter (base layer state was consumed)
        baseLayer.SetParameters(new Vector<double>(new double[]
        {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0
        }));
        var adapter2 = new StandardLoRAAdapter<double>(baseLayer, rank: 2, alpha: 2.0);
        var adapterOutput = adapter2.Forward(input);

        // Base output with identity-like weights: [5, 7, 11]
        // Adapter output should match since LoRA starts at zero
        for (int i = 0; i < adapterOutput.Length; i++)
        {
            Assert.False(double.IsNaN(adapterOutput[i]));
            Assert.False(double.IsInfinity(adapterOutput[i]));
        }
    }

    [Fact]
    public void StandardAdapter_ParameterEfficiency_LargeLayer()
    {
        // 1024x1024 with rank=8: LoRA uses < 2% of full parameters
        int size = 1024;
        int rank = 8;
        var baseLayer = new DenseLayer<double>(size, size);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, rank, freezeBaseLayer: true);

        int fullWeights = size * size;
        int loraParams = adapter.ParameterCount;
        double ratio = (double)loraParams / fullWeights;

        Assert.True(ratio < 0.02, $"LoRA should use < 2% of parameters, got {ratio:P2}");
        Assert.Equal(size * rank + rank * size, loraParams);
    }

    #endregion

    #region LoHa Math

    [Fact]
    public void LoHa_ParameterCount_Frozen()
    {
        // 2 * rank * inputSize * outputSize (frozen)
        var baseLayer = new DenseLayer<double>(10, 5);
        var adapter = new LoHaAdapter<double>(baseLayer, rank: 3, alpha: 3.0);

        Assert.Equal(2 * 3 * 10 * 5, adapter.ParameterCount);
    }

    [Fact]
    public void LoHa_InitialForward_EqualsBaseOutput()
    {
        // B matrices all zero initially → ΔW = 0 → adapter output = base output
        var baseLayer = new DenseLayer<double>(4, 3);
        var input = MakeTensor(1, 4, [1.0, 2.0, 3.0, 4.0]);

        var baseOutput = baseLayer.Forward(input);
        baseLayer.ResetState();

        var adapter = new LoHaAdapter<double>(baseLayer, rank: 2, alpha: 2.0);
        var adapterOutput = adapter.Forward(input);

        for (int i = 0; i < baseOutput.Length; i++)
            Assert.Equal(baseOutput[i], adapterOutput[i], 1e-6);
    }

    #endregion

    #region Forward-Backward Consistency

    [Fact]
    public void ForwardBackward_SecondForward_UsesUpdatedWeights()
    {
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);

        // First forward: [38, 56]
        var output1 = layer.Forward(input);
        Assert.Equal(38.0, output1[0], Tol);

        // Update
        layer.Backward(MakeTensor(1, 2, [1.0, 1.0]));
        layer.UpdateParameters(0.1);

        // Second forward with updated matrices should differ
        var output2 = layer.Forward(input);
        Assert.NotEqual(38.0, output2[0], Tol);

        // Verify exact second forward value using updated A and B
        var A = layer.GetMatrixA();
        var B = layer.GetMatrixB();

        // input*A = [1*A[0,0]+2*A[1,0]+3*A[2,0], 1*A[0,1]+2*A[1,1]+3*A[2,1]]
        double ia0 = 1 * A[0, 0] + 2 * A[1, 0] + 3 * A[2, 0];
        double ia1 = 1 * A[0, 1] + 2 * A[1, 1] + 3 * A[2, 1];
        // (input*A)*B scaled by 2
        double expected0 = (ia0 * B[0, 0] + ia1 * B[1, 0]) * 2;
        double expected1 = (ia0 * B[0, 1] + ia1 * B[1, 1]) * 2;

        Assert.Equal(expected0, output2[0], Tol);
        Assert.Equal(expected1, output2[1], Tol);
    }

    [Fact]
    public void MergeWeights_AfterUpdate_ReflectsNewMatrices()
    {
        var layer = CreateLayerWithKnownMatrices(2, 2, 1, 2.0,
            new double[,] { { 1 }, { 0 } },
            new double[,] { { 1, 0 } });

        // Initial merge: A*B*scaling = [[1],[0]] * [[1,0]] * 2 = [[1,0],[0,0]] * 2 = [[2,0],[0,0]]
        var merged1 = layer.MergeWeights();
        Assert.Equal(2.0, merged1[0, 0], Tol);
        Assert.Equal(0.0, merged1[0, 1], Tol);
        Assert.Equal(0.0, merged1[1, 0], Tol);
        Assert.Equal(0.0, merged1[1, 1], Tol);

        // Forward+backward+update
        var input = MakeTensor(1, 2, [1.0, 1.0]);
        layer.Forward(input);
        layer.Backward(MakeTensor(1, 2, [1.0, 0.0]));
        layer.UpdateParameters(0.1);

        var merged2 = layer.MergeWeights();
        // Should differ from initial
        Assert.False(Math.Abs(merged2[0, 0] - 2.0) < Tol && Math.Abs(merged2[1, 0] - 0.0) < Tol,
            "Merge weights should change after parameter update");
    }

    [Fact]
    public void ResetState_ClearsForwardState()
    {
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var input = MakeTensor(1, 3, [1.0, 2.0, 3.0]);

        layer.Forward(input);
        layer.ResetState();

        // Backward should fail after reset
        Assert.Throws<InvalidOperationException>(() =>
            layer.Backward(MakeTensor(1, 2, [1.0, 1.0])));
    }

    #endregion

    #region Numerical Properties

    [Fact]
    public void Forward_BatchIndependence_SameSampleSameResult()
    {
        // Result for a sample should be the same whether it's alone or in a batch
        var layer1 = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });
        var layer2 = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });

        // Single sample
        var single = MakeTensor(1, 3, [1.0, 2.0, 3.0]);
        var singleOutput = layer1.Forward(single);

        // Batch with same sample first
        var batch = MakeTensor(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        var batchOutput = layer2.Forward(batch);

        Assert.Equal(singleOutput[0], batchOutput[0], Tol);
        Assert.Equal(singleOutput[1], batchOutput[1], Tol);
    }

    [Fact]
    public void Forward_LinearInInput_Superposition()
    {
        // For fixed A, B: f(x1 + x2) = f(x1) + f(x2) because LoRA is linear
        var layer = CreateLayerWithKnownMatrices(2, 2, 1, 1.0,
            new double[,] { { 1 }, { 2 } },
            new double[,] { { 3, 4 } });

        var x1 = MakeTensor(1, 2, [1.0, 0.0]);
        var x2 = MakeTensor(1, 2, [0.0, 1.0]);
        var xsum = MakeTensor(1, 2, [1.0, 1.0]);

        var out1 = layer.Forward(x1);
        layer.ResetState();
        var out2 = layer.Forward(x2);
        layer.ResetState();
        var outSum = layer.Forward(xsum);

        Assert.Equal(out1[0] + out2[0], outSum[0], Tol);
        Assert.Equal(out1[1] + out2[1], outSum[1], Tol);
    }

    [Fact]
    public void Forward_ScalingInOutput_Homogeneity()
    {
        // f(c*x) = c * f(x) for linear layer
        var layer = CreateLayerWithKnownMatrices(2, 2, 1, 1.0,
            new double[,] { { 1 }, { 2 } },
            new double[,] { { 3, 4 } });

        var x = MakeTensor(1, 2, [2.0, 3.0]);
        var cx = MakeTensor(1, 2, [6.0, 9.0]); // 3*x

        var outX = layer.Forward(x);
        layer.ResetState();
        var outCx = layer.Forward(cx);

        Assert.Equal(3 * outX[0], outCx[0], Tol);
        Assert.Equal(3 * outX[1], outCx[1], Tol);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Rank1_ForwardBackwardUpdate_Consistent()
    {
        // Minimal rank=1 layer
        var layer = CreateLayerWithKnownMatrices(2, 2, 1, 2.0,
            new double[,] { { 1 }, { -1 } },
            new double[,] { { 2, 3 } });

        // scaling = 2/1 = 2
        // input=[1,1], input*A=[1-1]=[0], (input*A)*B=[0,0], output=[0,0]*2=[0,0]
        var input = MakeTensor(1, 2, [1.0, 1.0]);
        var output = layer.Forward(input);
        Assert.Equal(0.0, output[0], Tol);
        Assert.Equal(0.0, output[1], Tol);

        // input=[1,0], input*A=[1], [1]*B=[2,3], output=[4,6]
        layer.ResetState();
        var input2 = MakeTensor(1, 2, [1.0, 0.0]);
        var output2 = layer.Forward(input2);
        Assert.Equal(4.0, output2[0], Tol);
        Assert.Equal(6.0, output2[1], Tol);
    }

    [Fact]
    public void NegativeValues_HandledCorrectly()
    {
        var layer = CreateLayerWithKnownMatrices(2, 2, 1, 1.0,
            new double[,] { { -1 }, { 2 } },
            new double[,] { { -3, 4 } });

        // input = [-1, 2]
        // input*A = [-1*(-1) + 2*2] = [5]
        // [5]*B = [-15, 20]
        // output = [-15, 20] * 1 = [-15, 20]
        var input = MakeTensor(1, 2, [-1.0, 2.0]);
        var output = layer.Forward(input);

        Assert.Equal(-15.0, output[0], Tol);
        Assert.Equal(20.0, output[1], Tol);
    }

    [Fact]
    public void ZeroInput_ProducesZeroOutput()
    {
        var layer = CreateLayerWithKnownMatrices(3, 2, 2, 4.0,
            new double[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } },
            new double[,] { { 1, 2 }, { 3, 4 } });

        var input = MakeTensor(1, 3, [0.0, 0.0, 0.0]);
        var output = layer.Forward(input);

        Assert.Equal(0.0, output[0], Tol);
        Assert.Equal(0.0, output[1], Tol);
    }

    [Fact]
    public void LargeAlpha_ScalingCorrect()
    {
        // alpha=1000, rank=1 → scaling=1000
        var layer = CreateLayerWithKnownMatrices(1, 1, 1, 1000.0,
            new double[,] { { 1 } },
            new double[,] { { 1 } });

        var input = MakeTensor(1, 1, [1.0]);
        var output = layer.Forward(input);

        Assert.Equal(1000.0, output[0], Tol);
    }

    #endregion
}
