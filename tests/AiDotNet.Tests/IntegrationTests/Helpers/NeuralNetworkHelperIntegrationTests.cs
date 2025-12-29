using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for NeuralNetworkHelper to verify neural network utility operations.
/// </summary>
public class NeuralNetworkHelperIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const float FloatTolerance = 1e-5f;

    #region EuclideanDistance Tests

    [Fact]
    public void EuclideanDistance_IdenticalVectors_ReturnsZero()
    {
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        var distance = NeuralNetworkHelper<double>.EuclideanDistance(v1, v2);

        Assert.True(Math.Abs(distance) < Tolerance);
    }

    [Fact]
    public void EuclideanDistance_UnitVectors_ReturnsCorrectDistance()
    {
        var v1 = new Vector<double>(new[] { 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 3.0, 4.0 });

        var distance = NeuralNetworkHelper<double>.EuclideanDistance(v1, v2);

        Assert.True(Math.Abs(distance - 5.0) < Tolerance); // 3-4-5 triangle
    }

    [Fact]
    public void EuclideanDistance_OneDimensional_ReturnsAbsoluteDifference()
    {
        var v1 = new Vector<double>(new[] { 5.0 });
        var v2 = new Vector<double>(new[] { 2.0 });

        var distance = NeuralNetworkHelper<double>.EuclideanDistance(v1, v2);

        Assert.True(Math.Abs(distance - 3.0) < Tolerance);
    }

    [Fact]
    public void EuclideanDistance_NegativeValues_WorksCorrectly()
    {
        var v1 = new Vector<double>(new[] { -1.0, -2.0 });
        var v2 = new Vector<double>(new[] { 2.0, 2.0 });

        var distance = NeuralNetworkHelper<double>.EuclideanDistance(v1, v2);

        // sqrt((2-(-1))^2 + (2-(-2))^2) = sqrt(9 + 16) = 5
        Assert.True(Math.Abs(distance - 5.0) < Tolerance);
    }

    [Fact]
    public void EuclideanDistance_Float_WorksCorrectly()
    {
        var v1 = new Vector<float>(new[] { 0.0f, 0.0f, 0.0f });
        var v2 = new Vector<float>(new[] { 1.0f, 2.0f, 2.0f });

        var distance = NeuralNetworkHelper<float>.EuclideanDistance(v1, v2);

        // sqrt(1 + 4 + 4) = 3
        Assert.True(Math.Abs(distance - 3.0f) < FloatTolerance);
    }

    [Fact]
    public void EuclideanDistance_LargeVectors_WorksCorrectly()
    {
        var v1 = new Vector<double>(100);
        var v2 = new Vector<double>(100);

        for (int i = 0; i < 100; i++)
        {
            v1[i] = 0.0;
            v2[i] = 1.0;
        }

        var distance = NeuralNetworkHelper<double>.EuclideanDistance(v1, v2);

        // sqrt(100 * 1) = 10
        Assert.True(Math.Abs(distance - 10.0) < Tolerance);
    }

    #endregion

    #region GetDefaultLossFunction Tests

    [Fact]
    public void GetDefaultLossFunction_BinaryClassification_ReturnsBinaryCrossEntropy()
    {
        var lossFunction = NeuralNetworkHelper<double>.GetDefaultLossFunction(NeuralNetworkTaskType.BinaryClassification);

        Assert.NotNull(lossFunction);
        Assert.IsType<BinaryCrossEntropyLoss<double>>(lossFunction);
    }

    [Fact]
    public void GetDefaultLossFunction_MultiClassClassification_ReturnsCategoricalCrossEntropy()
    {
        var lossFunction = NeuralNetworkHelper<double>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification);

        Assert.NotNull(lossFunction);
        Assert.IsType<CategoricalCrossEntropyLoss<double>>(lossFunction);
    }

    [Fact]
    public void GetDefaultLossFunction_Regression_ReturnsMeanSquaredError()
    {
        var lossFunction = NeuralNetworkHelper<double>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression);

        Assert.NotNull(lossFunction);
        Assert.IsType<MeanSquaredErrorLoss<double>>(lossFunction);
    }

    [Fact]
    public void GetDefaultLossFunction_ImageSegmentation_ReturnsDiceLoss()
    {
        var lossFunction = NeuralNetworkHelper<double>.GetDefaultLossFunction(NeuralNetworkTaskType.ImageSegmentation);

        Assert.NotNull(lossFunction);
        Assert.IsType<DiceLoss<double>>(lossFunction);
    }

    [Fact]
    public void GetDefaultLossFunction_TimeSeriesForecasting_ReturnsMSE()
    {
        var lossFunction = NeuralNetworkHelper<double>.GetDefaultLossFunction(NeuralNetworkTaskType.TimeSeriesForecasting);

        Assert.NotNull(lossFunction);
        Assert.IsType<MeanSquaredErrorLoss<double>>(lossFunction);
    }

    [Fact]
    public void GetDefaultLossFunction_Custom_ReturnsMSE()
    {
        var lossFunction = NeuralNetworkHelper<double>.GetDefaultLossFunction(NeuralNetworkTaskType.Custom);

        Assert.NotNull(lossFunction);
        Assert.IsType<MeanSquaredErrorLoss<double>>(lossFunction);
    }

    [Fact]
    public void GetDefaultLossFunction_AllTaskTypes_ReturnsNonNull()
    {
        foreach (NeuralNetworkTaskType taskType in Enum.GetValues(typeof(NeuralNetworkTaskType)))
        {
            var lossFunction = NeuralNetworkHelper<double>.GetDefaultLossFunction(taskType);
            Assert.NotNull(lossFunction);
        }
    }

    #endregion

    #region GetDefaultActivationFunction Tests

    [Fact]
    public void GetDefaultActivationFunction_BinaryClassification_ReturnsSigmoid()
    {
        var activation = NeuralNetworkHelper<double>.GetDefaultActivationFunction(NeuralNetworkTaskType.BinaryClassification);

        Assert.NotNull(activation);
        Assert.IsType<SigmoidActivation<double>>(activation);
    }

    [Fact]
    public void GetDefaultActivationFunction_MultiClassClassification_ReturnsSoftmax()
    {
        var activation = NeuralNetworkHelper<double>.GetDefaultActivationFunction(NeuralNetworkTaskType.MultiClassClassification);

        Assert.NotNull(activation);
        Assert.IsType<SoftmaxActivation<double>>(activation);
    }

    [Fact]
    public void GetDefaultActivationFunction_Regression_ReturnsIdentity()
    {
        var activation = NeuralNetworkHelper<double>.GetDefaultActivationFunction(NeuralNetworkTaskType.Regression);

        Assert.NotNull(activation);
        Assert.IsType<IdentityActivation<double>>(activation);
    }

    [Fact]
    public void GetDefaultActivationFunction_ReinforcementLearning_ReturnsTanh()
    {
        var activation = NeuralNetworkHelper<double>.GetDefaultActivationFunction(NeuralNetworkTaskType.ReinforcementLearning);

        Assert.NotNull(activation);
        Assert.IsType<TanhActivation<double>>(activation);
    }

    [Fact]
    public void GetDefaultActivationFunction_AllTaskTypes_ReturnsNonNull()
    {
        foreach (NeuralNetworkTaskType taskType in Enum.GetValues(typeof(NeuralNetworkTaskType)))
        {
            var activation = NeuralNetworkHelper<double>.GetDefaultActivationFunction(taskType);
            Assert.NotNull(activation);
        }
    }

    #endregion

    #region GetDefaultVectorActivationFunction Tests

    [Fact]
    public void GetDefaultVectorActivationFunction_BinaryClassification_ReturnsSigmoid()
    {
        var activation = NeuralNetworkHelper<double>.GetDefaultVectorActivationFunction(NeuralNetworkTaskType.BinaryClassification);

        Assert.NotNull(activation);
        Assert.IsType<SigmoidActivation<double>>(activation);
    }

    [Fact]
    public void GetDefaultVectorActivationFunction_MultiClassClassification_ReturnsSoftmax()
    {
        var activation = NeuralNetworkHelper<double>.GetDefaultVectorActivationFunction(NeuralNetworkTaskType.MultiClassClassification);

        Assert.NotNull(activation);
        Assert.IsType<SoftmaxActivation<double>>(activation);
    }

    [Fact]
    public void GetDefaultVectorActivationFunction_AllTaskTypes_ReturnsNonNull()
    {
        foreach (NeuralNetworkTaskType taskType in Enum.GetValues(typeof(NeuralNetworkTaskType)))
        {
            var activation = NeuralNetworkHelper<double>.GetDefaultVectorActivationFunction(taskType);
            Assert.NotNull(activation);
        }
    }

    #endregion

    #region ApplyActivation Tests - Vector

    [Fact]
    public void ApplyActivation_Vector_NoActivation_ReturnsIdentity()
    {
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        var output = NeuralNetworkHelper<double>.ApplyActivation(input);

        Assert.Equal(3, output.Length);
        Assert.Equal(1.0, output[0]);
        Assert.Equal(2.0, output[1]);
        Assert.Equal(3.0, output[2]);
    }

    [Fact]
    public void ApplyActivation_Vector_WithSigmoid_AppliesSigmoid()
    {
        var input = new Vector<double>(new[] { 0.0 });
        var sigmoid = new SigmoidActivation<double>();

        var output = NeuralNetworkHelper<double>.ApplyActivation(input, scalarActivation: sigmoid);

        // sigmoid(0) = 0.5
        Assert.True(Math.Abs(output[0] - 0.5) < Tolerance);
    }

    [Fact]
    public void ApplyActivation_Vector_WithIdentity_ReturnsOriginal()
    {
        var input = new Vector<double>(new[] { -1.0, 0.0, 1.0 });
        var identity = new IdentityActivation<double>();

        var output = NeuralNetworkHelper<double>.ApplyActivation(input, scalarActivation: identity);

        Assert.Equal(-1.0, output[0]);
        Assert.Equal(0.0, output[1]);
        Assert.Equal(1.0, output[2]);
    }

    [Fact]
    public void ApplyActivation_Vector_WithTanh_AppliesTanh()
    {
        var input = new Vector<double>(new[] { 0.0 });
        var tanh = new TanhActivation<double>();

        var output = NeuralNetworkHelper<double>.ApplyActivation(input, scalarActivation: tanh);

        // tanh(0) = 0
        Assert.True(Math.Abs(output[0]) < Tolerance);
    }

    [Fact]
    public void ApplyActivation_Vector_WithVectorActivation_AppliesVectorActivation()
    {
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var softmax = new SoftmaxActivation<double>();

        var output = NeuralNetworkHelper<double>.ApplyActivation(input, vectorActivation: softmax);

        // Softmax outputs should sum to 1
        double sum = 0;
        for (int i = 0; i < output.Length; i++)
        {
            sum += output[i];
        }
        Assert.True(Math.Abs(sum - 1.0) < 0.01);
    }

    [Fact]
    public void ApplyActivation_Vector_VectorActivationTakesPrecedence()
    {
        var input = new Vector<double>(new[] { 0.0 });
        var sigmoid = new SigmoidActivation<double>();
        var identity = new IdentityActivation<double>();

        // Both provided, but vector activation should be used
        var output = NeuralNetworkHelper<double>.ApplyActivation(input, scalarActivation: sigmoid, vectorActivation: identity);

        // If vector activation (identity) is used, output should be 0, not 0.5
        Assert.Equal(0.0, output[0]);
    }

    #endregion

    #region ApplyActivation Tests - Tensor

    [Fact]
    public void ApplyActivation_Tensor_Rank1_NoActivation_ReturnsIdentity()
    {
        var input = new Tensor<double>(new[] { 3 });
        input[0] = 1.0;
        input[1] = 2.0;
        input[2] = 3.0;

        var output = NeuralNetworkHelper<double>.ApplyActivation(input);

        Assert.Equal(1, output.Rank);
        Assert.Equal(3, output.Shape[0]);
    }

    [Fact]
    public void ApplyActivation_Tensor_Rank2_ThrowsArgumentException()
    {
        var input = new Tensor<double>(new[] { 2, 3 }); // Rank 2

        Assert.Throws<ArgumentException>(() =>
            NeuralNetworkHelper<double>.ApplyActivation(input));
    }

    [Fact]
    public void ApplyActivation_Tensor_WithSigmoid_AppliesSigmoid()
    {
        var input = new Tensor<double>(new[] { 1 });
        input[0] = 0.0;
        var sigmoid = new SigmoidActivation<double>();

        var output = NeuralNetworkHelper<double>.ApplyActivation(input, scalarActivation: sigmoid);

        // sigmoid(0) = 0.5
        Assert.True(Math.Abs(output[0] - 0.5) < Tolerance);
    }

    #endregion

    #region ApplyOutputActivation Tests

    [Fact]
    public void ApplyOutputActivation_BinaryClassification_AppliesSigmoid()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            inputSize: 10,
            outputSize: 1);

        var output = new Tensor<double>(new[] { 2, 1 });
        output[0, 0] = 0.0;
        output[1, 0] = 0.0;

        NeuralNetworkHelper<double>.ApplyOutputActivation(output, architecture);

        // sigmoid(0) = 0.5
        Assert.True(Math.Abs(Convert.ToDouble(output[0, 0]) - 0.5) < 0.01);
        Assert.True(Math.Abs(Convert.ToDouble(output[1, 0]) - 0.5) < 0.01);
    }

    [Fact]
    public void ApplyOutputActivation_MultiClassClassification_AppliesSoftmax()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 10,
            outputSize: 3);

        var output = new Tensor<double>(new[] { 1, 3 });
        output[0, 0] = 1.0;
        output[0, 1] = 2.0;
        output[0, 2] = 3.0;

        NeuralNetworkHelper<double>.ApplyOutputActivation(output, architecture);

        // Softmax outputs should sum to 1
        double sum = Convert.ToDouble(output[0, 0]) + Convert.ToDouble(output[0, 1]) + Convert.ToDouble(output[0, 2]);
        Assert.True(Math.Abs(sum - 1.0) < 0.01);

        // Higher input should have higher output
        Assert.True(Convert.ToDouble(output[0, 2]) > Convert.ToDouble(output[0, 1]));
        Assert.True(Convert.ToDouble(output[0, 1]) > Convert.ToDouble(output[0, 0]));
    }

    [Fact]
    public void ApplyOutputActivation_Regression_NoChange()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        var output = new Tensor<double>(new[] { 2, 1 });
        output[0, 0] = 5.5;
        output[1, 0] = -3.2;

        NeuralNetworkHelper<double>.ApplyOutputActivation(output, architecture);

        // Values should be unchanged (identity activation)
        Assert.Equal(5.5, Convert.ToDouble(output[0, 0]));
        Assert.Equal(-3.2, Convert.ToDouble(output[1, 0]));
    }

    [Fact]
    public void ApplyOutputActivation_ReinforcementLearning_AppliesTanh()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.ReinforcementLearning,
            inputSize: 10,
            outputSize: 1);

        var output = new Tensor<double>(new[] { 1, 1 });
        output[0, 0] = 0.0;

        NeuralNetworkHelper<double>.ApplyOutputActivation(output, architecture);

        // tanh(0) = 0
        Assert.True(Math.Abs(Convert.ToDouble(output[0, 0])) < 0.01);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void GetDefaultLossFunction_Float_WorksCorrectly()
    {
        var lossFunction = NeuralNetworkHelper<float>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression);

        Assert.NotNull(lossFunction);
        Assert.IsType<MeanSquaredErrorLoss<float>>(lossFunction);
    }

    [Fact]
    public void GetDefaultActivationFunction_Float_WorksCorrectly()
    {
        var activation = NeuralNetworkHelper<float>.GetDefaultActivationFunction(NeuralNetworkTaskType.BinaryClassification);

        Assert.NotNull(activation);
        Assert.IsType<SigmoidActivation<float>>(activation);
    }

    [Fact]
    public void ApplyActivation_Float_Vector_WorksCorrectly()
    {
        var input = new Vector<float>(new[] { 0.0f });
        var sigmoid = new SigmoidActivation<float>();

        var output = NeuralNetworkHelper<float>.ApplyActivation(input, scalarActivation: sigmoid);

        Assert.True(Math.Abs(output[0] - 0.5f) < FloatTolerance);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void EuclideanDistance_EmptyVectors_ReturnsZero()
    {
        var v1 = new Vector<double>(0);
        var v2 = new Vector<double>(0);

        var distance = NeuralNetworkHelper<double>.EuclideanDistance(v1, v2);

        Assert.Equal(0.0, distance);
    }

    [Fact]
    public void ApplyActivation_Vector_EmptyInput_ReturnsEmpty()
    {
        var input = new Vector<double>(0);

        var output = NeuralNetworkHelper<double>.ApplyActivation(input);

        Assert.Equal(0, output.Length);
    }

    [Fact]
    public void ApplyActivation_Tensor_SingleElement_WorksCorrectly()
    {
        var input = new Tensor<double>(new[] { 1 });
        input[0] = 5.0;

        var output = NeuralNetworkHelper<double>.ApplyActivation(input);

        Assert.Equal(5.0, output[0]);
    }

    #endregion
}
