using AiDotNet.Enums;
using AiDotNet.Evaluation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Evaluation;

public class PredictionTypeInferenceFromTargetsTests
{
    [Fact]
    public void InferFromTargets_ReturnsMultiClass_ForOneHotTensor()
    {
        var values = new[]
        {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        };

        var tensor = Tensor<double>.FromVector(Vector<double>.FromArray(values), shape: [3, 3]);
        var predictionType = PredictionTypeInference.InferFromTargets<double, Tensor<double>>(tensor);

        Assert.Equal(PredictionType.MultiClass, predictionType);
    }

    [Fact]
    public void InferFromTargets_ReturnsMultiLabel_ForMultiHotTensor()
    {
        var values = new[]
        {
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0
        };

        var tensor = Tensor<double>.FromVector(Vector<double>.FromArray(values), shape: [3, 3]);
        var predictionType = PredictionTypeInference.InferFromTargets<double, Tensor<double>>(tensor);

        Assert.Equal(PredictionType.MultiLabel, predictionType);
    }

    [Fact]
    public void InferFromTargets_ReturnsRegression_ForDenseContinuousTensor()
    {
        var values = new[]
        {
            0.2, 0.3, 0.4,
            0.6, 0.7, 0.8
        };

        var tensor = Tensor<double>.FromVector(Vector<double>.FromArray(values), shape: [2, 3]);
        var predictionType = PredictionTypeInference.InferFromTargets<double, Tensor<double>>(tensor);

        Assert.Equal(PredictionType.Regression, predictionType);
    }
}

