using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AutoML;

public class EvolutionaryAutoMLIntegrationTests
{
    [Fact]
    public async Task SearchAsync_MatrixVector_ProducesBestModelAndTrialHistory()
    {
        // Arrange
        var trainX = new Matrix<double>(20, 2);
        var trainY = new Vector<double>(20);

        for (int i = 0; i < 20; i++)
        {
            double x1 = i;
            double x2 = i * i;
            trainX[i, 0] = x1;
            trainX[i, 1] = x2;
            trainY[i] = (2.0 * x1) + (3.0 * x2);
        }

        var valX = new Matrix<double>(10, 2);
        var valY = new Vector<double>(10);

        for (int i = 0; i < 10; i++)
        {
            double x1 = i + 0.25;
            double x2 = (i + 0.25) * (i + 0.25);
            valX[i, 0] = x1;
            valX[i, 1] = x2;
            valY[i] = (2.0 * x1) + (3.0 * x2);
        }

        var autoML = new EvolutionaryAutoML<double, Matrix<double>, Vector<double>>(random: RandomHelper.CreateSeededRandom(123));
        autoML.TrialLimit = 3;
        autoML.SetCandidateModels(new List<ModelType> { ModelType.MultipleRegression });

        // Act
        IFullModel<double, Matrix<double>, Vector<double>> best =
            await autoML.SearchAsync(trainX, trainY, valX, valY, TimeSpan.FromSeconds(10));

        // Assert
        Assert.NotNull(best);
        Assert.NotNull(autoML.BestModel);
        Assert.Equal(3, autoML.GetTrialHistory().Count);
        Assert.True(autoML.BestScore >= 0.0);
        Assert.True(autoML.BestScore < double.PositiveInfinity);
    }
}

