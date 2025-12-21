using AiDotNet.AutoML;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AutoML;

public class MultiFidelityAutoMLIntegrationTests
{
    [Fact]
    public async Task SearchAsync_MatrixVector_ProducesBestModelAndTrialHistory()
    {
        // Arrange
        var trainX = new Matrix<double>(60, 2);
        var trainY = new Vector<double>(60);

        for (int i = 0; i < 60; i++)
        {
            double x1 = i / 10.0;
            double x2 = x1 * x1;
            trainX[i, 0] = x1;
            trainX[i, 1] = x2;
            trainY[i] = (2.0 * x1) + (3.0 * x2);
        }

        var valX = new Matrix<double>(20, 2);
        var valY = new Vector<double>(20);

        for (int i = 0; i < 20; i++)
        {
            double x1 = (i + 0.25) / 10.0;
            double x2 = x1 * x1;
            valX[i, 0] = x1;
            valX[i, 1] = x2;
            valY[i] = (2.0 * x1) + (3.0 * x2);
        }

        var options = new AutoMLMultiFidelityOptions
        {
            TrainingFractions = new[] { 0.25, 0.5, 1.0 },
            ReductionFactor = 3.0
        };

        var autoML = new MultiFidelityAutoML<double, Matrix<double>, Vector<double>>(random: new Random(123), options: options);
        autoML.TrialLimit = 9;
        autoML.SetCandidateModels(new List<ModelType> { ModelType.MultipleRegression });

        // Act
        IFullModel<double, Matrix<double>, Vector<double>> best =
            await autoML.SearchAsync(trainX, trainY, valX, valY, TimeSpan.FromSeconds(15));

        // Assert
        Assert.NotNull(best);
        Assert.NotNull(autoML.BestModel);
        Assert.Equal(9, autoML.GetTrialHistory().Count);
        Assert.True(autoML.BestScore >= 0.0);
        Assert.True(autoML.BestScore < double.PositiveInfinity);
    }
}
