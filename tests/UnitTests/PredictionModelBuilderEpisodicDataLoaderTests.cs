using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests;

/// <summary>
/// Unit tests for episodic data loader integration with PredictionModelBuilder.
/// </summary>
public class PredictionModelBuilderEpisodicDataLoaderTests
{
    #region Test Helper Methods

    /// <summary>
    /// Creates a synthetic dataset for testing.
    /// </summary>
    private (Matrix<double> X, Vector<double> Y) CreateTestDataset(int numClasses, int examplesPerClass, int numFeatures)
    {
        int totalExamples = numClasses * examplesPerClass;
        var X = new Matrix<double>(totalExamples, numFeatures);
        var Y = new Vector<double>(totalExamples);

        for (int classIdx = 0; classIdx < numClasses; classIdx++)
        {
            for (int exampleIdx = 0; exampleIdx < examplesPerClass; exampleIdx++)
            {
                int rowIdx = classIdx * examplesPerClass + exampleIdx;

                for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++)
                {
                    X[rowIdx, featureIdx] = classIdx * 1000 + exampleIdx * 10 + featureIdx;
                }

                Y[rowIdx] = classIdx;
            }
        }

        return (X, Y);
    }

    #endregion

    #region ConfigureEpisodicDataLoader Tests

    [Fact]
    public void ConfigureEpisodicDataLoader_WithUniformLoader_StoresLoader()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 25, numFeatures: 10);
        var loader = new UniformEpisodicDataLoader<double>(X, Y);
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

        // Act
        var result = builder.ConfigureEpisodicDataLoader(loader);

        // Assert - Should return builder for method chaining
        Assert.NotNull(result);
        Assert.IsAssignableFrom<IPredictionModelBuilder<double, Matrix<double>, Vector<double>>>(result);
    }

    [Fact]
    public void ConfigureEpisodicDataLoader_WithBalancedLoader_StoresLoader()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 25, numFeatures: 10);
        var loader = new BalancedEpisodicDataLoader<double>(X, Y);
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

        // Act
        var result = builder.ConfigureEpisodicDataLoader(loader);

        // Assert
        Assert.NotNull(result);
        Assert.IsAssignableFrom<IPredictionModelBuilder<double, Matrix<double>, Vector<double>>>(result);
    }

    [Fact]
    public void ConfigureEpisodicDataLoader_WithStratifiedLoader_StoresLoader()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 25, numFeatures: 10);
        var loader = new StratifiedEpisodicDataLoader<double>(X, Y);
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

        // Act
        var result = builder.ConfigureEpisodicDataLoader(loader);

        // Assert
        Assert.NotNull(result);
        Assert.IsAssignableFrom<IPredictionModelBuilder<double, Matrix<double>, Vector<double>>>(result);
    }

    [Fact]
    public void ConfigureEpisodicDataLoader_WithCurriculumLoader_StoresLoader()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 30, numFeatures: 10);
        var loader = new CurriculumEpisodicDataLoader<double>(X, Y);
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

        // Act
        var result = builder.ConfigureEpisodicDataLoader(loader);

        // Assert
        Assert.NotNull(result);
        Assert.IsAssignableFrom<IPredictionModelBuilder<double, Matrix<double>, Vector<double>>>(result);
    }

    [Fact]
    public void ConfigureEpisodicDataLoader_WithNull_DoesNotThrow()
    {
        // Arrange
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

        // Act - Following the pattern, Configure methods don't validate - just set and return
        var result = builder.ConfigureEpisodicDataLoader(null!);

        // Assert - Should not throw, just set to null (following pattern of other Configure methods)
        Assert.NotNull(result);
    }

    [Fact]
    public void ConfigureEpisodicDataLoader_MethodChaining_WorksCorrectly()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 25, numFeatures: 10);
        var loader = new UniformEpisodicDataLoader<double>(X, Y);

        // Act - Method chaining
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureEpisodicDataLoader(loader);

        // Assert - Should work without errors
        Assert.NotNull(builder);
    }

    #endregion

    #region Interface Implementation Tests

    [Fact]
    public void UniformEpisodicDataLoader_ImplementsInterface_Correctly()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 25, numFeatures: 10);

        // Act
        IEpisodicDataLoader<double> loader = new UniformEpisodicDataLoader<double>(X, Y);

        // Assert - Can call GetNextTask through interface
        var task = loader.GetNextTask();
        Assert.NotNull(task);
        Assert.NotNull(task.SupportSetX);
        Assert.NotNull(task.SupportSetY);
        Assert.NotNull(task.QuerySetX);
        Assert.NotNull(task.QuerySetY);
    }

    [Fact]
    public void BalancedEpisodicDataLoader_ImplementsInterface_Correctly()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 25, numFeatures: 10);

        // Act
        IEpisodicDataLoader<double> loader = new BalancedEpisodicDataLoader<double>(X, Y);

        // Assert
        var task = loader.GetNextTask();
        Assert.NotNull(task);
    }

    [Fact]
    public void StratifiedEpisodicDataLoader_ImplementsInterface_Correctly()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 25, numFeatures: 10);

        // Act
        IEpisodicDataLoader<double> loader = new StratifiedEpisodicDataLoader<double>(X, Y);

        // Assert
        var task = loader.GetNextTask();
        Assert.NotNull(task);
    }

    [Fact]
    public void CurriculumEpisodicDataLoader_ImplementsInterface_Correctly()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 30, numFeatures: 10);

        // Act
        IEpisodicDataLoader<double> loader = new CurriculumEpisodicDataLoader<double>(X, Y);

        // Assert
        var task = loader.GetNextTask();
        Assert.NotNull(task);
    }

    [Fact]
    public void ConfigureEpisodicDataLoader_AcceptsAnyImplementation_ThroughInterface()
    {
        // Arrange
        var (X, Y) = CreateTestDataset(numClasses: 10, examplesPerClass: 25, numFeatures: 10);
        IEpisodicDataLoader<double> loader = new UniformEpisodicDataLoader<double>(X, Y);
        var builder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

        // Act
        var result = builder.ConfigureEpisodicDataLoader(loader);

        // Assert - Should accept any IEpisodicDataLoader<T> implementation
        Assert.NotNull(result);
    }

    #endregion
}
