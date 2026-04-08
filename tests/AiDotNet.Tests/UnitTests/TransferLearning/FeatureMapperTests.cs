using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TransferLearning.FeatureMapping;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.TransferLearning;

public class FeatureMapperTests
{
    [Fact(Timeout = 60000)]
    public async Task LinearFeatureMapper_InitializesCorrectly()
    {
        // Arrange & Act
        var mapper = new LinearFeatureMapper<double>();

        // Assert
        Assert.False(mapper.IsTrained);
    }

    [Fact(Timeout = 60000)]
    public async Task LinearFeatureMapper_TrainsSuccessfully()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(10, 5);
        var targetData = new Matrix<double>(10, 3);

        // Fill with some data
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                sourceData[i, j] = i + j * 0.1;
            }
            for (int j = 0; j < 3; j++)
            {
                targetData[i, j] = i * 0.5 + j;
            }
        }

        // Act
        mapper.Train(sourceData, targetData);

        // Assert
        Assert.True(mapper.IsTrained);
        Assert.True(mapper.GetMappingConfidence() >= 0.0);
        Assert.True(mapper.GetMappingConfidence() <= 1.0);
    }

    [Fact(Timeout = 60000)]
    public async Task LinearFeatureMapper_MapToTarget_WorksAfterTraining()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(10, 5);
        var targetData = new Matrix<double>(10, 3);

        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                sourceData[i, j] = i + j * 0.1;
            }
            for (int j = 0; j < 3; j++)
            {
                targetData[i, j] = i * 0.5 + j;
            }
        }

        mapper.Train(sourceData, targetData);

        var testData = new Matrix<double>(5, 5);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                testData[i, j] = i + j;
            }
        }

        // Act
        var mapped = mapper.MapToTarget(testData, 3);

        // Assert
        Assert.Equal(5, mapped.Rows);
        Assert.Equal(3, mapped.Columns);
    }

    [Fact(Timeout = 60000)]
    public async Task LinearFeatureMapper_ThrowsWhenNotTrained()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var testData = new Matrix<double>(5, 5);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
        {
            mapper.MapToTarget(testData, 3);
        });
    }
}
