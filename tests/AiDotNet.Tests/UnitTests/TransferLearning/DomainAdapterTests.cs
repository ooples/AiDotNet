using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TransferLearning.DomainAdaptation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TransferLearning;

public class DomainAdapterTests
{
    [Fact]
    public void MMDDomainAdapter_InitializesCorrectly()
    {
        // Arrange & Act
        var adapter = new MMDDomainAdapter<double>();

        // Assert
        Assert.Equal("Maximum Mean Discrepancy (MMD)", adapter.AdaptationMethod);
        Assert.False(adapter.RequiresTraining);
    }

    [Fact]
    public void MMDDomainAdapter_ComputesDomainDiscrepancy()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>();
        var sourceData = new Matrix<double>(20, 4);
        var targetData = new Matrix<double>(20, 4);

        // Fill with similar data (low discrepancy expected)
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                sourceData[i, j] = i * 0.1 + j;
                targetData[i, j] = i * 0.1 + j + 0.05; // Small shift
            }
        }

        // Act
        var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

        // Assert
        Assert.True(discrepancy >= 0.0); // MMD is always non-negative
    }

    [Fact]
    public void CORALDomainAdapter_InitializesCorrectly()
    {
        // Arrange & Act
        var adapter = new CORALDomainAdapter<double>();

        // Assert
        Assert.Equal("CORAL (CORrelation ALignment)", adapter.AdaptationMethod);
        Assert.True(adapter.RequiresTraining);
    }

    [Fact]
    public void CORALDomainAdapter_AdaptsSourceData()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = new Matrix<double>(15, 3);
        var targetData = new Matrix<double>(15, 3);

        for (int i = 0; i < 15; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                sourceData[i, j] = i + j * 0.5;
                targetData[i, j] = i * 2 + j; // Different distribution
            }
        }

        // Act
        adapter.Train(sourceData, targetData);
        var adapted = adapter.AdaptSource(sourceData, targetData);

        // Assert
        Assert.Equal(sourceData.Rows, adapted.Rows);
        Assert.Equal(sourceData.Columns, adapted.Columns);
    }
}
