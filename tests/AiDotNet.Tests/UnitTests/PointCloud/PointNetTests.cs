using AiDotNet.PointCloud.Data;
using AiDotNet.PointCloud.Models;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PointCloud;

/// <summary>
/// Tests for PointNet implementation.
/// </summary>
public class PointNetTests
{
    [Fact]
    public void PointNet_Construction_CreatesModel()
    {
        // Arrange & Act
        var pointNet = new PointNet<double>(numClasses: 10);

        // Assert
        Assert.NotNull(pointNet);
    }

    [Fact]
    public void PointNet_ExtractGlobalFeatures_ReturnsFeatureVector()
    {
        // Arrange
        var pointNet = new PointNet<double>(numClasses: 10);
        var pointCloud = CreateSamplePointCloud(numPoints: 100);

        // Act
        var features = pointNet.ExtractGlobalFeatures(pointCloud);

        // Assert
        Assert.NotNull(features);
        Assert.True(features.Length > 0);
    }

    [Fact]
    public void PointNet_ClassifyPointCloud_ReturnsClassProbabilities()
    {
        // Arrange
        var pointNet = new PointNet<double>(numClasses: 10);
        var pointCloud = CreateSamplePointCloud(numPoints: 100);

        // Act
        var probabilities = pointNet.ClassifyPointCloud(pointCloud);

        // Assert
        Assert.NotNull(probabilities);
        Assert.Equal(10, probabilities.Length);
    }

    [Fact]
    public void PointCloudData_FromCoordinates_CreatesPointCloud()
    {
        // Arrange
        var coordinates = new Matrix<double>(10, 3);
        for (int i = 0; i < 10; i++)
        {
            coordinates[i, 0] = i * 0.1;  // X
            coordinates[i, 1] = i * 0.2;  // Y
            coordinates[i, 2] = i * 0.3;  // Z
        }

        // Act
        var pointCloudData = PointCloudData<double>.FromCoordinates(coordinates);

        // Assert
        Assert.NotNull(pointCloudData);
        Assert.Equal(10, pointCloudData.NumPoints);
        Assert.Equal(3, pointCloudData.NumFeatures);
    }

    [Fact]
    public void PointCloudData_GetCoordinates_ExtractsPositions()
    {
        // Arrange
        var data = new double[10 * 6];  // 10 points with XYZ + RGB
        for (int i = 0; i < 10; i++)
        {
            data[i * 6] = i;      // X
            data[i * 6 + 1] = i;  // Y
            data[i * 6 + 2] = i;  // Z
            data[i * 6 + 3] = 1;  // R
            data[i * 6 + 4] = 0;  // G
            data[i * 6 + 5] = 0;  // B
        }
        var tensor = new Tensor<double>(data, [10, 6]);
        var pointCloudData = new PointCloudData<double>(tensor);

        // Act
        var coordinates = pointCloudData.GetCoordinates();

        // Assert
        Assert.NotNull(coordinates);
        Assert.Equal(10, coordinates.Shape[0]);
        Assert.Equal(3, coordinates.Shape[1]);
    }

    [Fact]
    public void PointCloudData_GetFeatures_ExtractsAdditionalFeatures()
    {
        // Arrange
        var data = new double[10 * 6];  // 10 points with XYZ + RGB
        var tensor = new Tensor<double>(data, [10, 6]);
        var pointCloudData = new PointCloudData<double>(tensor);

        // Act
        var features = pointCloudData.GetFeatures();

        // Assert
        Assert.NotNull(features);
        Assert.Equal(10, features.Shape[0]);
        Assert.Equal(3, features.Shape[1]);  // RGB features
    }

    private Tensor<double> CreateSamplePointCloud(int numPoints)
    {
        var random = RandomHelper.CreateSeededRandom(42);
        var data = new double[numPoints * 3];

        for (int i = 0; i < numPoints * 3; i++)
        {
            data[i] = random.NextDouble();
        }

        return new Tensor<double>(data, [numPoints, 3]);
    }
}
