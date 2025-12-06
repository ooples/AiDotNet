using AiDotNet.NeuralRadianceFields.Data;
using AiDotNet.NeuralRadianceFields.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// Tests for NeRF implementation.
/// </summary>
public class NeRFTests
{
    [Fact]
    public void NeRF_Construction_CreatesModel()
    {
        // Arrange & Act
        var nerf = new NeRF<double>();

        // Assert
        Assert.NotNull(nerf);
    }

    [Fact]
    public void NeRF_QueryField_ReturnsRGBAndDensity()
    {
        // Arrange
        var nerf = new NeRF<double>();
        var positions = CreateSamplePositions(numPoints: 10);
        var directions = CreateSampleDirections(numPoints: 10);

        // Act
        var (rgb, density) = nerf.QueryField(positions, directions);

        // Assert
        Assert.NotNull(rgb);
        Assert.NotNull(density);
        Assert.Equal(10, rgb.Shape[0]);
        Assert.Equal(3, rgb.Shape[1]);
        Assert.Equal(10, density.Shape[0]);
    }

    [Fact]
    public void Ray_Construction_CreatesRay()
    {
        // Arrange
        var origin = new Vector<double>(new double[] { 0, 0, 0 });
        var direction = new Vector<double>(new double[] { 0, 0, -1 });
        var nearBound = 2.0;
        var farBound = 6.0;

        // Act
        var ray = new Ray<double>(origin, direction, nearBound, farBound);

        // Assert
        Assert.NotNull(ray);
        Assert.Equal(0.0, ray.Origin[0]);
        Assert.Equal(-1.0, ray.Direction[2]);
        Assert.Equal(2.0, ray.NearBound);
        Assert.Equal(6.0, ray.FarBound);
    }

    [Fact]
    public void Ray_PointAt_CalculatesCorrectPosition()
    {
        // Arrange
        var origin = new Vector<double>(new double[] { 1, 2, 3 });
        var direction = new Vector<double>(new double[] { 0, 0, 1 });
        var ray = new Ray<double>(origin, direction, 0, 10);

        // Act
        var point = ray.PointAt(5.0);

        // Assert
        Assert.Equal(1.0, point[0]);
        Assert.Equal(2.0, point[1]);
        Assert.Equal(8.0, point[2]);  // 3 + 5*1
    }

    [Fact]
    public void InstantNGP_Construction_CreatesModel()
    {
        // Arrange & Act
        var instantNGP = new InstantNGP<double>();

        // Assert
        Assert.NotNull(instantNGP);
    }

    [Fact]
    public void GaussianSplatting_Construction_CreatesModel()
    {
        // Arrange & Act
        var gaussianSplatting = new GaussianSplatting<double>();

        // Assert
        Assert.NotNull(gaussianSplatting);
    }

    [Fact]
    public void GaussianSplatting_InitializeFromPointCloud_CreatesGaussians()
    {
        // Arrange
        var pointCloud = CreateSamplePointCloudMatrix(numPoints: 100);
        var colors = CreateSampleColorMatrix(numPoints: 100);

        // Act
        var gaussianSplatting = new GaussianSplatting<double>(
            initialPointCloud: pointCloud,
            initialColors: colors
        );

        // Assert
        Assert.NotNull(gaussianSplatting);
        Assert.Equal(100, gaussianSplatting.GaussianCount);
    }

    private Tensor<double> CreateSamplePositions(int numPoints)
    {
        var random = new Random(42);
        var data = new double[numPoints * 3];

        for (int i = 0; i < numPoints * 3; i++)
        {
            data[i] = random.NextDouble() * 2 - 1;  // Range [-1, 1]
        }

        return new Tensor<double>(data, [numPoints, 3]);
    }

    private Tensor<double> CreateSampleDirections(int numPoints)
    {
        var random = new Random(43);
        var data = new double[numPoints * 3];

        for (int i = 0; i < numPoints; i++)
        {
            // Create random unit vector
            var x = random.NextDouble() * 2 - 1;
            var y = random.NextDouble() * 2 - 1;
            var z = random.NextDouble() * 2 - 1;
            var length = Math.Sqrt(x * x + y * y + z * z);

            data[i * 3] = x / length;
            data[i * 3 + 1] = y / length;
            data[i * 3 + 2] = z / length;
        }

        return new Tensor<double>(data, [numPoints, 3]);
    }

    private Matrix<double> CreateSamplePointCloudMatrix(int numPoints)
    {
        var random = new Random(44);
        var matrix = new Matrix<double>(numPoints, 3);

        for (int i = 0; i < numPoints; i++)
        {
            matrix[i, 0] = random.NextDouble() * 2 - 1;
            matrix[i, 1] = random.NextDouble() * 2 - 1;
            matrix[i, 2] = random.NextDouble() * 2 - 1;
        }

        return matrix;
    }

    private Matrix<double> CreateSampleColorMatrix(int numPoints)
    {
        var random = new Random(45);
        var matrix = new Matrix<double>(numPoints, 3);

        for (int i = 0; i < numPoints; i++)
        {
            matrix[i, 0] = random.NextDouble();
            matrix[i, 1] = random.NextDouble();
            matrix[i, 2] = random.NextDouble();
        }

        return matrix;
    }
}
