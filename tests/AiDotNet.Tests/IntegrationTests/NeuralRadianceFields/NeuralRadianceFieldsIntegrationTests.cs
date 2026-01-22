using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralRadianceFields;

/// <summary>
/// Integration tests for neural radiance field models.
/// </summary>
public class NeuralRadianceFieldsIntegrationTests
{
    [Fact]
    public void InstantNGP_RenderRays_ProducesFiniteColors()
    {
        var options = new InstantNGPOptions<double>
        {
            HashTableSize = 128,
            NumLevels = 4,
            FeaturesPerLevel = 2,
            FinestResolution = 32,
            CoarsestResolution = 4,
            MlpHiddenDim = 16,
            MlpNumLayers = 2,
            UseOccupancyGrid = true,
            OccupancyGridResolution = 8,
            OccupancyDecay = 0.95,
            OccupancyThreshold = 0.01,
            OccupancyUpdateInterval = 1,
            OccupancySamplesPerCell = 1,
            OccupancyJitter = 1.0,
            LearningRate = 1e-2,
            RenderSamples = 8
        };
        var model = new InstantNGP<double>(options);

        var rayOrigins = CreateRayOrigins(numRays: 4, 0.5, 0.5, 0.5);
        var rayDirections = CreateRayDirections(numRays: 4, seed: 123);
        var output = model.RenderRays(
            rayOrigins,
            rayDirections,
            numSamples: 8,
            nearBound: 0.0,
            farBound: 1.0);

        Assert.Equal(4, output.Shape[0]);
        Assert.Equal(3, output.Shape[1]);

        for (int i = 0; i < output.Length; i++)
        {
            double value = output[i];
            Assert.False(double.IsNaN(value));
            Assert.False(double.IsInfinity(value));
            Assert.True(value >= 0.0);
            Assert.True(value <= 1.0 + 1e-6);
        }
    }

    [Fact]
    public void InstantNGP_Train_ChangesSerializedState()
    {
        var options = new InstantNGPOptions<double>
        {
            HashTableSize = 128,
            NumLevels = 4,
            FeaturesPerLevel = 2,
            FinestResolution = 32,
            CoarsestResolution = 4,
            MlpHiddenDim = 16,
            MlpNumLayers = 2,
            UseOccupancyGrid = true,
            OccupancyGridResolution = 8,
            OccupancyDecay = 0.95,
            OccupancyThreshold = 0.01,
            OccupancyUpdateInterval = 1,
            OccupancySamplesPerCell = 1,
            OccupancyJitter = 1.0,
            LearningRate = 1e-2
        };
        var model = new InstantNGP<double>(options);

        var input = CreateInstantNgpInput(numPoints: 16, seed: 7);
        var expected = new Tensor<double>(new double[16 * 4], new[] { 16, 4 });

        var before = model.Serialize();
        model.Train(input, expected);
        var after = model.Serialize();

        Assert.False(before.SequenceEqual(after));

        var rayOrigins = CreateRayOrigins(numRays: 2, 0.4, 0.6, 0.5);
        var rayDirections = CreateRayDirections(numRays: 2, seed: 31);
        var output = model.RenderRays(
            rayOrigins,
            rayDirections,
            numSamples: 4,
            nearBound: 0.0,
            farBound: 1.0);

        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(3, output.Shape[1]);
    }

    [Fact]
    public void NeRF_Train_ProducesFiniteOutput()
    {
        // Create NeRF with individual parameters (golden standard pattern)
        var model = new NeRF<double>(
            positionEncodingLevels: 2,
            directionEncodingLevels: 2,
            hiddenDim: 16,
            numLayers: 2,
            colorHiddenDim: 8,
            colorNumLayers: 1,
            useHierarchicalSampling: false,
            renderSamples: 8,
            hierarchicalSamples: 0,
            renderNearBound: 0.0,
            renderFarBound: 1.0,
            learningRate: 1e-3);

        int numPoints = 16;
        var input = CreateInstantNgpInput(numPoints: numPoints, seed: 19);
        var expected = new Tensor<double>(new double[numPoints * 4], new[] { numPoints, 4 });

        model.Train(input, expected);
        var output = model.Predict(input);

        Assert.Equal(numPoints, output.Shape[0]);
        Assert.Equal(4, output.Shape[1]);
        AssertAllFinite(output);
    }

    [Fact]
    public void GaussianSplatting_Train_ChangesRenderOutput()
    {
        var pointCloud = new Matrix<double>(2, 3);
        pointCloud[0, 0] = 0.0;
        pointCloud[0, 1] = 0.0;
        pointCloud[0, 2] = 0.0;
        pointCloud[1, 0] = 0.1;
        pointCloud[1, 1] = 0.0;
        pointCloud[1, 2] = 0.0;

        var colors = new Matrix<double>(2, 3);
        colors[0, 0] = 1.0;
        colors[0, 1] = 0.0;
        colors[0, 2] = 0.0;
        colors[1, 0] = 0.0;
        colors[1, 1] = 1.0;
        colors[1, 2] = 0.0;

        var options = new GaussianSplattingOptions
        {
            UseSphericalHarmonics = false,
            ShDegree = 0,
            ColorLearningRate = 0.2,
            OpacityLearningRate = 0.2,
            PositionLearningRate = 0.05
        };
        var model = new GaussianSplatting<double>(
            options,
            initialPointCloud: pointCloud,
            initialColors: colors);

        var cameraPosition = new Vector<double>(new[] { 0.0, 0.0, -1.0 });
        var cameraRotation = CreateIdentityRotation();
        double focalLength = 8.0;
        int width = 8;
        int height = 8;

        var cameraInput = CreateCameraInput(cameraPosition, cameraRotation, focalLength);

        var before = model.RenderImage(cameraPosition, cameraRotation, width, height, focalLength);
        var expected = InvertImage(before);

        model.Train(cameraInput, expected);

        var after = model.RenderImage(cameraPosition, cameraRotation, width, height, focalLength);
        double delta = SumAbsoluteDifference(before, after);

        Assert.True(delta > 1e-8, $"delta={delta}");
    }

    private static Tensor<double> CreateInstantNgpInput(int numPoints, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var data = new double[numPoints * 6];

        for (int i = 0; i < numPoints; i++)
        {
            double px = random.NextDouble();
            double py = random.NextDouble();
            double pz = random.NextDouble();
            double dx = random.NextDouble() * 2.0 - 1.0;
            double dy = random.NextDouble() * 2.0 - 1.0;
            double dz = random.NextDouble() * 2.0 - 1.0;
            Normalize(ref dx, ref dy, ref dz);

            int baseIdx = i * 6;
            data[baseIdx] = px;
            data[baseIdx + 1] = py;
            data[baseIdx + 2] = pz;
            data[baseIdx + 3] = dx;
            data[baseIdx + 4] = dy;
            data[baseIdx + 5] = dz;
        }

        return new Tensor<double>(data, new[] { numPoints, 6 });
    }

    private static Tensor<double> CreateRayOrigins(int numRays, double x, double y, double z)
    {
        var data = new double[numRays * 3];
        for (int i = 0; i < numRays; i++)
        {
            int baseIdx = i * 3;
            data[baseIdx] = x;
            data[baseIdx + 1] = y;
            data[baseIdx + 2] = z;
        }

        return new Tensor<double>(data, new[] { numRays, 3 });
    }

    private static Tensor<double> CreateRayDirections(int numRays, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var data = new double[numRays * 3];

        for (int i = 0; i < numRays; i++)
        {
            double x = random.NextDouble() * 2.0 - 1.0;
            double y = random.NextDouble() * 2.0 - 1.0;
            double z = random.NextDouble() * 2.0 - 1.0;
            Normalize(ref x, ref y, ref z);

            int baseIdx = i * 3;
            data[baseIdx] = x;
            data[baseIdx + 1] = y;
            data[baseIdx + 2] = z;
        }

        return new Tensor<double>(data, new[] { numRays, 3 });
    }

    private static Tensor<double> CreateCameraInput(
        Vector<double> cameraPosition,
        Matrix<double> cameraRotation,
        double focalLength)
    {
        var data = new double[13];
        data[0] = cameraPosition[0];
        data[1] = cameraPosition[1];
        data[2] = cameraPosition[2];

        int offset = 3;
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                data[offset++] = cameraRotation[r, c];
            }
        }

        data[12] = focalLength;

        return new Tensor<double>(data, new[] { 1, 13 });
    }

    private static Matrix<double> CreateIdentityRotation()
    {
        var rotation = new Matrix<double>(3, 3);
        rotation[0, 0] = 1.0;
        rotation[1, 1] = 1.0;
        rotation[2, 2] = 1.0;
        return rotation;
    }

    private static Tensor<double> InvertImage(Tensor<double> image)
    {
        var data = new double[image.Length];
        for (int i = 0; i < image.Length; i++)
        {
            double value = image[i];
            data[i] = Clamp01(1.0 - value);
        }

        return new Tensor<double>(data, image.Shape);
    }

    private static double SumAbsoluteDifference(Tensor<double> a, Tensor<double> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Tensor lengths must match.");
        }

        double total = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            total += Math.Abs(a[i] - b[i]);
        }

        return total;
    }

    private static void Normalize(ref double x, ref double y, ref double z)
    {
        double norm = Math.Sqrt(x * x + y * y + z * z);
        if (norm <= 0.0)
        {
            x = 0.0;
            y = 0.0;
            z = 1.0;
            return;
        }

        double inv = 1.0 / norm;
        x *= inv;
        y *= inv;
        z *= inv;
    }

    private static double Clamp01(double value)
    {
        if (value <= 0.0)
        {
            return 0.0;
        }

        if (value >= 1.0)
        {
            return 1.0;
        }

        return value;
    }

    private static void AssertAllFinite(Tensor<double> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            double value = tensor[i];
            Assert.False(double.IsNaN(value));
            Assert.False(double.IsInfinity(value));
        }
    }
}
