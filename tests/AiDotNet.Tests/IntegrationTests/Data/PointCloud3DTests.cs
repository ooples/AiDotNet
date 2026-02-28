using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Data.Geometry;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Data;

public class PointCloud3DTests
{
    [Fact]
    public void KittiOptions_DefaultValues()
    {
        var options = new KittiDataLoaderOptions();
        Assert.Equal(16384, options.PointsPerSample);
        Assert.True(options.IncludeReflectance);
    }

    [Fact]
    public void SemanticKittiOptions_DefaultValues()
    {
        var options = new SemanticKittiDataLoaderOptions();
        Assert.Equal(16384, options.PointsPerSample);
        Assert.Equal(28, options.NumClasses);
    }

    [Fact]
    public void WaymoOptions_DefaultValues()
    {
        var options = new WaymoDataLoaderOptions();
        Assert.Equal(65536, options.PointsPerSample);
        Assert.True(options.IncludeIntensity);
    }

    [Fact]
    public void NuScenesOptions_DefaultValues()
    {
        var options = new NuScenesDataLoaderOptions();
        Assert.Equal(32768, options.PointsPerSample);
        Assert.True(options.IncludeIntensity);
    }

    [Fact]
    public async Task KittiDataLoader_LoadsBinaryPointClouds()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "kitti_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string veloDir = Path.Combine(tempDir, "training", "velodyne");
            Directory.CreateDirectory(veloDir);

            // Create synthetic .bin files (4 floats per point: x, y, z, reflectance)
            for (int i = 0; i < 3; i++)
            {
                int numPoints = 100;
                byte[] binData = new byte[numPoints * 4 * 4]; // 4 floats * 4 bytes
                for (int p = 0; p < numPoints; p++)
                {
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.1)), 0, binData, p * 16, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.2)), 0, binData, p * 16 + 4, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.3)), 0, binData, p * 16 + 8, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(0.5f), 0, binData, p * 16 + 12, 4);
                }
                File.WriteAllBytes(Path.Combine(veloDir, $"{i:D6}.bin"), binData);
            }

            var options = new KittiDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                PointsPerSample = 100,
                IncludeReflectance = true,
                MaxSamples = 3
            };

            var loader = new KittiDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(3, loader.TotalCount);
            Assert.Equal(400, loader.FeatureCount); // 100 * 4
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task SemanticKittiDataLoader_LoadsPointsAndLabels()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "semkitti_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string veloDir = Path.Combine(tempDir, "sequences", "00", "velodyne");
            string lblDir = Path.Combine(tempDir, "sequences", "00", "labels");
            Directory.CreateDirectory(veloDir);
            Directory.CreateDirectory(lblDir);

            int numPoints = 50;
            for (int i = 0; i < 2; i++)
            {
                // Point cloud: 4 floats per point
                byte[] binData = new byte[numPoints * 16];
                for (int p = 0; p < numPoints; p++)
                {
                    Buffer.BlockCopy(BitConverter.GetBytes((float)p), 0, binData, p * 16, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)p), 0, binData, p * 16 + 4, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)p), 0, binData, p * 16 + 8, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(0.5f), 0, binData, p * 16 + 12, 4);
                }
                File.WriteAllBytes(Path.Combine(veloDir, $"{i:D6}.bin"), binData);

                // Labels: uint32 per point (lower 16 bits = semantic label)
                byte[] lblData = new byte[numPoints * 4];
                for (int p = 0; p < numPoints; p++)
                {
                    uint label = (uint)(p % 10); // 10 classes
                    Buffer.BlockCopy(BitConverter.GetBytes(label), 0, lblData, p * 4, 4);
                }
                File.WriteAllBytes(Path.Combine(lblDir, $"{i:D6}.label"), lblData);
            }

            var options = new SemanticKittiDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                PointsPerSample = 50,
                NumClasses = 10,
                MaxSamples = 2
            };

            var loader = new SemanticKittiDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(2, loader.TotalCount);
            Assert.Equal(150, loader.FeatureCount); // 50 * 3
            Assert.Equal(50, loader.OutputDimension);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task WaymoDataLoader_LoadsBinaryData()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "waymo_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string veloDir = Path.Combine(tempDir, "training", "velodyne");
            Directory.CreateDirectory(veloDir);

            int numPoints = 80;
            for (int i = 0; i < 2; i++)
            {
                byte[] binData = new byte[numPoints * 16];
                for (int p = 0; p < numPoints; p++)
                {
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.1)), 0, binData, p * 16, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.2)), 0, binData, p * 16 + 4, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.3)), 0, binData, p * 16 + 8, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(0.8f), 0, binData, p * 16 + 12, 4);
                }
                File.WriteAllBytes(Path.Combine(veloDir, $"frame_{i:D6}.bin"), binData);
            }

            var options = new WaymoDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                PointsPerSample = 80,
                IncludeIntensity = true,
                MaxSamples = 2
            };

            var loader = new WaymoDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(2, loader.TotalCount);
            Assert.Equal(320, loader.FeatureCount); // 80 * 4
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task NuScenesDataLoader_LoadsBinaryData()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "nuscenes_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string sampleDir = Path.Combine(tempDir, "samples", "LIDAR_TOP");
            Directory.CreateDirectory(sampleDir);

            int numPoints = 60;
            for (int i = 0; i < 2; i++)
            {
                // nuScenes: 5 floats per point (x, y, z, intensity, ring_index)
                byte[] binData = new byte[numPoints * 20]; // 5 * 4 bytes
                for (int p = 0; p < numPoints; p++)
                {
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.1)), 0, binData, p * 20, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.2)), 0, binData, p * 20 + 4, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.3)), 0, binData, p * 20 + 8, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(0.5f), 0, binData, p * 20 + 12, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)i), 0, binData, p * 20 + 16, 4);
                }
                File.WriteAllBytes(Path.Combine(sampleDir, $"sample_{i:D6}.bin"), binData);
            }

            var options = new NuScenesDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                PointsPerSample = 60,
                IncludeIntensity = true,
                MaxSamples = 2
            };

            var loader = new NuScenesDataLoader<double>(options);
            await loader.LoadAsync();

            Assert.Equal(2, loader.TotalCount);
            Assert.Equal(240, loader.FeatureCount); // 60 * 4
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public async Task KittiDataLoader_SplitsData()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "kitti_split_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string veloDir = Path.Combine(tempDir, "training", "velodyne");
            Directory.CreateDirectory(veloDir);

            for (int i = 0; i < 10; i++)
            {
                byte[] binData = new byte[50 * 16];
                for (int p = 0; p < 50; p++)
                {
                    Buffer.BlockCopy(BitConverter.GetBytes((float)p), 0, binData, p * 16, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)p), 0, binData, p * 16 + 4, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)p), 0, binData, p * 16 + 8, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(0.5f), 0, binData, p * 16 + 12, 4);
                }
                File.WriteAllBytes(Path.Combine(veloDir, $"{i:D6}.bin"), binData);
            }

            var options = new KittiDataLoaderOptions
            {
                DataPath = tempDir,
                AutoDownload = false,
                PointsPerSample = 50
            };

            var loader = new KittiDataLoader<double>(options);
            await loader.LoadAsync();

            var (train, val, test) = loader.Split(0.7, 0.15, seed: 42);
            Assert.Equal(7, train.TotalCount);
            Assert.Equal(1, val.TotalCount);
            Assert.Equal(2, test.TotalCount);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }
}
