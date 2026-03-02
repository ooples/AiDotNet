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
    public async Task KittiDataLoader_LoadsBinaryPointCloudsWithLabels()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "kitti_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string veloDir = Path.Combine(tempDir, "training", "velodyne");
            string labelDir = Path.Combine(tempDir, "training", "label_2");
            Directory.CreateDirectory(veloDir);
            Directory.CreateDirectory(labelDir);

            // KITTI classes: Car=0, Van=1, Truck=2, Pedestrian=3, Person_sitting=4, Cyclist=5, Tram=6, Misc=7
            string[] labelContents =
            {
                // Frame 0: 2 Cars, 1 Pedestrian -> dominant = Car (0)
                "Car 0.00 0 -1.82 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59\n" +
                "Car 0.00 0 -1.50 400.00 170.00 450.00 200.00 1.50 1.60 3.50 1.00 1.70 30.00 -1.20\n" +
                "Pedestrian 0.00 0 0.21 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01",
                // Frame 1: 3 Pedestrians -> dominant = Pedestrian (3)
                "Pedestrian 0.00 0 0.21 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n" +
                "Pedestrian 0.00 0 0.50 500.00 150.00 550.00 300.00 1.80 0.50 1.10 2.00 1.50 10.00 0.10\n" +
                "Pedestrian 0.00 0 -0.30 600.00 160.00 650.00 310.00 1.75 0.45 1.15 1.50 1.60 12.00 -0.20",
                // Frame 2: 1 Cyclist -> dominant = Cyclist (5)
                "Cyclist 0.00 0 -1.00 300.00 180.00 350.00 250.00 1.70 0.60 1.80 0.50 1.40 15.00 -0.80"
            };

            // Create synthetic .bin files (4 floats per point: x, y, z, reflectance)
            for (int i = 0; i < 3; i++)
            {
                int numPoints = 100;
                byte[] binData = new byte[numPoints * 4 * 4];
                for (int p = 0; p < numPoints; p++)
                {
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.1)), 0, binData, p * 16, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.2)), 0, binData, p * 16 + 4, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes((float)(p * 0.3)), 0, binData, p * 16 + 8, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(0.5f), 0, binData, p * 16 + 12, 4);
                }
                File.WriteAllBytes(Path.Combine(veloDir, $"{i:D6}.bin"), binData);
                File.WriteAllText(Path.Combine(labelDir, $"{i:D6}.txt"), labelContents[i]);
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

            // Verify labels are parsed from label files (not just sample indices)
            // Use GetBatches to retrieve all data in a single batch, no shuffling
            var batch = loader.GetBatches(batchSize: 3, shuffle: false).First();
            var labels = batch.Labels;
            // Frame 0: Car=0, Frame 1: Pedestrian=3, Frame 2: Cyclist=5
            Assert.Equal(0.0, labels[0, 0], 0.01); // Car
            Assert.Equal(3.0, labels[1, 0], 0.01); // Pedestrian
            Assert.Equal(5.0, labels[2, 0], 0.01); // Cyclist
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
    public async Task WaymoDataLoader_LoadsBinaryDataWithLabels()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "waymo_test_" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            string veloDir = Path.Combine(tempDir, "training", "velodyne");
            string labelDir = Path.Combine(tempDir, "training", "label");
            Directory.CreateDirectory(veloDir);
            Directory.CreateDirectory(labelDir);

            // Waymo classes: 1=Vehicle(->0), 2=Pedestrian(->1), 3=Cyclist(->2), 4=Sign(->3)
            string[] labelContents =
            {
                // Frame 0: 2 Vehicles (class_id=1) -> dominant = Vehicle (0)
                "1 10.0 5.0 0.5 4.5 2.0 1.5 0.1\n1 20.0 -3.0 0.4 4.0 1.8 1.4 -0.5",
                // Frame 1: 1 Pedestrian (class_id=2) + 2 Cyclists (class_id=3) -> dominant = Cyclist (2)
                "2 5.0 2.0 0.0 0.5 0.5 1.8 0.0\n3 8.0 1.0 0.2 1.8 0.6 1.7 0.3\n3 12.0 -1.0 0.1 1.7 0.5 1.6 -0.2"
            };

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
                File.WriteAllText(Path.Combine(labelDir, $"frame_{i:D6}.txt"), labelContents[i]);
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

            // Verify labels parsed from label files
            var batch = loader.GetBatches(batchSize: 2, shuffle: false).First();
            var labels = batch.Labels;
            Assert.Equal(0.0, labels[0, 0], 0.01); // Vehicle (class_id 1 -> index 0)
            Assert.Equal(2.0, labels[1, 0], 0.01); // Cyclist (class_id 3 -> index 2)
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
            string labelDir = Path.Combine(tempDir, "training", "label_2");
            Directory.CreateDirectory(veloDir);
            Directory.CreateDirectory(labelDir);

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
                // Simple label file with a Car in each frame
                File.WriteAllText(Path.Combine(labelDir, $"{i:D6}.txt"),
                    "Car 0.00 0 -1.82 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59");
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
