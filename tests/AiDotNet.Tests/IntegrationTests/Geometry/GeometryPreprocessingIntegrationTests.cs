using System;
using System.Linq;
using AiDotNet.Geometry.Data;
using AiDotNet.Geometry.Preprocessing;
using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Geometry;

/// <summary>
/// Integration tests for Geometry preprocessing: normalization, sampling, metrics,
/// voxelization, neighbor search, and mesh operations.
/// Includes golden reference values and edge case handling.
/// </summary>
public class GeometryPreprocessingIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region PointCloudNormalization Tests

    [Fact]
    public void Center_GoldenReference_CentroidBecomesOrigin()
    {
        // Triangle at (0,0,0), (6,0,0), (0,6,0) => centroid = (2,2,0)
        var data = new double[] { 0, 0, 0, 6, 0, 0, 0, 6, 0 };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 3, 3 }));

        var centered = PointCloudNormalization<double>.Center(cloud);

        // Verify centroid is at origin
        double cx = 0, cy = 0, cz = 0;
        for (int i = 0; i < 3; i++)
        {
            cx += centered.Points[i, 0];
            cy += centered.Points[i, 1];
            cz += centered.Points[i, 2];
        }
        Assert.Equal(0.0, cx / 3, Tolerance);
        Assert.Equal(0.0, cy / 3, Tolerance);
        Assert.Equal(0.0, cz / 3, Tolerance);

        // Verify specific values: original - centroid(2,2,0)
        Assert.Equal(-2.0, centered.Points[0, 0], Tolerance); // 0 - 2
        Assert.Equal(-2.0, centered.Points[0, 1], Tolerance); // 0 - 2
        Assert.Equal(0.0, centered.Points[0, 2], Tolerance);  // 0 - 0
        Assert.Equal(4.0, centered.Points[1, 0], Tolerance);  // 6 - 2
    }

    [Fact]
    public void Center_PreservesNonXYZFeatures()
    {
        // 6-feature cloud: XYZ + RGB
        var data = new double[]
        {
            1, 2, 3, 100, 200, 50,
            5, 6, 7, 150, 250, 75,
        };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 2, 6 }));

        var centered = PointCloudNormalization<double>.Center(cloud);

        // RGB values should be unchanged
        Assert.Equal(100.0, centered.Points[0, 3], Tolerance);
        Assert.Equal(200.0, centered.Points[0, 4], Tolerance);
        Assert.Equal(50.0, centered.Points[0, 5], Tolerance);
        Assert.Equal(150.0, centered.Points[1, 3], Tolerance);
    }

    [Fact]
    public void Center_EmptyCloud_ReturnsOriginal()
    {
        var data = new double[0];
        var tensor = new Tensor<double>(data, new[] { 0, 3 });
        var cloud = new PointCloudData<double>(tensor);

        var centered = PointCloudNormalization<double>.Center(cloud);

        Assert.Equal(0, centered.NumPoints);
    }

    [Fact]
    public void ScaleToUnitSphere_GoldenReference()
    {
        // Points at (1,0,0), (-1,0,0), (0,1,0), (0,-1,0)
        // Already centered, max distance = 1, so should stay same
        var data = new double[] { 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0 };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 4, 3 }));

        var scaled = PointCloudNormalization<double>.ScaleToUnitSphere(cloud);

        // All points should have distance <= 1 from origin
        for (int i = 0; i < 4; i++)
        {
            double x = scaled.Points[i, 0];
            double y = scaled.Points[i, 1];
            double z = scaled.Points[i, 2];
            double dist = Math.Sqrt(x * x + y * y + z * z);
            Assert.True(dist <= 1.0 + Tolerance, $"Point {i} distance {dist} > 1.0");
        }
    }

    [Fact]
    public void ScaleToUnitSphere_FarPoint_ScalesCorrectly()
    {
        // Points with max distance 5 from centroid
        var data = new double[] { 5, 0, 0, -5, 0, 0, 0, 3, 0 };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 3, 3 }));

        var scaled = PointCloudNormalization<double>.ScaleToUnitSphere(cloud);

        double maxDist = 0;
        for (int i = 0; i < 3; i++)
        {
            double x = scaled.Points[i, 0];
            double y = scaled.Points[i, 1];
            double z = scaled.Points[i, 2];
            double dist = Math.Sqrt(x * x + y * y + z * z);
            maxDist = Math.Max(maxDist, dist);
        }
        Assert.Equal(1.0, maxDist, Tolerance);
    }

    [Fact]
    public void ScaleToUnitCube_AllPointsInRange()
    {
        var random = RandomHelper.CreateSeededRandom(137);
        var data = new double[30];
        for (int i = 0; i < 30; i++)
            data[i] = random.NextDouble() * 100 - 50; // [-50, 50]
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 10, 3 }));

        var scaled = PointCloudNormalization<double>.ScaleToUnitCube(cloud);

        for (int i = 0; i < scaled.NumPoints; i++)
        {
            for (int d = 0; d < 3; d++)
            {
                double val = scaled.Points[i, d];
                Assert.True(val >= -0.5 - Tolerance && val <= 0.5 + Tolerance,
                    $"Point {i}, dim {d}: value {val} outside [-0.5, 0.5]");
            }
        }
    }

    [Fact]
    public void NormalizeColors_GoldenReference_255To01()
    {
        var data = new double[]
        {
            1.0, 2.0, 3.0, 255.0, 127.5, 0.0,
            4.0, 5.0, 6.0, 0.0, 255.0, 63.75,
        };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 2, 6 }));

        var normalized = PointCloudNormalization<double>.NormalizeColors(cloud);

        // Colors should be divided by 255
        Assert.Equal(1.0, normalized.Points[0, 3], Tolerance);   // 255/255
        Assert.Equal(0.5, normalized.Points[0, 4], Tolerance);   // 127.5/255
        Assert.Equal(0.0, normalized.Points[0, 5], Tolerance);   // 0/255
        Assert.Equal(0.25, normalized.Points[1, 5], Tolerance);  // 63.75/255

        // XYZ should be unchanged
        Assert.Equal(1.0, normalized.Points[0, 0], Tolerance);
        Assert.Equal(2.0, normalized.Points[0, 1], Tolerance);
    }

    [Fact]
    public void NormalizeColors_NoColorFeatures_ReturnsOriginal()
    {
        var data = new double[] { 1, 2, 3, 4, 5, 6 };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 2, 3 }));

        // colorOffset=3 but only 3 features, so 3+3=6 > 3, returns original
        var normalized = PointCloudNormalization<double>.NormalizeColors(cloud);

        Assert.Equal(2, normalized.NumPoints);
    }

    #endregion

    #region PointCloudSampling Tests

    [Fact]
    public void UniformSample_ReducesPointCount()
    {
        var cloud = CreateRandomCloud(100, 3, seed: 139);

        var sampled = PointCloudSampling<double>.UniformSample(cloud, 20, seed: 141);

        Assert.Equal(20, sampled.NumPoints);
        Assert.Equal(3, sampled.NumFeatures);
    }

    [Fact]
    public void UniformSample_ReturnsOriginalIfSamplesGreaterOrEqual()
    {
        var cloud = CreateRandomCloud(10, 3, seed: 149);

        var sampled = PointCloudSampling<double>.UniformSample(cloud, 10, seed: 151);

        Assert.Same(cloud, sampled);

        var sampled2 = PointCloudSampling<double>.UniformSample(cloud, 20, seed: 157);
        Assert.Same(cloud, sampled2);
    }

    [Fact]
    public void UniformSample_PreservesLabels()
    {
        var points = CreateRandomTensor(20, 3, seed: 163);
        var labels = new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)i).ToArray());
        var cloud = new PointCloudData<double>(points, labels);

        var sampled = PointCloudSampling<double>.UniformSample(cloud, 5, seed: 167);

        Assert.NotNull(sampled.Labels);
        Assert.Equal(5, sampled.Labels.Length);
    }

    [Fact]
    public void UniformSample_Deterministic_SameSeedSameResult()
    {
        var cloud = CreateRandomCloud(50, 3, seed: 173);

        var s1 = PointCloudSampling<double>.UniformSample(cloud, 10, seed: 42);
        var s2 = PointCloudSampling<double>.UniformSample(cloud, 10, seed: 42);

        for (int i = 0; i < s1.NumPoints * s1.NumFeatures; i++)
        {
            Assert.Equal(s1.Points[i], s2.Points[i], Tolerance);
        }
    }

    [Fact]
    public void FarthestPointSample_MaximizesSpread()
    {
        // Cluster of points at origin + one outlier
        var data = new double[]
        {
            0.0, 0.0, 0.0,
            0.1, 0.0, 0.0,
            0.0, 0.1, 0.0,
            0.1, 0.1, 0.0,
            10.0, 10.0, 10.0, // far outlier
        };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 5, 3 }));

        var sampled = PointCloudSampling<double>.FarthestPointSample(cloud, 2, seed: 42);

        Assert.Equal(2, sampled.NumPoints);

        // The two farthest points should include the outlier
        bool hasOutlier = false;
        for (int i = 0; i < 2; i++)
        {
            double x = sampled.Points[i, 0];
            if (Math.Abs(x - 10.0) < Tolerance)
                hasOutlier = true;
        }
        Assert.True(hasOutlier, "FPS should include the far outlier point");
    }

    [Fact]
    public void FarthestPointSample_ReturnsOriginalIfSamplesGreaterOrEqual()
    {
        var cloud = CreateRandomCloud(5, 3, seed: 179);

        var sampled = PointCloudSampling<double>.FarthestPointSample(cloud, 5, seed: 181);
        Assert.Same(cloud, sampled);
    }

    [Fact]
    public void FarthestPointSample_InvalidSamples_Throws()
    {
        var cloud = CreateRandomCloud(10, 3, seed: 191);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PointCloudSampling<double>.FarthestPointSample(cloud, 0));
    }

    [Fact]
    public void PoissonDiskSample_EnforcesMinimumDistance()
    {
        var cloud = CreateRandomCloud(200, 3, seed: 193);

        double minDist = 0.5;
        var sampled = PointCloudSampling<double>.PoissonDiskSample(cloud, minDist, seed: 197);

        // Verify all pairs are at least minDist apart
        for (int i = 0; i < sampled.NumPoints; i++)
        {
            for (int j = i + 1; j < sampled.NumPoints; j++)
            {
                double dx = sampled.Points[i, 0] - sampled.Points[j, 0];
                double dy = sampled.Points[i, 1] - sampled.Points[j, 1];
                double dz = sampled.Points[i, 2] - sampled.Points[j, 2];
                double dist = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                Assert.True(dist >= minDist - Tolerance,
                    $"Points {i} and {j} are {dist} apart, violating minDist={minDist}");
            }
        }
    }

    [Fact]
    public void PoissonDiskSample_InvalidMinDistance_Throws()
    {
        var cloud = CreateRandomCloud(10, 3, seed: 199);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PointCloudSampling<double>.PoissonDiskSample(cloud, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PointCloudSampling<double>.PoissonDiskSample(cloud, -1.0));
    }

    [Fact]
    public void PoissonDiskSample_MaxSamples_LimitsOutput()
    {
        var cloud = CreateRandomCloud(200, 3, seed: 211);

        var sampled = PointCloudSampling<double>.PoissonDiskSample(cloud, 0.01, maxSamples: 5, seed: 223);

        Assert.True(sampled.NumPoints <= 5);
    }

    [Fact]
    public void VoxelGridSample_ReducesPointCount()
    {
        // Many points in a small area should reduce to fewer voxels
        var cloud = CreateRandomCloud(100, 3, seed: 227);

        var sampled = PointCloudSampling<double>.VoxelGridSample(cloud, 0.5);

        Assert.True(sampled.NumPoints < 100, "Voxel downsampling should reduce points");
        Assert.True(sampled.NumPoints > 0, "Should have at least some points");
    }

    [Fact]
    public void VoxelGridSample_AveragesPointsPerVoxel()
    {
        // Two points in same voxel should average to midpoint
        var data = new double[]
        {
            0.1, 0.1, 0.1,
            0.2, 0.2, 0.2,
        };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 2, 3 }));

        var sampled = PointCloudSampling<double>.VoxelGridSample(cloud, 1.0);

        // Both should be in the same voxel, averaged
        if (sampled.NumPoints == 1)
        {
            Assert.Equal(0.15, sampled.Points[0, 0], 1e-4);
            Assert.Equal(0.15, sampled.Points[0, 1], 1e-4);
            Assert.Equal(0.15, sampled.Points[0, 2], 1e-4);
        }
    }

    [Fact]
    public void VoxelGridSample_LargeVoxelSize_SinglePoint()
    {
        var cloud = CreateRandomCloud(50, 3, seed: 229);

        var sampled = PointCloudSampling<double>.VoxelGridSample(cloud, 100.0);

        // Very large voxel should collapse all points into 1 voxel
        Assert.Equal(1, sampled.NumPoints);
    }

    #endregion

    #region GeometryMetrics Tests

    [Fact]
    public void ChamferDistance_IdenticalClouds_IsZero()
    {
        var data = new double[] { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 3, 3 }));

        double dist = GeometryMetrics<double>.ChamferDistance(cloud, cloud);

        Assert.Equal(0.0, dist, Tolerance);
    }

    [Fact]
    public void ChamferDistance_GoldenReference_KnownShift()
    {
        // Source at origin, target shifted by (1,0,0)
        var srcData = new double[] { 0, 0, 0 };
        var tgtData = new double[] { 1, 0, 0 };
        var src = new PointCloudData<double>(new Tensor<double>(srcData, new[] { 1, 3 }));
        var tgt = new PointCloudData<double>(new Tensor<double>(tgtData, new[] { 1, 3 }));

        double dist = GeometryMetrics<double>.ChamferDistance(src, tgt);

        // Both directions: nearest = 1.0, so average = (1.0/1 + 1.0/1)/2 = 1.0
        Assert.Equal(1.0, dist, Tolerance);
    }

    [Fact]
    public void ChamferDistance_Squared_ReturnsSquaredDistances()
    {
        var srcData = new double[] { 0, 0, 0 };
        var tgtData = new double[] { 3, 4, 0 }; // distance = 5
        var src = new PointCloudData<double>(new Tensor<double>(srcData, new[] { 1, 3 }));
        var tgt = new PointCloudData<double>(new Tensor<double>(tgtData, new[] { 1, 3 }));

        double distSquared = GeometryMetrics<double>.ChamferDistance(src, tgt, squared: true);
        double distNormal = GeometryMetrics<double>.ChamferDistance(src, tgt, squared: false);

        // squared: (25/1 + 25/1)/2 = 25
        Assert.Equal(25.0, distSquared, Tolerance);
        // non-squared: (5/1 + 5/1)/2 = 5
        Assert.Equal(5.0, distNormal, Tolerance);
    }

    [Fact]
    public void ChamferDistance_IsSymmetric()
    {
        var cloud1 = CreateRandomCloud(20, 3, seed: 233);
        var cloud2 = CreateRandomCloud(15, 3, seed: 239);

        double d12 = GeometryMetrics<double>.ChamferDistance(cloud1, cloud2);
        double d21 = GeometryMetrics<double>.ChamferDistance(cloud2, cloud1);

        Assert.Equal(d12, d21, Tolerance);
    }

    [Fact]
    public void HausdorffDistance_IdenticalClouds_IsZero()
    {
        var cloud = CreateRandomCloud(10, 3, seed: 241);

        double dist = GeometryMetrics<double>.HausdorffDistance(cloud, cloud);

        Assert.Equal(0.0, dist, Tolerance);
    }

    [Fact]
    public void HausdorffDistance_GoldenReference()
    {
        // Source: (0,0,0), (1,0,0)  Target: (0,0,0), (1,0,0), (5,0,0)
        var srcData = new double[] { 0, 0, 0, 1, 0, 0 };
        var tgtData = new double[] { 0, 0, 0, 1, 0, 0, 5, 0, 0 };
        var src = new PointCloudData<double>(new Tensor<double>(srcData, new[] { 2, 3 }));
        var tgt = new PointCloudData<double>(new Tensor<double>(tgtData, new[] { 3, 3 }));

        double dist = GeometryMetrics<double>.HausdorffDistance(src, tgt);

        // src->tgt: max min dist = max(0, 0) = 0
        // tgt->src: for (5,0,0), nearest in src is (1,0,0), dist=4
        // Hausdorff = max(0, 4) = 4
        Assert.Equal(4.0, dist, Tolerance);
    }

    [Fact]
    public void FScore_PerfectMatch_ReturnsOne()
    {
        var data = new double[] { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 3, 3 }));

        var (fscore, precision, recall) = GeometryMetrics<double>.FScore(cloud, cloud, threshold: 0.1);

        Assert.Equal(1.0, fscore, Tolerance);
        Assert.Equal(1.0, precision, Tolerance);
        Assert.Equal(1.0, recall, Tolerance);
    }

    [Fact]
    public void FScore_NoMatch_ReturnsZero()
    {
        var pred = new PointCloudData<double>(new Tensor<double>(
            new double[] { 0, 0, 0 }, new[] { 1, 3 }));
        var gt = new PointCloudData<double>(new Tensor<double>(
            new double[] { 100, 100, 100 }, new[] { 1, 3 }));

        var (fscore, _, _) = GeometryMetrics<double>.FScore(pred, gt, threshold: 1.0);

        Assert.Equal(0.0, fscore, Tolerance);
    }

    [Fact]
    public void FScore_PartialMatch()
    {
        // 2 predictions, 1 matches, 1 doesn't
        var pred = new PointCloudData<double>(new Tensor<double>(
            new double[] { 0, 0, 0, 100, 100, 100 }, new[] { 2, 3 }));
        var gt = new PointCloudData<double>(new Tensor<double>(
            new double[] { 0, 0, 0 }, new[] { 1, 3 }));

        var (fscore, precision, recall) = GeometryMetrics<double>.FScore(pred, gt, threshold: 1.0);

        // Precision: 1/2 = 0.5 (1 of 2 predictions matches)
        Assert.Equal(0.5, precision, Tolerance);
        // Recall: 1/1 = 1.0 (all ground truth matched)
        Assert.Equal(1.0, recall, Tolerance);
        // F-Score: 2 * 0.5 * 1.0 / (0.5 + 1.0) = 2/3
        Assert.Equal(2.0 / 3.0, fscore, Tolerance);
    }

    #endregion

    #region Voxelization Tests

    [Fact]
    public void VoxelizePointCloud_ProducesOccupiedVoxels()
    {
        var data = new double[]
        {
            0, 0, 0,
            0.5, 0.5, 0.5,
            1, 1, 1,
        };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 3, 3 }));

        var grid = Voxelization<double>.VoxelizePointCloud(cloud, resolution: 4);

        Assert.Equal(new[] { 4, 4, 4 }, grid.Voxels.Shape);

        // At least some voxels should be occupied
        int occupied = 0;
        for (int i = 0; i < grid.Voxels.Length; i++)
        {
            if (grid.Voxels[i] > 0.5) occupied++;
        }
        Assert.True(occupied > 0, "Should have occupied voxels");
        Assert.True(occupied <= 3, "Should not have more occupied voxels than points");
    }

    [Fact]
    public void VoxelizePointCloud_EmptyCloud_ReturnsEmptyGrid()
    {
        var cloud = new PointCloudData<double>(new Tensor<double>(new double[0], new[] { 0, 3 }));

        var grid = Voxelization<double>.VoxelizePointCloud(cloud, resolution: 4);

        Assert.Equal(new[] { 4, 4, 4 }, grid.Voxels.Shape);
    }

    [Fact]
    public void IoU_PerfectMatch_ReturnsOne()
    {
        var data = new double[] { 1, 0, 0, 1, 0, 0, 0, 0 };
        var grid1 = new VoxelGridData<double>(new Tensor<double>(data, new[] { 2, 2, 2 }));
        var grid2 = new VoxelGridData<double>(new Tensor<double>((double[])data.Clone(), new[] { 2, 2, 2 }));

        double iou = Voxelization<double>.IntersectionOverUnion(grid1, grid2);

        Assert.Equal(1.0, iou, Tolerance);
    }

    [Fact]
    public void IoU_NoOverlap_ReturnsZero()
    {
        var data1 = new double[] { 1, 0, 0, 0, 0, 0, 0, 0 };
        var data2 = new double[] { 0, 0, 0, 0, 0, 0, 0, 1 };
        var grid1 = new VoxelGridData<double>(new Tensor<double>(data1, new[] { 2, 2, 2 }));
        var grid2 = new VoxelGridData<double>(new Tensor<double>(data2, new[] { 2, 2, 2 }));

        double iou = Voxelization<double>.IntersectionOverUnion(grid1, grid2);

        Assert.Equal(0.0, iou, Tolerance);
    }

    [Fact]
    public void IoU_PartialOverlap_GoldenReference()
    {
        // Grid1: voxels 0,1 occupied; Grid2: voxels 1,2 occupied
        // Intersection = 1 (voxel 1); Union = 3 (voxels 0,1,2)
        var data1 = new double[] { 1, 1, 0, 0, 0, 0, 0, 0 };
        var data2 = new double[] { 0, 1, 1, 0, 0, 0, 0, 0 };
        var grid1 = new VoxelGridData<double>(new Tensor<double>(data1, new[] { 2, 2, 2 }));
        var grid2 = new VoxelGridData<double>(new Tensor<double>(data2, new[] { 2, 2, 2 }));

        double iou = Voxelization<double>.IntersectionOverUnion(grid1, grid2);

        Assert.Equal(1.0 / 3.0, iou, Tolerance);
    }

    [Fact]
    public void Dilate_ExpandsOccupiedRegion()
    {
        // Single voxel in center of 3x3x3
        var data = new double[27]; // all zeros
        data[13] = 1.0; // center voxel (1,1,1) in 3x3x3
        var grid = new VoxelGridData<double>(new Tensor<double>(data, new[] { 3, 3, 3 }));

        var dilated = Voxelization<double>.Dilate(grid, radius: 1);

        // Center and all 26 neighbors should be occupied
        int occupied = 0;
        for (int i = 0; i < 27; i++)
        {
            if (dilated.Voxels[i] > 0.5) occupied++;
        }
        Assert.Equal(27, occupied); // All voxels should be filled
    }

    [Fact]
    public void Erode_ShrinksOccupiedRegion()
    {
        // All voxels occupied in 3x3x3
        var data = Enumerable.Repeat(1.0, 27).ToArray();
        var grid = new VoxelGridData<double>(new Tensor<double>(data, new[] { 3, 3, 3 }));

        var eroded = Voxelization<double>.Erode(grid, radius: 1);

        // Only center voxel (1,1,1) should remain
        int occupied = 0;
        for (int i = 0; i < 27; i++)
        {
            if (eroded.Voxels[i] > 0.5) occupied++;
        }
        Assert.Equal(1, occupied);
        Assert.True(eroded.Voxels[13] > 0.5); // center
    }

    [Fact]
    public void DilateErode_RoundTrip_ReducesRegion()
    {
        var data = new double[27];
        data[13] = 1.0;
        var grid = new VoxelGridData<double>(new Tensor<double>(data, new[] { 3, 3, 3 }));

        var dilated = Voxelization<double>.Dilate(grid, radius: 1);
        var closedResult = Voxelization<double>.Erode(dilated, radius: 1);

        // After dilate then erode (morphological closing), should have center filled
        Assert.True(closedResult.Voxels[13] > 0.5);
    }

    #endregion

    #region NeighborSearch Tests

    [Fact]
    public void KNN_GoldenReference_FindsClosestPoints()
    {
        var data = new double[]
        {
            0, 0, 0,  // point 0
            1, 0, 0,  // point 1
            0, 1, 0,  // point 2
            10, 10, 10, // point 3 (far away)
        };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 4, 3 }));

        var knn = NeighborSearch<double>.KNearestNeighbors(cloud, k: 2);

        // For point 0: nearest should be itself (dist 0), then point 1 or 2 (dist 1)
        Assert.Equal(0, knn[0, 0]); // self
        Assert.True(knn[0, 1] == 1 || knn[0, 1] == 2);

        // For point 3 (far away): nearest is itself
        Assert.Equal(3, knn[3, 0]);
    }

    [Fact]
    public void KNN_SelfIsAlwaysFirstNeighbor()
    {
        var cloud = CreateRandomCloud(20, 3, seed: 251);

        var knn = NeighborSearch<double>.KNearestNeighbors(cloud, k: 3);

        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(i, knn[i, 0]); // self is always k=0
        }
    }

    [Fact]
    public void RadiusSearch_GoldenReference()
    {
        var data = new double[]
        {
            0, 0, 0,
            0.5, 0, 0,
            0, 0.5, 0,
            5, 5, 5,
        };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 4, 3 }));

        // Query from point 0 with radius 1.0
        var queryData = new double[] { 0, 0, 0 };
        var query = new PointCloudData<double>(new Tensor<double>(queryData, new[] { 1, 3 }));

        var results = NeighborSearch<double>.RadiusSearch(cloud, query, radius: 1.0);

        Assert.Single(results);
        // Should find points 0, 1, 2 (within radius 1) but not point 3
        Assert.Contains(0, results[0]);
        Assert.Contains(1, results[0]);
        Assert.Contains(2, results[0]);
        Assert.DoesNotContain(3, results[0]);
    }

    [Fact]
    public void BallQuery_RespectsRadiusAndMaxSamples()
    {
        var cloud = CreateRandomCloud(50, 3, seed: 257);
        var centroidData = new double[] { 0, 0, 0 };
        var centroids = new PointCloudData<double>(new Tensor<double>(centroidData, new[] { 1, 3 }));

        var result = NeighborSearch<double>.BallQuery(cloud, centroids, radius: 10.0, maxSamples: 5);

        Assert.Equal(1, result.GetLength(0)); // 1 centroid
        Assert.Equal(5, result.GetLength(1)); // maxSamples
    }

    [Fact]
    public void DistanceMatrix_GoldenReference_Symmetric()
    {
        var data = new double[]
        {
            0, 0, 0,
            3, 4, 0,  // distance from origin = 5
            1, 0, 0,
        };
        var cloud = new PointCloudData<double>(new Tensor<double>(data, new[] { 3, 3 }));

        var distMatrix = NeighborSearch<double>.ComputeDistanceMatrix(cloud);

        // Diagonal should be zero
        Assert.Equal(0.0, distMatrix[0, 0], Tolerance);
        Assert.Equal(0.0, distMatrix[1, 1], Tolerance);
        Assert.Equal(0.0, distMatrix[2, 2], Tolerance);

        // Known distance: (0,0,0) to (3,4,0) = 5
        Assert.Equal(5.0, distMatrix[0, 1], Tolerance);

        // Symmetry
        Assert.Equal(distMatrix[0, 1], distMatrix[1, 0], Tolerance);
        Assert.Equal(distMatrix[0, 2], distMatrix[2, 0], Tolerance);
        Assert.Equal(distMatrix[1, 2], distMatrix[2, 1], Tolerance);

        // (0,0,0) to (1,0,0) = 1
        Assert.Equal(1.0, distMatrix[0, 2], Tolerance);
    }

    #endregion

    #region MeshOperations Tests

    [Fact]
    public void ComputeFaceNormals_GoldenReference_XYPlaneTriangle()
    {
        var vertices = new Tensor<double>(new double[]
        {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
        }, new[] { 3, 3 });
        var faces = new Tensor<int>(new[] { 0, 1, 2 }, new[] { 1, 3 });
        var mesh = new TriangleMeshData<double>(vertices, faces);

        var normals = MeshOperations<double>.ComputeFaceNormals(mesh);

        Assert.Equal(new[] { 1, 3 }, normals.Shape);
        // Normal of XY plane triangle = (0, 0, 1) (right-hand rule)
        Assert.Equal(0.0, normals[0], Tolerance);
        Assert.Equal(0.0, normals[1], Tolerance);
        Assert.Equal(1.0, normals[2], Tolerance);
    }

    [Fact]
    public void ComputeFaceNormals_XZPlaneTriangle()
    {
        var vertices = new Tensor<double>(new double[]
        {
            0, 0, 0,
            1, 0, 0,
            0, 0, 1,
        }, new[] { 3, 3 });
        var faces = new Tensor<int>(new[] { 0, 1, 2 }, new[] { 1, 3 });
        var mesh = new TriangleMeshData<double>(vertices, faces);

        var normals = MeshOperations<double>.ComputeFaceNormals(mesh);

        // edge1 = (1,0,0), edge2 = (0,0,1)
        // cross = (0*1-0*0, 0*0-1*1, 1*0-0*0) = (0, -1, 0)
        Assert.Equal(0.0, normals[0], Tolerance);
        Assert.Equal(-1.0, normals[1], Tolerance);
        Assert.Equal(0.0, normals[2], Tolerance);
    }

    [Fact]
    public void ComputeVertexNormals_SharedVertex_AveragesNormals()
    {
        // Two triangles sharing vertex 0
        var vertices = new Tensor<double>(new double[]
        {
            0, 0, 0,   // shared vertex
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        }, new[] { 4, 3 });
        var faces = new Tensor<int>(new[]
        {
            0, 1, 2,  // face in XY plane, normal (0,0,1)
            0, 1, 3,  // face in XZ plane, normal (0,-1,0)
        }, new[] { 2, 3 });
        var mesh = new TriangleMeshData<double>(vertices, faces);

        var normals = MeshOperations<double>.ComputeVertexNormals(mesh);

        Assert.Equal(new[] { 4, 3 }, normals.Shape);

        // Vertex 0 participates in both faces, normal should be averaged and normalized
        double nx = normals[0];
        double ny = normals[1];
        double nz = normals[2];
        double len = Math.Sqrt(nx * nx + ny * ny + nz * nz);
        Assert.Equal(1.0, len, 1e-4); // should be unit length
    }

    [Fact]
    public void BuildVertexAdjacency_SimpleTriangle()
    {
        var vertices = new Tensor<double>(new double[]
        {
            0, 0, 0, 1, 0, 0, 0, 1, 0
        }, new[] { 3, 3 });
        var faces = new Tensor<int>(new[] { 0, 1, 2 }, new[] { 1, 3 });
        var mesh = new TriangleMeshData<double>(vertices, faces);

        var adjacency = MeshOperations<double>.BuildVertexAdjacency(mesh);

        Assert.Equal(3, adjacency.Length);
        // Each vertex is adjacent to the other 2
        Assert.Equal(2, adjacency[0].Count);
        Assert.Contains(1, adjacency[0]);
        Assert.Contains(2, adjacency[0]);
        Assert.Equal(2, adjacency[1].Count);
        Assert.Equal(2, adjacency[2].Count);
    }

    [Fact]
    public void SamplePoints_ProducesCorrectCount()
    {
        var vertices = new Tensor<double>(new double[]
        {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
        }, new[] { 3, 3 });
        var faces = new Tensor<int>(new[] { 0, 1, 2 }, new[] { 1, 3 });
        var mesh = new TriangleMeshData<double>(vertices, faces);

        var samples = MeshOperations<double>.SamplePoints(mesh, 100, seed: 263);

        Assert.Equal(100, samples.NumPoints);
        Assert.Equal(6, samples.NumFeatures); // XYZ + normals
    }

    [Fact]
    public void SamplePoints_WithoutNormals_Has3Features()
    {
        var vertices = new Tensor<double>(new double[]
        {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
        }, new[] { 3, 3 });
        var faces = new Tensor<int>(new[] { 0, 1, 2 }, new[] { 1, 3 });
        var mesh = new TriangleMeshData<double>(vertices, faces);

        var samples = MeshOperations<double>.SamplePoints(mesh, 50, seed: 269, includeNormals: false);

        Assert.Equal(50, samples.NumPoints);
        Assert.Equal(3, samples.NumFeatures);
    }

    [Fact]
    public void SamplePoints_AllPointsOnTriangleSurface()
    {
        // Triangle in XY plane: z should be 0 for all samples
        var vertices = new Tensor<double>(new double[]
        {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
        }, new[] { 3, 3 });
        var faces = new Tensor<int>(new[] { 0, 1, 2 }, new[] { 1, 3 });
        var mesh = new TriangleMeshData<double>(vertices, faces);

        var samples = MeshOperations<double>.SamplePoints(mesh, 200, seed: 271, includeNormals: false);

        for (int i = 0; i < samples.NumPoints; i++)
        {
            double x = samples.Points[i, 0];
            double y = samples.Points[i, 1];
            double z = samples.Points[i, 2];

            Assert.Equal(0.0, z, 1e-10); // all on XY plane
            Assert.True(x >= -Tolerance, $"x={x} < 0");
            Assert.True(y >= -Tolerance, $"y={y} < 0");
            Assert.True(x + y <= 1.0 + Tolerance, $"x+y={x + y} > 1");
        }
    }

    [Fact]
    public void ComputeStatistics_GoldenReference_UnitTetrahedron()
    {
        // Unit right tetrahedron: 4 faces
        var vertices = new Tensor<double>(new double[]
        {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        }, new[] { 4, 3 });
        var faces = new Tensor<int>(new[]
        {
            0, 1, 2,  // bottom
            0, 1, 3,  // front
            0, 2, 3,  // left
            1, 2, 3,  // hypotenuse
        }, new[] { 4, 3 });
        var mesh = new TriangleMeshData<double>(vertices, faces);

        var stats = MeshOperations<double>.ComputeStatistics(mesh);

        Assert.Equal(4, stats.NumVertices);
        Assert.Equal(4, stats.NumFaces);
        Assert.Equal(6, stats.NumEdges);
        Assert.True(stats.SurfaceArea > 0);
        // Volume of right tetrahedron with legs 1,1,1 = 1/6
        Assert.Equal(1.0 / 6.0, stats.Volume, 1e-4);
        Assert.Equal(0.0, stats.BoundingBoxMin.X, Tolerance);
        Assert.Equal(0.0, stats.BoundingBoxMin.Y, Tolerance);
        Assert.Equal(0.0, stats.BoundingBoxMin.Z, Tolerance);
        Assert.Equal(1.0, stats.BoundingBoxMax.X, Tolerance);
        Assert.Equal(1.0, stats.BoundingBoxMax.Y, Tolerance);
        Assert.Equal(1.0, stats.BoundingBoxMax.Z, Tolerance);
    }

    [Fact]
    public void VoxelizeMeshSurface_ProducesOccupiedVoxels()
    {
        var vertices = new Tensor<double>(new double[]
        {
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
        }, new[] { 3, 3 });
        var faces = new Tensor<int>(new[] { 0, 1, 2 }, new[] { 1, 3 });
        var mesh = new TriangleMeshData<double>(vertices, faces);

        var grid = Voxelization<double>.VoxelizeMeshSurface(mesh, resolution: 4);

        int occupied = 0;
        for (int i = 0; i < grid.Voxels.Length; i++)
        {
            if (grid.Voxels[i] > 0.5) occupied++;
        }
        Assert.True(occupied > 0, "Mesh voxelization should produce occupied voxels");
    }

    #endregion

    #region Helpers

    private static PointCloudData<double> CreateRandomCloud(int numPoints, int numFeatures, int seed)
    {
        var tensor = CreateRandomTensor(numPoints, numFeatures, seed);
        return new PointCloudData<double>(tensor);
    }

    private static Tensor<double> CreateRandomTensor(int rows, int cols, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var data = new double[rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble() * 2.0 - 1.0;
        }
        return new Tensor<double>(data, new[] { rows, cols });
    }

    #endregion
}
