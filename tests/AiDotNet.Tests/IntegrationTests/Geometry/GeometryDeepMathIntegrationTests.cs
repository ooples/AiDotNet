using AiDotNet.Geometry.Data;
using AiDotNet.Geometry.Preprocessing;
using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Geometry;

public class GeometryDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    // ── Helper: create a PointCloudData<double> from flat array ────────────
    private static PointCloudData<double> MakeCloud(double[,] points, Vector<double>? labels = null)
    {
        int n = points.GetLength(0);
        int f = points.GetLength(1);
        var flat = new double[n * f];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < f; j++)
                flat[i * f + j] = points[i, j];
        var tensor = new Tensor<double>(flat, [n, f]);
        return new PointCloudData<double>(tensor, labels);
    }

    private static TriangleMeshData<double> MakeTriangleMesh(double[,] verts, int[,] faces)
    {
        int nv = verts.GetLength(0);
        var flatV = new double[nv * 3];
        for (int i = 0; i < nv; i++)
            for (int j = 0; j < 3; j++)
                flatV[i * 3 + j] = verts[i, j];

        int nf = faces.GetLength(0);
        var flatF = new int[nf * 3];
        for (int i = 0; i < nf; i++)
            for (int j = 0; j < 3; j++)
                flatF[i * 3 + j] = faces[i, j];

        var vertTensor = new Tensor<double>(flatV, [nv, 3]);
        var faceTensor = new Tensor<int>(flatF, [nf, 3]);
        return new TriangleMeshData<double>(vertTensor, faceTensor);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  CHAMFER DISTANCE
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void ChamferDistance_IdenticalClouds_ReturnsZero()
    {
        var cloud = MakeCloud(new double[,] { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } });
        double cd = GeometryMetrics<double>.ChamferDistance(cloud, cloud);
        Assert.Equal(0.0, cd, Tolerance);
    }

    [Fact]
    public void ChamferDistance_SinglePointPair_HandComputed()
    {
        // Source: (0,0,0), Target: (3,4,0)
        // Distance = sqrt(9+16) = 5
        // CD = (5/1 + 5/1) / 2 = 5
        var source = MakeCloud(new double[,] { { 0, 0, 0 } });
        var target = MakeCloud(new double[,] { { 3, 4, 0 } });
        double cd = GeometryMetrics<double>.ChamferDistance(source, target);
        Assert.Equal(5.0, cd, Tolerance);
    }

    [Fact]
    public void ChamferDistance_Squared_HandComputed()
    {
        // Source: (0,0,0), Target: (3,4,0)
        // Squared distance = 25
        // CD_sq = (25/1 + 25/1) / 2 = 25
        var source = MakeCloud(new double[,] { { 0, 0, 0 } });
        var target = MakeCloud(new double[,] { { 3, 4, 0 } });
        double cd = GeometryMetrics<double>.ChamferDistance(source, target, squared: true);
        Assert.Equal(25.0, cd, Tolerance);
    }

    [Fact]
    public void ChamferDistance_AsymmetricClouds_HandComputed()
    {
        // Source: (0,0,0), (2,0,0)
        // Target: (1,0,0)
        // source->target: min d(s0,t0)=1, min d(s1,t0)=1 => sum=2, avg=1
        // target->source: min d(t0,s0)=1, min d(t0,s1)=1 => min=1, avg=1
        // CD = (1 + 1) / 2 = 1.0
        var source = MakeCloud(new double[,] { { 0, 0, 0 }, { 2, 0, 0 } });
        var target = MakeCloud(new double[,] { { 1, 0, 0 } });
        double cd = GeometryMetrics<double>.ChamferDistance(source, target);
        Assert.Equal(1.0, cd, Tolerance);
    }

    [Fact]
    public void ChamferDistance_IsSymmetric()
    {
        var a = MakeCloud(new double[,] { { 0, 0, 0 }, { 1, 1, 1 } });
        var b = MakeCloud(new double[,] { { 2, 0, 0 }, { 0, 2, 0 } });
        double cdAB = GeometryMetrics<double>.ChamferDistance(a, b);
        double cdBA = GeometryMetrics<double>.ChamferDistance(b, a);
        Assert.Equal(cdAB, cdBA, Tolerance);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  HAUSDORFF DISTANCE
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void HausdorffDistance_IdenticalClouds_ReturnsZero()
    {
        var cloud = MakeCloud(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        double hd = GeometryMetrics<double>.HausdorffDistance(cloud, cloud);
        Assert.Equal(0.0, hd, Tolerance);
    }

    [Fact]
    public void HausdorffDistance_SinglePointPair_EqualsEuclidean()
    {
        var a = MakeCloud(new double[,] { { 0, 0, 0 } });
        var b = MakeCloud(new double[,] { { 1, 2, 2 } });
        // dist = sqrt(1+4+4) = 3
        double hd = GeometryMetrics<double>.HausdorffDistance(a, b);
        Assert.Equal(3.0, hd, Tolerance);
    }

    [Fact]
    public void HausdorffDistance_HandComputed_MaxOfMinDistances()
    {
        // A: (0,0,0), (10,0,0)
        // B: (1,0,0)
        // A->B: min(0->1=1, 10->1=9) => max = 9
        // B->A: min(1->0=1, 1->10=9) => max = 1
        // HD = max(9, 1) = 9
        var a = MakeCloud(new double[,] { { 0, 0, 0 }, { 10, 0, 0 } });
        var b = MakeCloud(new double[,] { { 1, 0, 0 } });
        double hd = GeometryMetrics<double>.HausdorffDistance(a, b);
        Assert.Equal(9.0, hd, Tolerance);
    }

    [Fact]
    public void HausdorffDistance_IsSymmetric()
    {
        var a = MakeCloud(new double[,] { { 0, 0, 0 }, { 10, 0, 0 } });
        var b = MakeCloud(new double[,] { { 1, 0, 0 } });
        double hdAB = GeometryMetrics<double>.HausdorffDistance(a, b);
        double hdBA = GeometryMetrics<double>.HausdorffDistance(b, a);
        Assert.Equal(hdAB, hdBA, Tolerance);
    }

    [Fact]
    public void HausdorffDistance_GreaterOrEqualChamferDistance()
    {
        var a = MakeCloud(new double[,] { { 0, 0, 0 }, { 5, 0, 0 }, { 10, 0, 0 } });
        var b = MakeCloud(new double[,] { { 1, 0, 0 }, { 6, 0, 0 } });
        double hd = GeometryMetrics<double>.HausdorffDistance(a, b);
        double cd = GeometryMetrics<double>.ChamferDistance(a, b);
        Assert.True(hd >= cd, $"Hausdorff {hd} should be >= Chamfer {cd}");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  F-SCORE
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void FScore_PerfectMatch_ReturnsOne()
    {
        var cloud = MakeCloud(new double[,] { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } });
        var (fScore, precision, recall) = GeometryMetrics<double>.FScore(cloud, cloud, threshold: 0.1);
        Assert.Equal(1.0, precision, Tolerance);
        Assert.Equal(1.0, recall, Tolerance);
        Assert.Equal(1.0, fScore, Tolerance);
    }

    [Fact]
    public void FScore_NoOverlap_ReturnsZero()
    {
        var pred = MakeCloud(new double[,] { { 0, 0, 0 } });
        var gt = MakeCloud(new double[,] { { 100, 100, 100 } });
        var (fScore, precision, recall) = GeometryMetrics<double>.FScore(pred, gt, threshold: 0.1);
        Assert.Equal(0.0, precision, Tolerance);
        Assert.Equal(0.0, recall, Tolerance);
        Assert.Equal(0.0, fScore, Tolerance);
    }

    [Fact]
    public void FScore_PartialMatch_HandComputed()
    {
        // Pred: (0,0,0), (10,0,0)   GT: (0,0,0), (5,0,0)
        // threshold = 0.5
        // Precision: pred(0,0,0)->gt? yes (d=0), pred(10,0,0)->gt? no => 1/2 = 0.5
        // Recall:    gt(0,0,0)->pred? yes (d=0), gt(5,0,0)->pred? no  => 1/2 = 0.5
        // F = 2*0.5*0.5 / (0.5+0.5) = 0.5
        var pred = MakeCloud(new double[,] { { 0, 0, 0 }, { 10, 0, 0 } });
        var gt = MakeCloud(new double[,] { { 0, 0, 0 }, { 5, 0, 0 } });
        var (fScore, precision, recall) = GeometryMetrics<double>.FScore(pred, gt, threshold: 0.5);
        Assert.Equal(0.5, precision, Tolerance);
        Assert.Equal(0.5, recall, Tolerance);
        Assert.Equal(0.5, fScore, Tolerance);
    }

    [Fact]
    public void FScore_HarmonicMean_Formula()
    {
        // With larger threshold, more matches
        var pred = MakeCloud(new double[,] { { 0, 0, 0 }, { 1, 0, 0 }, { 5, 0, 0 } });
        var gt = MakeCloud(new double[,] { { 0.1, 0, 0 }, { 0.9, 0, 0 } });
        var (fScore, precision, recall) = GeometryMetrics<double>.FScore(pred, gt, threshold: 0.2);
        // Verify F-score is the harmonic mean: F = 2*P*R / (P+R)
        if (precision + recall > 0)
        {
            double expected = 2 * precision * recall / (precision + recall);
            Assert.Equal(expected, fScore, Tolerance);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  POINT CLOUD NORMALIZATION - Center
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void Center_CentroidIsZero()
    {
        var cloud = MakeCloud(new double[,] { { 2, 4, 6 }, { 4, 6, 8 }, { 6, 2, 4 } });
        // Centroid: (4, 4, 6)
        var centered = PointCloudNormalization<double>.Center(cloud);

        double sumX = 0, sumY = 0, sumZ = 0;
        for (int i = 0; i < centered.NumPoints; i++)
        {
            sumX += centered.Points[i, 0];
            sumY += centered.Points[i, 1];
            sumZ += centered.Points[i, 2];
        }
        Assert.Equal(0.0, sumX / centered.NumPoints, Tolerance);
        Assert.Equal(0.0, sumY / centered.NumPoints, Tolerance);
        Assert.Equal(0.0, sumZ / centered.NumPoints, Tolerance);
    }

    [Fact]
    public void Center_PreservesRelativeDistances()
    {
        var cloud = MakeCloud(new double[,] { { 1, 0, 0 }, { 4, 0, 0 } });
        var centered = PointCloudNormalization<double>.Center(cloud);
        // Original distance: 3
        // Centered: (-1.5, 0, 0) and (1.5, 0, 0) => distance = 3
        double dx = centered.Points[1, 0] - centered.Points[0, 0];
        Assert.Equal(3.0, dx, Tolerance);
    }

    [Fact]
    public void Center_HandComputed()
    {
        // Points: (1,2,3), (3,4,5)
        // Centroid: (2, 3, 4)
        // Centered: (-1,-1,-1), (1,1,1)
        var cloud = MakeCloud(new double[,] { { 1, 2, 3 }, { 3, 4, 5 } });
        var centered = PointCloudNormalization<double>.Center(cloud);

        Assert.Equal(-1.0, centered.Points[0, 0], Tolerance);
        Assert.Equal(-1.0, centered.Points[0, 1], Tolerance);
        Assert.Equal(-1.0, centered.Points[0, 2], Tolerance);
        Assert.Equal(1.0, centered.Points[1, 0], Tolerance);
        Assert.Equal(1.0, centered.Points[1, 1], Tolerance);
        Assert.Equal(1.0, centered.Points[1, 2], Tolerance);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  POINT CLOUD NORMALIZATION - ScaleToUnitSphere
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void ScaleToUnitSphere_MaxDistanceIsOne()
    {
        var cloud = MakeCloud(new double[,] { { 10, 0, 0 }, { 0, 10, 0 }, { 0, 0, 10 }, { 0, 0, 0 } });
        var normalized = PointCloudNormalization<double>.ScaleToUnitSphere(cloud);

        double maxDist = 0;
        for (int i = 0; i < normalized.NumPoints; i++)
        {
            double x = normalized.Points[i, 0];
            double y = normalized.Points[i, 1];
            double z = normalized.Points[i, 2];
            double dist = Math.Sqrt(x * x + y * y + z * z);
            if (dist > maxDist) maxDist = dist;
        }
        Assert.Equal(1.0, maxDist, 1e-8);
    }

    [Fact]
    public void ScaleToUnitSphere_AllPointsInsideSphere()
    {
        var cloud = MakeCloud(new double[,]
        {
            { 5, 5, 5 }, { -5, 5, 5 }, { 5, -5, 5 }, { 5, 5, -5 },
            { -5, -5, 5 }, { -5, 5, -5 }, { 5, -5, -5 }, { -5, -5, -5 }
        });
        var normalized = PointCloudNormalization<double>.ScaleToUnitSphere(cloud);

        for (int i = 0; i < normalized.NumPoints; i++)
        {
            double x = normalized.Points[i, 0];
            double y = normalized.Points[i, 1];
            double z = normalized.Points[i, 2];
            double dist = Math.Sqrt(x * x + y * y + z * z);
            Assert.True(dist <= 1.0 + 1e-10, $"Point {i} at distance {dist} > 1.0");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  POINT CLOUD NORMALIZATION - ScaleToUnitCube
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void ScaleToUnitCube_AllPointsInRange()
    {
        var cloud = MakeCloud(new double[,]
        {
            { 10, 20, 30 }, { 50, 60, 70 }, { 30, 40, 50 }
        });
        var normalized = PointCloudNormalization<double>.ScaleToUnitCube(cloud);

        for (int i = 0; i < normalized.NumPoints; i++)
        {
            double x = normalized.Points[i, 0];
            double y = normalized.Points[i, 1];
            double z = normalized.Points[i, 2];
            Assert.True(x >= -0.5 - 1e-10 && x <= 0.5 + 1e-10, $"x={x} out of [-0.5, 0.5]");
            Assert.True(y >= -0.5 - 1e-10 && y <= 0.5 + 1e-10, $"y={y} out of [-0.5, 0.5]");
            Assert.True(z >= -0.5 - 1e-10 && z <= 0.5 + 1e-10, $"z={z} out of [-0.5, 0.5]");
        }
    }

    [Fact]
    public void ScaleToUnitCube_MaxRangeSpansFullUnit()
    {
        // Points along x-axis from 0 to 10
        var cloud = MakeCloud(new double[,] { { 0, 0, 0 }, { 10, 0, 0 } });
        var normalized = PointCloudNormalization<double>.ScaleToUnitCube(cloud);

        // The max range (x) should span from -0.5 to 0.5
        double minX = double.MaxValue, maxX = double.MinValue;
        for (int i = 0; i < normalized.NumPoints; i++)
        {
            double x = normalized.Points[i, 0];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
        }
        Assert.Equal(1.0, maxX - minX, 1e-8);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  POINT CLOUD NORMALIZATION - NormalizeColors
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void NormalizeColors_ScalesCorrectly()
    {
        // 6-feature cloud: xyz + rgb
        var data = new double[,] { { 1, 2, 3, 255, 128, 0 }, { 4, 5, 6, 0, 255, 64 } };
        var cloud = MakeCloud(data);
        var normalized = PointCloudNormalization<double>.NormalizeColors(cloud, colorOffset: 3);

        // XYZ should be unchanged
        Assert.Equal(1.0, normalized.Points[0, 0], Tolerance);
        Assert.Equal(4.0, normalized.Points[1, 0], Tolerance);

        // Colors should be /255
        Assert.Equal(1.0, normalized.Points[0, 3], 1e-8);        // 255/255
        Assert.Equal(128.0 / 255.0, normalized.Points[0, 4], 1e-8); // 128/255
        Assert.Equal(0.0, normalized.Points[0, 5], 1e-8);        // 0/255
        Assert.Equal(0.0, normalized.Points[1, 3], 1e-8);        // 0/255
        Assert.Equal(1.0, normalized.Points[1, 4], 1e-8);        // 255/255
        Assert.Equal(64.0 / 255.0, normalized.Points[1, 5], 1e-8); // 64/255
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  VOXELIZATION
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void VoxelizePointCloud_SinglePoint_SingleVoxelOccupied()
    {
        var cloud = MakeCloud(new double[,] { { 0, 0, 0 } });
        var grid = Voxelization<double>.VoxelizePointCloud(cloud, resolution: 4, padding: 0.1);

        Assert.Equal(4, grid.Depth);
        Assert.Equal(4, grid.Height);
        Assert.Equal(4, grid.Width);

        // Count occupied voxels
        int occupied = 0;
        for (int z = 0; z < 4; z++)
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    if (grid.Voxels[z, y, x] > 0.5) occupied++;

        Assert.Equal(1, occupied);
    }

    [Fact]
    public void VoxelizePointCloud_CornerPoints_OccupyDistinctVoxels()
    {
        // Two widely separated points should occupy different voxels
        var cloud = MakeCloud(new double[,] { { 0, 0, 0 }, { 10, 10, 10 } });
        var grid = Voxelization<double>.VoxelizePointCloud(cloud, resolution: 4, padding: 0.1);

        int occupied = 0;
        for (int z = 0; z < 4; z++)
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    if (grid.Voxels[z, y, x] > 0.5) occupied++;

        Assert.Equal(2, occupied);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  VOXEL GRID IoU
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void IoU_IdenticalGrids_ReturnsOne()
    {
        var voxels = new double[2 * 2 * 2];
        voxels[0] = 1.0; // (0,0,0)
        voxels[3] = 1.0; // (1,1,0)
        var tensor = new Tensor<double>(voxels, [2, 2, 2]);
        var origin = new Vector<double>([0, 0, 0]);
        var voxSize = new Vector<double>([1, 1, 1]);
        var grid = new VoxelGridData<double>(tensor, origin, voxSize);

        double iou = Voxelization<double>.IntersectionOverUnion(grid, grid);
        Assert.Equal(1.0, iou, Tolerance);
    }

    [Fact]
    public void IoU_DisjointGrids_ReturnsZero()
    {
        var voxels1 = new double[2 * 2 * 2];
        voxels1[0] = 1.0; // only (0,0,0)
        var tensor1 = new Tensor<double>(voxels1, [2, 2, 2]);

        var voxels2 = new double[2 * 2 * 2];
        voxels2[7] = 1.0; // only (1,1,1)
        var tensor2 = new Tensor<double>(voxels2, [2, 2, 2]);

        var origin = new Vector<double>([0, 0, 0]);
        var voxSize = new Vector<double>([1, 1, 1]);
        var grid1 = new VoxelGridData<double>(tensor1, origin, voxSize);
        var grid2 = new VoxelGridData<double>(tensor2, origin, voxSize);

        double iou = Voxelization<double>.IntersectionOverUnion(grid1, grid2);
        Assert.Equal(0.0, iou, Tolerance);
    }

    [Fact]
    public void IoU_HandComputed_HalfOverlap()
    {
        // Grid1: {(0,0,0), (1,0,0)} occupied
        // Grid2: {(1,0,0), (0,1,0)} occupied
        // Intersection: {(1,0,0)} => 1
        // Union: {(0,0,0), (1,0,0), (0,1,0)} => 3
        // IoU = 1/3
        var voxels1 = new double[2 * 2 * 2];
        voxels1[0] = 1.0; // z=0,y=0,x=0
        voxels1[1] = 1.0; // z=0,y=0,x=1
        var tensor1 = new Tensor<double>(voxels1, [2, 2, 2]);

        var voxels2 = new double[2 * 2 * 2];
        voxels2[1] = 1.0; // z=0,y=0,x=1
        voxels2[2] = 1.0; // z=0,y=1,x=0
        var tensor2 = new Tensor<double>(voxels2, [2, 2, 2]);

        var origin = new Vector<double>([0, 0, 0]);
        var voxSize = new Vector<double>([1, 1, 1]);
        var grid1 = new VoxelGridData<double>(tensor1, origin, voxSize);
        var grid2 = new VoxelGridData<double>(tensor2, origin, voxSize);

        double iou = Voxelization<double>.IntersectionOverUnion(grid1, grid2);
        Assert.Equal(1.0 / 3.0, iou, 1e-10);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  MORPHOLOGICAL OPERATIONS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void Dilate_ExpandsOccupiedRegion()
    {
        // 3x3x3 grid with center (1,1,1) occupied
        var voxels = new double[27]; // 3*3*3
        voxels[1 * 9 + 1 * 3 + 1] = 1.0; // center
        var tensor = new Tensor<double>(voxels, [3, 3, 3]);
        var origin = new Vector<double>([0, 0, 0]);
        var voxSize = new Vector<double>([1, 1, 1]);
        var grid = new VoxelGridData<double>(tensor, origin, voxSize);

        var dilated = Voxelization<double>.Dilate(grid, radius: 1);

        // With radius 1, dilation should fill the entire 3x3x3 grid since center reaches all
        int occupied = 0;
        for (int z = 0; z < 3; z++)
            for (int y = 0; y < 3; y++)
                for (int x = 0; x < 3; x++)
                    if (dilated.Voxels[z, y, x] > 0.5) occupied++;

        Assert.Equal(27, occupied); // All voxels within radius 1 of center
    }

    [Fact]
    public void Erode_ShrinksOccupiedRegion()
    {
        // 3x3x3 fully occupied grid
        var voxels = new double[27];
        for (int i = 0; i < 27; i++) voxels[i] = 1.0;
        var tensor = new Tensor<double>(voxels, [3, 3, 3]);
        var origin = new Vector<double>([0, 0, 0]);
        var voxSize = new Vector<double>([1, 1, 1]);
        var grid = new VoxelGridData<double>(tensor, origin, voxSize);

        var eroded = Voxelization<double>.Erode(grid, radius: 1);

        // After erosion with radius 1, only center (1,1,1) should survive
        // because it's the only voxel that has all neighbors within bounds
        int occupied = 0;
        for (int z = 0; z < 3; z++)
            for (int y = 0; y < 3; y++)
                for (int x = 0; x < 3; x++)
                    if (eroded.Voxels[z, y, x] > 0.5) occupied++;

        Assert.Equal(1, occupied);
        Assert.True(eroded.Voxels[1, 1, 1] > 0.5);
    }

    [Fact]
    public void DilateErode_OnCenter_RecoversCenter()
    {
        // Start with only center occupied in 5x5x5
        var voxels = new double[125]; // 5*5*5
        voxels[2 * 25 + 2 * 5 + 2] = 1.0; // center (2,2,2)
        var tensor = new Tensor<double>(voxels, [5, 5, 5]);
        var origin = new Vector<double>([0, 0, 0]);
        var voxSize = new Vector<double>([1, 1, 1]);
        var grid = new VoxelGridData<double>(tensor, origin, voxSize);

        var dilated = Voxelization<double>.Dilate(grid, radius: 1);
        var eroded = Voxelization<double>.Erode(dilated, radius: 1);

        // The center should still be occupied after dilate-then-erode
        Assert.True(eroded.Voxels[2, 2, 2] > 0.5);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  VOXEL GRID DATA
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void VoxelGrid_ComputeBounds_HandComputed()
    {
        var voxels = new double[8]; // 2x2x2
        var tensor = new Tensor<double>(voxels, [2, 2, 2]);
        var origin = new Vector<double>([1.0, 2.0, 3.0]);
        var voxSize = new Vector<double>([0.5, 0.5, 0.5]);
        var grid = new VoxelGridData<double>(tensor, origin, voxSize);

        var (min, max) = grid.ComputeBounds();
        // min = origin = (1, 2, 3)
        // max = origin + voxelSize * resolution = (1+0.5*2, 2+0.5*2, 3+0.5*2) = (2, 3, 4)
        Assert.Equal(1.0, min[0], Tolerance);
        Assert.Equal(2.0, min[1], Tolerance);
        Assert.Equal(3.0, min[2], Tolerance);
        Assert.Equal(2.0, max[0], Tolerance);
        Assert.Equal(3.0, max[1], Tolerance);
        Assert.Equal(4.0, max[2], Tolerance);
    }

    [Fact]
    public void VoxelGrid_GetVoxelCenter_HandComputed()
    {
        var voxels = new double[8]; // 2x2x2
        var tensor = new Tensor<double>(voxels, [2, 2, 2]);
        var origin = new Vector<double>([0, 0, 0]);
        var voxSize = new Vector<double>([2, 2, 2]);
        var grid = new VoxelGridData<double>(tensor, origin, voxSize);

        // Voxel (0,0,0): center = origin + (0+0.5)*voxelSize = (1, 1, 1)
        var center00 = grid.GetVoxelCenter(0, 0, 0);
        Assert.Equal(1.0, center00[0], Tolerance);
        Assert.Equal(1.0, center00[1], Tolerance);
        Assert.Equal(1.0, center00[2], Tolerance);

        // Voxel (1,1,1): center = origin + (1+0.5)*2 = (3, 3, 3)
        var center11 = grid.GetVoxelCenter(1, 1, 1);
        Assert.Equal(3.0, center11[0], Tolerance);
        Assert.Equal(3.0, center11[1], Tolerance);
        Assert.Equal(3.0, center11[2], Tolerance);
    }

    [Fact]
    public void VoxelGrid_ToPointCloud_CorrectCount()
    {
        var voxels = new double[8];
        voxels[0] = 1.0; // (0,0,0)
        voxels[7] = 1.0; // (1,1,1)
        var tensor = new Tensor<double>(voxels, [2, 2, 2]);
        var origin = new Vector<double>([0, 0, 0]);
        var voxSize = new Vector<double>([1, 1, 1]);
        var grid = new VoxelGridData<double>(tensor, origin, voxSize);

        var cloud = grid.ToPointCloud(0.5);
        Assert.Equal(2, cloud.NumPoints);
        Assert.Equal(3, cloud.NumFeatures);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  POINT CLOUD SAMPLING - Uniform
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void UniformSample_ReturnsRequestedCount()
    {
        var points = new double[10, 3];
        for (int i = 0; i < 10; i++)
        {
            points[i, 0] = i;
            points[i, 1] = 0;
            points[i, 2] = 0;
        }
        var cloud = MakeCloud(points);
        var sampled = PointCloudSampling<double>.UniformSample(cloud, 5, seed: 42);
        Assert.Equal(5, sampled.NumPoints);
    }

    [Fact]
    public void UniformSample_RequestMoreThanAvailable_ReturnsAll()
    {
        var cloud = MakeCloud(new double[,] { { 0, 0, 0 }, { 1, 1, 1 } });
        var sampled = PointCloudSampling<double>.UniformSample(cloud, 10, seed: 42);
        Assert.Equal(2, sampled.NumPoints);
    }

    [Fact]
    public void UniformSample_AllPointsFromOriginal()
    {
        var points = new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } };
        var cloud = MakeCloud(points);
        var sampled = PointCloudSampling<double>.UniformSample(cloud, 2, seed: 42);

        // Every sampled point must exist in original
        for (int i = 0; i < sampled.NumPoints; i++)
        {
            bool found = false;
            for (int j = 0; j < cloud.NumPoints; j++)
            {
                if (Math.Abs(sampled.Points[i, 0] - cloud.Points[j, 0]) < Tolerance &&
                    Math.Abs(sampled.Points[i, 1] - cloud.Points[j, 1]) < Tolerance &&
                    Math.Abs(sampled.Points[i, 2] - cloud.Points[j, 2]) < Tolerance)
                {
                    found = true;
                    break;
                }
            }
            Assert.True(found, $"Sampled point {i} not found in original cloud");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  POINT CLOUD SAMPLING - Farthest Point Sampling
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void FPS_ReturnsRequestedCount()
    {
        var points = new double[10, 3];
        for (int i = 0; i < 10; i++) { points[i, 0] = i; points[i, 1] = 0; points[i, 2] = 0; }
        var cloud = MakeCloud(points);
        var sampled = PointCloudSampling<double>.FarthestPointSample(cloud, 4, seed: 42);
        Assert.Equal(4, sampled.NumPoints);
    }

    [Fact]
    public void FPS_MaximizesMinimumDistance()
    {
        // Line of points 0,1,...,9 on x-axis
        // FPS with 3 samples should pick well-spaced points
        var points = new double[10, 3];
        for (int i = 0; i < 10; i++) { points[i, 0] = i; points[i, 1] = 0; points[i, 2] = 0; }
        var cloud = MakeCloud(points);
        var sampled = PointCloudSampling<double>.FarthestPointSample(cloud, 3, seed: 42);

        // The minimum pairwise distance among sampled points should be >= 3
        // (optimal: pick 0, 4/5, 9 => min dist ~4)
        double minDist = double.MaxValue;
        for (int i = 0; i < sampled.NumPoints; i++)
        {
            for (int j = i + 1; j < sampled.NumPoints; j++)
            {
                double dx = sampled.Points[i, 0] - sampled.Points[j, 0];
                double dy = sampled.Points[i, 1] - sampled.Points[j, 1];
                double dz = sampled.Points[i, 2] - sampled.Points[j, 2];
                double d = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                if (d < minDist) minDist = d;
            }
        }
        Assert.True(minDist >= 3.0, $"FPS min dist = {minDist}, expected >= 3");
    }

    [Fact]
    public void FPS_NoDuplicates()
    {
        var points = new double[20, 3];
        for (int i = 0; i < 20; i++) { points[i, 0] = i; points[i, 1] = i * 0.5; points[i, 2] = 0; }
        var cloud = MakeCloud(points);
        var sampled = PointCloudSampling<double>.FarthestPointSample(cloud, 5, seed: 42);

        // No two sampled points should be identical
        for (int i = 0; i < sampled.NumPoints; i++)
        {
            for (int j = i + 1; j < sampled.NumPoints; j++)
            {
                double dist = Math.Abs(sampled.Points[i, 0] - sampled.Points[j, 0]) +
                              Math.Abs(sampled.Points[i, 1] - sampled.Points[j, 1]) +
                              Math.Abs(sampled.Points[i, 2] - sampled.Points[j, 2]);
                Assert.True(dist > Tolerance, $"Duplicate points at indices {i} and {j}");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  POINT CLOUD SAMPLING - Poisson Disk
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void PoissonDiskSample_MinDistanceRespected()
    {
        var points = new double[100, 3];
        var rng = new Random(42);
        for (int i = 0; i < 100; i++)
        {
            points[i, 0] = rng.NextDouble() * 10;
            points[i, 1] = rng.NextDouble() * 10;
            points[i, 2] = rng.NextDouble() * 10;
        }
        var cloud = MakeCloud(points);
        double minDist = 2.0;
        var sampled = PointCloudSampling<double>.PoissonDiskSample(cloud, minDist, seed: 42);

        // Check all pairs have distance >= minDist
        for (int i = 0; i < sampled.NumPoints; i++)
        {
            for (int j = i + 1; j < sampled.NumPoints; j++)
            {
                double dx = sampled.Points[i, 0] - sampled.Points[j, 0];
                double dy = sampled.Points[i, 1] - sampled.Points[j, 1];
                double dz = sampled.Points[i, 2] - sampled.Points[j, 2];
                double d = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                Assert.True(d >= minDist - 1e-10, $"Points {i},{j} at distance {d} < {minDist}");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  POINT CLOUD SAMPLING - Voxel Grid
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void VoxelGridSample_ReducesPointCount()
    {
        // Dense cluster of points should merge into fewer
        var points = new double[100, 3];
        var rng = new Random(42);
        for (int i = 0; i < 100; i++)
        {
            points[i, 0] = rng.NextDouble() * 10;
            points[i, 1] = rng.NextDouble() * 10;
            points[i, 2] = rng.NextDouble() * 10;
        }
        var cloud = MakeCloud(points);
        var sampled = PointCloudSampling<double>.VoxelGridSample(cloud, voxelSize: 5.0);

        Assert.True(sampled.NumPoints < 100, "Voxel grid should reduce point count");
        Assert.True(sampled.NumPoints >= 1, "Should have at least one point");
    }

    [Fact]
    public void VoxelGridSample_AveragesPoints()
    {
        // Two points in same voxel => average
        var points = new double[,] { { 0, 0, 0 }, { 0.1, 0.1, 0.1 } };
        var cloud = MakeCloud(points);
        var sampled = PointCloudSampling<double>.VoxelGridSample(cloud, voxelSize: 1.0);

        Assert.Equal(1, sampled.NumPoints);
        Assert.Equal(0.05, sampled.Points[0, 0], 1e-8);
        Assert.Equal(0.05, sampled.Points[0, 1], 1e-8);
        Assert.Equal(0.05, sampled.Points[0, 2], 1e-8);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  MESH OPERATIONS - Face Normals
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void ComputeFaceNormals_XYPlaneTriangle_NormalAlongZ()
    {
        // Triangle in XY plane: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0)
        // e1 = (1,0,0), e2 = (0,1,0)
        // normal = e1 x e2 = (0,0,1)
        var mesh = MakeTriangleMesh(
            new double[,] { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } },
            new int[,] { { 0, 1, 2 } }
        );
        var normals = MeshOperations<double>.ComputeFaceNormals(mesh);

        Assert.Equal(0.0, normals[0, 0], Tolerance); // nx
        Assert.Equal(0.0, normals[0, 1], Tolerance); // ny
        Assert.Equal(1.0, normals[0, 2], Tolerance); // nz
    }

    [Fact]
    public void ComputeFaceNormals_XZPlaneTriangle_NormalAlongY()
    {
        // Triangle in XZ plane: v0=(0,0,0), v1=(1,0,0), v2=(0,0,1)
        // e1 = (1,0,0), e2 = (0,0,1)
        // normal = e1 x e2 = (0*1-0*0, 0*0-1*1, 1*0-0*0) = (0,-1,0)
        // Normalized: (0, -1, 0)
        var mesh = MakeTriangleMesh(
            new double[,] { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 1 } },
            new int[,] { { 0, 1, 2 } }
        );
        var normals = MeshOperations<double>.ComputeFaceNormals(mesh);

        Assert.Equal(0.0, normals[0, 0], Tolerance);
        Assert.Equal(-1.0, normals[0, 1], Tolerance);
        Assert.Equal(0.0, normals[0, 2], Tolerance);
    }

    [Fact]
    public void ComputeFaceNormals_UnitLength()
    {
        // Arbitrary triangle
        var mesh = MakeTriangleMesh(
            new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 0 } },
            new int[,] { { 0, 1, 2 } }
        );
        var normals = MeshOperations<double>.ComputeFaceNormals(mesh);

        double nx = normals[0, 0];
        double ny = normals[0, 1];
        double nz = normals[0, 2];
        double length = Math.Sqrt(nx * nx + ny * ny + nz * nz);
        Assert.Equal(1.0, length, 1e-10);
    }

    [Fact]
    public void ComputeVertexNormals_SharedVertex_AveragesAdjacentFaces()
    {
        // Two triangles sharing vertex 0 at origin:
        // Face 0: (0,0,0), (1,0,0), (0,1,0) => normal = (0,0,1)
        // Face 1: (0,0,0), (0,1,0), (0,0,1) => normal computed
        // e1=(0,1,0), e2=(0,0,1) => cross = (1,0,0)
        var mesh = MakeTriangleMesh(
            new double[,] { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } },
            new int[,] { { 0, 1, 2 }, { 0, 2, 3 } }
        );
        var vertexNormals = MeshOperations<double>.ComputeVertexNormals(mesh);

        // Vertex 0 is shared by both faces:
        // avg = normalize((0,0,1) + (1,0,0)) = normalize(1,0,1) = (1/sqrt(2), 0, 1/sqrt(2))
        double invSqrt2 = 1.0 / Math.Sqrt(2);
        Assert.Equal(invSqrt2, vertexNormals[0, 0], 1e-8);
        Assert.Equal(0.0, vertexNormals[0, 1], 1e-8);
        Assert.Equal(invSqrt2, vertexNormals[0, 2], 1e-8);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  MESH VOXELIZATION
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void VoxelizeMeshSurface_SingleTriangle_OccupiesVoxels()
    {
        var mesh = MakeTriangleMesh(
            new double[,] { { 0, 0, 0 }, { 5, 0, 0 }, { 0, 5, 0 } },
            new int[,] { { 0, 1, 2 } }
        );
        var grid = Voxelization<double>.VoxelizeMeshSurface(mesh, resolution: 4);

        int occupied = 0;
        for (int z = 0; z < 4; z++)
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    if (grid.Voxels[z, y, x] > 0.5) occupied++;

        Assert.True(occupied >= 3, $"Expected at least 3 occupied voxels for a triangle, got {occupied}");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  EDGE CASES
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void ChamferDistance_EmptySource_ReturnsZero()
    {
        var empty = MakeCloud(new double[0, 3]);
        var nonempty = MakeCloud(new double[,] { { 1, 2, 3 } });
        double cd = GeometryMetrics<double>.ChamferDistance(empty, nonempty);
        Assert.Equal(0.0, cd, Tolerance);
    }

    [Fact]
    public void HausdorffDistance_EmptySource_ReturnsZero()
    {
        var empty = MakeCloud(new double[0, 3]);
        var nonempty = MakeCloud(new double[,] { { 1, 2, 3 } });
        double hd = GeometryMetrics<double>.HausdorffDistance(empty, nonempty);
        Assert.Equal(0.0, hd, Tolerance);
    }

    [Fact]
    public void Center_SinglePoint_CenteredAtOrigin()
    {
        var cloud = MakeCloud(new double[,] { { 5, 10, 15 } });
        var centered = PointCloudNormalization<double>.Center(cloud);
        Assert.Equal(0.0, centered.Points[0, 0], Tolerance);
        Assert.Equal(0.0, centered.Points[0, 1], Tolerance);
        Assert.Equal(0.0, centered.Points[0, 2], Tolerance);
    }

    [Fact]
    public void UniformSample_PreservesLabels()
    {
        var points = new double[,] { { 0, 0, 0 }, { 1, 0, 0 }, { 2, 0, 0 }, { 3, 0, 0 } };
        var labels = new Vector<double>([10, 20, 30, 40]);
        var cloud = MakeCloud(points, labels);
        var sampled = PointCloudSampling<double>.UniformSample(cloud, 2, seed: 42);

        Assert.NotNull(sampled.Labels);
        Assert.Equal(2, sampled.Labels.Length);
        // Each label must be from original set
        for (int i = 0; i < sampled.Labels.Length; i++)
        {
            double label = sampled.Labels[i];
            Assert.True(label == 10 || label == 20 || label == 30 || label == 40,
                $"Label {label} not in original set");
        }
    }

    [Fact]
    public void VoxelGrid_InvalidDimensions_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            var voxSize = new Vector<double>([0, 1, 1]); // zero voxel size
            var tensor = new Tensor<double>(new double[8], [2, 2, 2]);
            var origin = new Vector<double>([0, 0, 0]);
            _ = new VoxelGridData<double>(tensor, origin, voxSize);
        });
    }

    [Fact]
    public void FPS_PreservesLabels()
    {
        var points = new double[,] { { 0, 0, 0 }, { 5, 0, 0 }, { 10, 0, 0 }, { 15, 0, 0 } };
        var labels = new Vector<double>([1, 2, 3, 4]);
        var cloud = MakeCloud(points, labels);
        var sampled = PointCloudSampling<double>.FarthestPointSample(cloud, 2, seed: 42);

        Assert.NotNull(sampled.Labels);
        Assert.Equal(2, sampled.Labels.Length);
    }
}
