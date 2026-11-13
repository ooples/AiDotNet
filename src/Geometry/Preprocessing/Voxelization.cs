using AiDotNet.Geometry.Data;
using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Geometry.Preprocessing;

/// <summary>
/// Provides voxelization utilities for converting point clouds and meshes to voxel grids.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class Voxelization<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Converts a point cloud to an occupancy voxel grid.
    /// </summary>
    public static VoxelGridData<T> VoxelizePointCloud(
        PointCloudData<T> pointCloud,
        int resolution,
        double padding = 0.1)
    {
        int numPoints = pointCloud.NumPoints;
        if (numPoints == 0 || resolution <= 0)
        {
            return CreateEmptyGrid(resolution);
        }

        if (pointCloud.NumFeatures < 3)
        {
            throw new ArgumentException(
                $"VoxelizePointCloud requires at least 3 features (xyz coordinates). Got {pointCloud.NumFeatures} features.",
                nameof(pointCloud));
        }

        if (UsesDoubleGeometry)
        {
            return FromDouble(Voxelization<double>.VoxelizePointCloud(
                new PointCloudData<double>(pointCloud.Points.Cast<double>()), resolution, padding));
        }

        var positions = GeometryEngineOps<T>.Positions(pointCloud.Points);
        var (origin, voxelSize) = GridLayout(positions, resolution, padding);
        var voxels = MarkCells(
            new Tensor<T>(new[] { resolution, resolution, resolution }), positions, origin, voxelSize, resolution);
        return CreateGrid(voxels, origin, voxelSize);
    }

    /// <summary>
    /// Computes the padded cubic grid around the positions: the origin (bounding-box minimum minus the padding)
    /// and the edge length of one voxel. Bounds follow Math.Min/Math.Max, so NaN coordinates propagate.
    /// </summary>
    private static (Tensor<T> origin, T voxelSize) GridLayout(Tensor<T> positions, int resolution, double padding)
    {
        var engine = AiDotNetEngine.Current;
        var (min, max) = GeometryEngineOps<T>.PropagatingColumnBounds(positions);
        T maxRange = GeometryEngineOps<T>.PropagatingMax(engine.TensorSubtract(max, min));

        if (NumOps.LessThan(maxRange, NumOps.FromDouble(1e-10)))
        {
            maxRange = NumOps.One;
        }

        T padAmount = NumOps.Multiply(maxRange, NumOps.FromDouble(padding));
        var origin = engine.TensorSubtractScalar(min, padAmount);
        T extent = NumOps.Add(maxRange, NumOps.Multiply(NumOps.FromDouble(2), padAmount));
        return (origin, NumOps.Divide(extent, NumOps.FromDouble(resolution)));
    }

    /// <summary>
    /// Marks the voxel containing each [count, 3] position as occupied. Cell coordinates are floored and clamped
    /// to the grid (NaN coordinates land in cell 0, as the saturating integer conversion places them).
    /// </summary>
    private static Tensor<T> MarkCells(Tensor<T> voxels, Tensor<T> positions, Tensor<T> origin, T voxelSize, int resolution)
    {
        var engine = AiDotNetEngine.Current;
        var relative = engine.TensorSubtract(positions, engine.TensorBroadcastTo(origin.Reshape(1, 3), positions.Shape.ToArray()));
        var cells = engine.TensorFloor(engine.TensorDivideScalar(relative, voxelSize));
        cells = engine.TensorWhere(engine.TensorIsNan(cells), engine.TensorZerosLike(cells), cells);
        cells = engine.TensorClamp(cells, NumOps.Zero, NumOps.FromDouble(resolution - 1));

        // The clamped cells are whole numbers in [0, resolution), so converting them to int gives the index
        // directly; the indices never pass through T, where Half stops being exact above 2048.
        var cellIndices = cells.Cast<int>();
        var ix = GeometryEngineOps<T>.Column(cellIndices, 0);
        var iy = GeometryEngineOps<T>.Column(cellIndices, 1);
        var iz = GeometryEngineOps<T>.Column(cellIndices, 2);
        return engine.TensorIndexPut(voxels, new[] { iz, iy, ix }, engine.TensorOnesLike(GeometryEngineOps<T>.Column(cells, 0)), false);
    }

    private static VoxelGridData<T> CreateGrid(Tensor<T> voxels, Tensor<T> origin, T voxelSize)
    {
        var voxelSizeVec = new Vector<T>([voxelSize, voxelSize, voxelSize]);
        return new VoxelGridData<T>(voxels, origin.ToVector(), voxelSizeVec);
    }

    /// <summary>
    /// Converts a triangle mesh to an occupancy voxel grid using surface voxelization.
    /// </summary>
    public static VoxelGridData<T> VoxelizeMeshSurface(
        TriangleMeshData<T> mesh,
        int resolution,
        double padding = 0.1)
    {
        int numVertices = mesh.NumVertices;
        int numFaces = mesh.NumFaces;

        if (numVertices == 0 || numFaces == 0 || resolution <= 0)
        {
            return CreateEmptyGrid(resolution);
        }

        if (mesh.Vertices.Shape.Length < 2 || mesh.Vertices.Shape[1] < 3)
        {
            throw new ArgumentException(
                $"VoxelizeMeshSurface requires at least 3 vertex coordinates (xyz). Got {(mesh.Vertices.Shape.Length >= 2 ? mesh.Vertices.Shape[1] : 0)} columns.",
                nameof(mesh));
        }

        if (mesh.Faces.Shape.Length < 2 || mesh.Faces.Shape[1] < 3)
        {
            throw new ArgumentException(
                $"VoxelizeMeshSurface requires at least 3 face indices per triangle. Got {(mesh.Faces.Shape.Length >= 2 ? mesh.Faces.Shape[1] : 0)} columns.",
                nameof(mesh));
        }

        if (UsesDoubleGeometry)
        {
            return FromDouble(Voxelization<double>.VoxelizeMeshSurface(
                new TriangleMeshData<double>(mesh.Vertices.Cast<double>(), mesh.Faces, validateIndices: false),
                resolution,
                padding));
        }

        var engine = AiDotNetEngine.Current;
        var vertices = GeometryEngineOps<T>.Positions(mesh.Vertices);
        var (origin, voxelSize) = GridLayout(vertices, resolution, padding);
        var voxels = new Tensor<T>(new[] { resolution, resolution, resolution });

        // Skip faces that reference a vertex outside the mesh.
        var faceData = mesh.Faces.ToArray();
        int faceStride = mesh.Faces.Shape[1];
        var validFaces = new List<int>();
        for (int f = 0; f < numFaces; f++)
        {
            int v0 = faceData[f * faceStride];
            int v1 = faceData[f * faceStride + 1];
            int v2 = faceData[f * faceStride + 2];
            if (v0 >= 0 && v0 < numVertices && v1 >= 0 && v1 < numVertices && v2 >= 0 && v2 < numVertices)
            {
                validFaces.Add(f);
            }
        }

        if (validFaces.Count > 0)
        {
            // Barycentric sample weights u = i / s, v = j / s for i + j <= s, in the (i, j) scan order.
            int samplesPerEdge = Math.Max(1, resolution / 4);
            var steps = engine.TensorArange<T>(0, samplesPerEdge + 1, 1);
            int side = samplesPerEdge + 1;
            var rowSteps = engine.TensorBroadcastTo(steps.Reshape(side, 1), new[] { side, side });
            var columnSteps = engine.TensorBroadcastTo(steps.Reshape(1, side), new[] { side, side });
            var inside = engine.TensorLessEqual(
                engine.TensorAdd(rowSteps, columnSteps), engine.TensorFullLike(rowSteps, NumOps.FromDouble(samplesPerEdge)));
            var stepPairs = engine.TensorNonzero(inside);
            var stepRows = GeometryEngineOps<T>.Column(stepPairs, 0);
            var stepColumns = GeometryEngineOps<T>.Column(stepPairs, 1);
            var divisor = NumOps.FromDouble(samplesPerEdge);
            var u = engine.TensorDivideScalar(engine.TensorIndexSelect(steps, stepRows, 0), divisor);
            var v = engine.TensorDivideScalar(engine.TensorIndexSelect(steps, stepColumns, 0), divisor);
            var w = engine.ScalarMinusTensor(NumOps.One, engine.TensorAdd(u, v));
            int samples = u.Shape[0];

            var faces = engine.TensorIndexSelect(
                mesh.Faces, new Tensor<int>(validFaces.ToArray(), new[] { validFaces.Count }), 0);
            int chunk = Math.Max(1, (1 << 20) / samples);
            for (int start = 0; start < validFaces.Count; start += chunk)
            {
                int count = Math.Min(chunk, validFaces.Count - start);
                var chunkFaces = engine.TensorSlice(faces, new[] { start, 0 }, new[] { count, faces.Shape[1] });
                var points = BarycentricPoints(vertices, chunkFaces, w, u, v);
                voxels = MarkCells(voxels, points.Reshape(count * samples, 3), origin, voxelSize, resolution);
            }
        }

        return CreateGrid(voxels, origin, voxelSize);
    }

    private static Tensor<T> BarycentricPoints(Tensor<T> vertices, Tensor<int> faces, Tensor<T> w, Tensor<T> u, Tensor<T> v)
    {
        var engine = AiDotNetEngine.Current;
        int count = faces.Shape[0];
        int samples = w.Shape[0];
        var shape = new[] { count, samples, 3 };
        Tensor<T> Weighted(int corner, Tensor<T> weight)
        {
            var position = GeometryEngineOps<T>.FaceCorner(vertices, faces, corner).Reshape(count, 1, 3);
            return engine.TensorMultiply(
                engine.TensorBroadcastTo(weight.Reshape(1, samples, 1), shape),
                engine.TensorBroadcastTo(position, shape));
        }

        return engine.TensorAdd(engine.TensorAdd(Weighted(0, w), Weighted(1, u)), Weighted(2, v));
    }

    /// <summary>
    /// Computes the Intersection over Union (IoU) between two voxel grids.
    /// </summary>
    public static double IntersectionOverUnion(
        VoxelGridData<T> prediction,
        VoxelGridData<T> groundTruth,
        double threshold = 0.5)
    {
        if (prediction.Depth != groundTruth.Depth ||
            prediction.Height != groundTruth.Height ||
            prediction.Width != groundTruth.Width)
        {
            throw new ArgumentException("Voxel grids must have the same dimensions");
        }

        var engine = AiDotNetEngine.Current;
        T thresholdT = NumOps.FromDouble(threshold);
        var predOccupied = engine.TensorGreaterThan(prediction.Voxels, thresholdT);
        var gtOccupied = engine.TensorGreaterThan(groundTruth.Voxels, thresholdT);

        int intersection = engine.TensorCountNonzero(engine.TensorMultiply(predOccupied, gtOccupied));
        int union = engine.TensorCountNonzero(engine.TensorMax(predOccupied, gtOccupied));

        return union > 0 ? (double)intersection / union : 0;
    }

    /// <summary>
    /// Applies morphological dilation to a voxel grid.
    /// </summary>
    public static VoxelGridData<T> Dilate(VoxelGridData<T> voxelGrid, int radius = 1)
    {
        var engine = AiDotNetEngine.Current;
        var occupied = engine.TensorGreaterThan(voxelGrid.Voxels, NumOps.Zero);

        // A voxel is set when any in-bounds voxel of the (2r + 1)^3 cube around it is occupied: a max-pool over
        // the occupancy padded with empty voxels.
        var result = radius < 0
            ? engine.TensorZerosLike(occupied)
            : CubeMax(occupied, radius, NumOps.Zero);
        return new VoxelGridData<T>(result, voxelGrid.Origin, voxelGrid.VoxelSize);
    }

    /// <summary>
    /// Applies morphological erosion to a voxel grid.
    /// </summary>
    public static VoxelGridData<T> Erode(VoxelGridData<T> voxelGrid, int radius = 1)
    {
        var engine = AiDotNetEngine.Current;
        var occupied = engine.TensorGreaterThan(voxelGrid.Voxels, NumOps.Zero);

        // A voxel survives when every voxel of the (2r + 1)^3 cube around it is in bounds and occupied: the
        // complement of a max-pool over the empty mask padded with "empty" (out-of-bounds) voxels.
        Tensor<T> result;
        if (radius < 0)
        {
            result = engine.TensorOnesLike(occupied);
        }
        else
        {
            var empty = engine.ScalarMinusTensor(NumOps.One, occupied);
            result = engine.ScalarMinusTensor(NumOps.One, CubeMax(empty, radius, NumOps.One));
        }

        return new VoxelGridData<T>(result, voxelGrid.Origin, voxelGrid.VoxelSize);
    }

    private static Tensor<T> CubeMax(Tensor<T> mask, int radius, T outside)
    {
        var engine = AiDotNetEngine.Current;
        int depth = mask.Shape[0];
        int height = mask.Shape[1];
        int width = mask.Shape[2];
        if (radius == 0)
        {
            return mask;
        }

        // The cube maximum is separable: a (2r + 1)-wide maximum along width, then height, then depth, each over the
        // grid padded with the outside value on that axis only. Three 1-axis pools cost O(r) per voxel instead of
        // the O(r^3) of one cubic window, and give the identical maximum.
        int size = 2 * radius + 1;
        var current = mask;
        for (int dim = 2; dim >= 0; dim--)
        {
            // Padding pairs run from the last dimension backwards.
            var padding = new int[6];
            padding[2 * (2 - dim)] = radius;
            padding[2 * (2 - dim) + 1] = radius;
            var kernel = new[] { dim == 0 ? size : 1, dim == 1 ? size : 1, dim == 2 ? size : 1 };
            var padded = engine.TensorConstantPad(current, padding, outside).Reshape(
                1, 1,
                depth + (dim == 0 ? 2 * radius : 0),
                height + (dim == 1 ? 2 * radius : 0),
                width + (dim == 2 ? 2 * radius : 0));
            current = engine.MaxPool3D(padded, kernel, new[] { 1, 1, 1 }, new[] { 0, 0, 0 }).Reshape(depth, height, width);
        }

        return current;
    }

    /// <summary>
    /// True for the types the grid geometry is computed for in double, as the scalar code did: integral types
    /// (whose fractional barycentric weights and voxel sizes truncate to zero) and Half (whose ranges and
    /// products lose the cell boundaries).
    /// </summary>
    private static bool UsesDoubleGeometry
        => typeof(T) == typeof(Half) || NumOps.Equals(NumOps.FromDouble(0.5), NumOps.Zero);

    private static VoxelGridData<T> FromDouble(VoxelGridData<double> grid)
        => new VoxelGridData<T>(
            grid.Voxels.Cast<T>(),
            Tensor<double>.FromVector(grid.Origin).Cast<T>().ToVector(),
            Tensor<double>.FromVector(grid.VoxelSize).Cast<T>().ToVector());

    private static VoxelGridData<T> CreateEmptyGrid(int resolution)
    {
        resolution = Math.Max(1, resolution);
        var voxelTensor = new Tensor<T>(new T[resolution * resolution * resolution], [resolution, resolution, resolution]);
        var origin = new Vector<T>([NumOps.Zero, NumOps.Zero, NumOps.Zero]);
        var voxelSize = new Vector<T>([NumOps.One, NumOps.One, NumOps.One]);
        return new VoxelGridData<T>(voxelTensor, origin, voxelSize);
    }
}
