using AiDotNet.Geometry.Data;
using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Geometry.Preprocessing;

/// <summary>
/// Provides mesh processing operations for triangle meshes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class MeshOperations<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes face normals for a triangle mesh.
    /// </summary>
    public static Tensor<T> ComputeFaceNormals(TriangleMeshData<T> mesh)
    {
        if (mesh.NumFaces == 0)
        {
            return new Tensor<T>(new[] { 0, 3 });
        }

        var crossProducts = GeometryEngineOps<T>.FaceCrossProducts(mesh.Vertices, mesh.Faces);
        return GeometryEngineOps<T>.NormalizeRowsAbove(crossProducts, NumOps.FromDouble(1e-10));
    }

    /// <summary>
    /// Computes vertex normals by averaging adjacent face normals.
    /// </summary>
    public static Tensor<T> ComputeVertexNormals(TriangleMeshData<T> mesh)
    {
        var faceNormals = ComputeFaceNormals(mesh);
        var sums = GeometryEngineOps<T>.AccumulateToVertices(faceNormals, mesh.Faces, mesh.NumVertices);
        return GeometryEngineOps<T>.NormalizeRowsAbove(sums, NumOps.FromDouble(1e-10));
    }

    /// <summary>
    /// Builds a vertex adjacency list (which vertices are connected by edges).
    /// </summary>
    public static List<int>[] BuildVertexAdjacency(TriangleMeshData<T> mesh)
    {
        int numVertices = mesh.NumVertices;

        var adjacency = new List<int>[numVertices];
        for (int v = 0; v < numVertices; v++)
        {
            adjacency[v] = [];
        }

        var faces = mesh.Faces.Data.ToArray();
        for (int offset = 0; offset + 2 < faces.Length; offset += 3)
        {
            int v0 = faces[offset];
            int v1 = faces[offset + 1];
            int v2 = faces[offset + 2];

            AddEdge(adjacency, v0, v1);
            AddEdge(adjacency, v1, v2);
            AddEdge(adjacency, v2, v0);
        }

        return adjacency;
    }

    /// <summary>
    /// Samples points uniformly from the mesh surface.
    /// </summary>
    public static PointCloudData<T> SamplePoints(
        TriangleMeshData<T> mesh,
        int numSamples,
        int? seed = null,
        bool includeNormals = true)
    {
        int numFaces = mesh.NumFaces;
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        var areas = new double[numFaces];
        double totalArea = 0;

        for (int f = 0; f < numFaces; f++)
        {
            areas[f] = ComputeFaceArea(mesh, f);
            totalArea += areas[f];
        }

        var cdf = new double[numFaces];
        double cumulative = 0;
        for (int f = 0; f < numFaces; f++)
        {
            cumulative += areas[f] / totalArea;
            cdf[f] = cumulative;
        }

        Tensor<T>? faceNormals = includeNormals ? ComputeFaceNormals(mesh) : null;

        int numFeatures = includeNormals ? 6 : 3;
        var pointData = new T[numSamples * numFeatures];

        for (int i = 0; i < numSamples; i++)
        {
            double r = random.NextDouble();
            int faceIdx = Array.BinarySearch(cdf, r);
            if (faceIdx < 0)
            {
                faceIdx = ~faceIdx;
            }
            faceIdx = Math.Min(faceIdx, numFaces - 1);

            double u = random.NextDouble();
            double v = random.NextDouble();
            if (u + v > 1)
            {
                u = 1 - u;
                v = 1 - v;
            }
            double w = 1 - u - v;

            int v0 = mesh.Faces[faceIdx, 0];
            int v1 = mesh.Faces[faceIdx, 1];
            int v2 = mesh.Faces[faceIdx, 2];

            double px = w * NumOps.ToDouble(mesh.Vertices[v0, 0]) +
                       u * NumOps.ToDouble(mesh.Vertices[v1, 0]) +
                       v * NumOps.ToDouble(mesh.Vertices[v2, 0]);
            double py = w * NumOps.ToDouble(mesh.Vertices[v0, 1]) +
                       u * NumOps.ToDouble(mesh.Vertices[v1, 1]) +
                       v * NumOps.ToDouble(mesh.Vertices[v2, 1]);
            double pz = w * NumOps.ToDouble(mesh.Vertices[v0, 2]) +
                       u * NumOps.ToDouble(mesh.Vertices[v1, 2]) +
                       v * NumOps.ToDouble(mesh.Vertices[v2, 2]);

            int baseIdx = i * numFeatures;
            pointData[baseIdx] = NumOps.FromDouble(px);
            pointData[baseIdx + 1] = NumOps.FromDouble(py);
            pointData[baseIdx + 2] = NumOps.FromDouble(pz);

            if (includeNormals && faceNormals != null)
            {
                pointData[baseIdx + 3] = faceNormals[faceIdx, 0];
                pointData[baseIdx + 4] = faceNormals[faceIdx, 1];
                pointData[baseIdx + 5] = faceNormals[faceIdx, 2];
            }
        }

        var tensor = new Tensor<T>(pointData, [numSamples, numFeatures]);
        return new PointCloudData<T>(tensor);
    }

    /// <summary>
    /// Computes mesh statistics (surface area, volume, bounding box).
    /// </summary>
    public static MeshStatistics<T> ComputeStatistics(TriangleMeshData<T> mesh)
    {
        var engine = AiDotNetEngine.Current;
        int numFaces = mesh.NumFaces;
        int numVertices = mesh.NumVertices;

        var boundsMin = (NumOps.MaxValue, NumOps.MaxValue, NumOps.MaxValue);
        var boundsMax = (NumOps.MinValue, NumOps.MinValue, NumOps.MinValue);
        if (numVertices > 0)
        {
            // Math.Min/Math.Max propagate NaN, so a column holding NaN reports NaN for both bounds.
            var (min, max) = GeometryEngineOps<T>.ColumnBounds(mesh.Vertices);
            var hasNan = GeometryEngineOps<T>.ColumnHasNan(mesh.Vertices);
            var nan = engine.TensorFullLike(min, NumOps.FromDouble(double.NaN));
            min = engine.TensorWhere(hasNan, nan, min);
            max = engine.TensorWhere(hasNan, nan, max);
            boundsMin = (min[0], min[1], min[2]);
            boundsMax = (max[0], max[1], max[2]);
        }

        T surfaceArea = NumOps.Zero;
        T signedVolume = NumOps.Zero;
        if (numFaces > 0)
        {
            var crossProducts = GeometryEngineOps<T>.FaceCrossProducts(mesh.Vertices, mesh.Faces);
            var doubleAreas = engine.TensorSqrt(GeometryEngineOps<T>.RowSquaredNorms(crossProducts));
            surfaceArea = NumOps.Multiply(NumOps.FromDouble(0.5), engine.TensorSum(doubleAreas));

            // Divergence theorem: each face contributes p0 . (p1 x p2) / 6.
            var p0 = GeometryEngineOps<T>.FaceCorner(mesh.Vertices, mesh.Faces, 0);
            var p1 = GeometryEngineOps<T>.FaceCorner(mesh.Vertices, mesh.Faces, 1);
            var p2 = GeometryEngineOps<T>.FaceCorner(mesh.Vertices, mesh.Faces, 2);
            var tripleProducts = engine.TensorMultiply(p0, engine.TensorCross(p1, p2, 1));
            signedVolume = NumOps.Divide(engine.TensorSum(tripleProducts), NumOps.FromDouble(6.0));
        }

        return new MeshStatistics<T>(
            NumVertices: numVertices,
            NumFaces: numFaces,
            NumEdges: CountEdges(mesh),
            SurfaceArea: surfaceArea,
            Volume: NumOps.Abs(signedVolume),
            BoundingBoxMin: boundsMin,
            BoundingBoxMax: boundsMax);
    }

    private static double ComputeFaceArea(TriangleMeshData<T> mesh, int faceIdx)
    {
        int v0 = mesh.Faces[faceIdx, 0];
        int v1 = mesh.Faces[faceIdx, 1];
        int v2 = mesh.Faces[faceIdx, 2];

        double x0 = NumOps.ToDouble(mesh.Vertices[v0, 0]);
        double y0 = NumOps.ToDouble(mesh.Vertices[v0, 1]);
        double z0 = NumOps.ToDouble(mesh.Vertices[v0, 2]);

        double x1 = NumOps.ToDouble(mesh.Vertices[v1, 0]);
        double y1 = NumOps.ToDouble(mesh.Vertices[v1, 1]);
        double z1 = NumOps.ToDouble(mesh.Vertices[v1, 2]);

        double x2 = NumOps.ToDouble(mesh.Vertices[v2, 0]);
        double y2 = NumOps.ToDouble(mesh.Vertices[v2, 1]);
        double z2 = NumOps.ToDouble(mesh.Vertices[v2, 2]);

        double e1x = x1 - x0, e1y = y1 - y0, e1z = z1 - z0;
        double e2x = x2 - x0, e2y = y2 - y0, e2z = z2 - z0;

        double cx = e1y * e2z - e1z * e2y;
        double cy = e1z * e2x - e1x * e2z;
        double cz = e1x * e2y - e1y * e2x;

        return 0.5 * Math.Sqrt(cx * cx + cy * cy + cz * cz);
    }

    private static int CountEdges(TriangleMeshData<T> mesh)
    {
        var edges = new HashSet<(int, int)>();
        var faces = mesh.Faces.Data.ToArray();

        for (int offset = 0; offset + 2 < faces.Length; offset += 3)
        {
            int v0 = faces[offset];
            int v1 = faces[offset + 1];
            int v2 = faces[offset + 2];

            edges.Add((Math.Min(v0, v1), Math.Max(v0, v1)));
            edges.Add((Math.Min(v1, v2), Math.Max(v1, v2)));
            edges.Add((Math.Min(v2, v0), Math.Max(v2, v0)));
        }

        return edges.Count;
    }

    private static void AddEdge(List<int>[] adjacency, int v1, int v2)
    {
        if (!adjacency[v1].Contains(v2))
        {
            adjacency[v1].Add(v2);
        }
        if (!adjacency[v2].Contains(v1))
        {
            adjacency[v2].Add(v1);
        }
    }
}
/// <summary>
/// Contains statistics about a triangle mesh.
/// </summary>
/// <typeparam name="T">The numeric type of the mesh coordinates.</typeparam>
public record MeshStatistics<T>(
    int NumVertices,
    int NumFaces,
    int NumEdges,
    T SurfaceArea,
    T Volume,
    (T X, T Y, T Z) BoundingBoxMin,
    (T X, T Y, T Z) BoundingBoxMax);
