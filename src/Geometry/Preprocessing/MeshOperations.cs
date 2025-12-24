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
        int numFaces = mesh.NumFaces;
        var normals = new T[numFaces * 3];

        for (int f = 0; f < numFaces; f++)
        {
            int v0 = mesh.Faces[f, 0];
            int v1 = mesh.Faces[f, 1];
            int v2 = mesh.Faces[f, 2];

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

            double nx = e1y * e2z - e1z * e2y;
            double ny = e1z * e2x - e1x * e2z;
            double nz = e1x * e2y - e1y * e2x;

            double length = Math.Sqrt(nx * nx + ny * ny + nz * nz);
            if (length > 1e-10)
            {
                nx /= length;
                ny /= length;
                nz /= length;
            }

            normals[f * 3] = NumOps.FromDouble(nx);
            normals[f * 3 + 1] = NumOps.FromDouble(ny);
            normals[f * 3 + 2] = NumOps.FromDouble(nz);
        }

        return new Tensor<T>(normals, [numFaces, 3]);
    }

    /// <summary>
    /// Computes vertex normals by averaging adjacent face normals.
    /// </summary>
    public static Tensor<T> ComputeVertexNormals(TriangleMeshData<T> mesh)
    {
        int numVertices = mesh.NumVertices;
        int numFaces = mesh.NumFaces;
        
        var normalSums = new double[numVertices * 3];
        var faceNormals = ComputeFaceNormals(mesh);

        for (int f = 0; f < numFaces; f++)
        {
            double nx = NumOps.ToDouble(faceNormals[f, 0]);
            double ny = NumOps.ToDouble(faceNormals[f, 1]);
            double nz = NumOps.ToDouble(faceNormals[f, 2]);

            for (int i = 0; i < 3; i++)
            {
                int v = mesh.Faces[f, i];
                normalSums[v * 3] += nx;
                normalSums[v * 3 + 1] += ny;
                normalSums[v * 3 + 2] += nz;
            }
        }

        var normals = new T[numVertices * 3];
        for (int v = 0; v < numVertices; v++)
        {
            double nx = normalSums[v * 3];
            double ny = normalSums[v * 3 + 1];
            double nz = normalSums[v * 3 + 2];

            double length = Math.Sqrt(nx * nx + ny * ny + nz * nz);
            if (length > 1e-10)
            {
                nx /= length;
                ny /= length;
                nz /= length;
            }

            normals[v * 3] = NumOps.FromDouble(nx);
            normals[v * 3 + 1] = NumOps.FromDouble(ny);
            normals[v * 3 + 2] = NumOps.FromDouble(nz);
        }

        return new Tensor<T>(normals, [numVertices, 3]);
    }

    /// <summary>
    /// Builds a vertex adjacency list (which vertices are connected by edges).
    /// </summary>
    public static List<int>[] BuildVertexAdjacency(TriangleMeshData<T> mesh)
    {
        int numVertices = mesh.NumVertices;
        int numFaces = mesh.NumFaces;

        var adjacency = new List<int>[numVertices];
        for (int v = 0; v < numVertices; v++)
        {
            adjacency[v] = [];
        }

        for (int f = 0; f < numFaces; f++)
        {
            int v0 = mesh.Faces[f, 0];
            int v1 = mesh.Faces[f, 1];
            int v2 = mesh.Faces[f, 2];

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
    public static MeshStatistics ComputeStatistics(TriangleMeshData<T> mesh)
    {
        int numFaces = mesh.NumFaces;
        int numVertices = mesh.NumVertices;

        double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;

        for (int v = 0; v < numVertices; v++)
        {
            double x = NumOps.ToDouble(mesh.Vertices[v, 0]);
            double y = NumOps.ToDouble(mesh.Vertices[v, 1]);
            double z = NumOps.ToDouble(mesh.Vertices[v, 2]);

            minX = Math.Min(minX, x);
            minY = Math.Min(minY, y);
            minZ = Math.Min(minZ, z);
            maxX = Math.Max(maxX, x);
            maxY = Math.Max(maxY, y);
            maxZ = Math.Max(maxZ, z);
        }

        double surfaceArea = 0;
        for (int f = 0; f < numFaces; f++)
        {
            surfaceArea += ComputeFaceArea(mesh, f);
        }

        double signedVolume = 0;
        for (int f = 0; f < numFaces; f++)
        {
            int v0 = mesh.Faces[f, 0];
            int v1 = mesh.Faces[f, 1];
            int v2 = mesh.Faces[f, 2];

            double x0 = NumOps.ToDouble(mesh.Vertices[v0, 0]);
            double y0 = NumOps.ToDouble(mesh.Vertices[v0, 1]);
            double z0 = NumOps.ToDouble(mesh.Vertices[v0, 2]);

            double x1 = NumOps.ToDouble(mesh.Vertices[v1, 0]);
            double y1 = NumOps.ToDouble(mesh.Vertices[v1, 1]);
            double z1 = NumOps.ToDouble(mesh.Vertices[v1, 2]);

            double x2 = NumOps.ToDouble(mesh.Vertices[v2, 0]);
            double y2 = NumOps.ToDouble(mesh.Vertices[v2, 1]);
            double z2 = NumOps.ToDouble(mesh.Vertices[v2, 2]);

            signedVolume += (x0 * (y1 * z2 - y2 * z1) +
                            x1 * (y2 * z0 - y0 * z2) +
                            x2 * (y0 * z1 - y1 * z0)) / 6.0;
        }

        return new MeshStatistics(
            NumVertices: numVertices,
            NumFaces: numFaces,
            NumEdges: CountEdges(mesh),
            SurfaceArea: surfaceArea,
            Volume: Math.Abs(signedVolume),
            BoundingBoxMin: (minX, minY, minZ),
            BoundingBoxMax: (maxX, maxY, maxZ));
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
        int numFaces = mesh.NumFaces;
        var edges = new HashSet<(int, int)>();

        for (int f = 0; f < numFaces; f++)
        {
            int v0 = mesh.Faces[f, 0];
            int v1 = mesh.Faces[f, 1];
            int v2 = mesh.Faces[f, 2];

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
public record MeshStatistics(
    int NumVertices,
    int NumFaces,
    int NumEdges,
    double SurfaceArea,
    double Volume,
    (double X, double Y, double Z) BoundingBoxMin,
    (double X, double Y, double Z) BoundingBoxMax);
