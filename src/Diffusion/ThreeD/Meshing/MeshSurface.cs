using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// A triangle mesh in double precision: the working form of every extractor and reconstructor here, converted to
/// <see cref="Mesh3D{T}"/> once at the end.
/// </summary>
internal sealed class MeshSurface
{
    internal MeshSurface(double[] coordinates, int[] triangles, double[]? normals)
    {
        if (coordinates.Length % 3 != 0)
            throw new ArgumentException("Coordinates must come in (x, y, z) triples.", nameof(coordinates));
        if (triangles.Length % 3 != 0)
            throw new ArgumentException("Triangles must come in vertex-index triples.", nameof(triangles));
        if (normals is not null && normals.Length != coordinates.Length)
            throw new ArgumentException("There must be one normal per vertex.", nameof(normals));

        Coordinates = coordinates;
        Triangles = triangles;
        Normals = normals;
    }

    /// <summary>Vertex coordinates, (x, y, z) per vertex.</summary>
    internal double[] Coordinates { get; }

    /// <summary>Vertex indices, three per triangle, counter-clockwise seen from outside.</summary>
    internal int[] Triangles { get; }

    /// <summary>Unit vertex normals, or null when the mesh carries none.</summary>
    internal double[]? Normals { get; private set; }

    internal int VertexCount => Coordinates.Length / 3;

    internal int TriangleCount => Triangles.Length / 3;

    /// <summary>
    /// Replaces zero-length vertex normals by the normalised area-weighted sum of the adjacent face normals (or,
    /// when the mesh has no normals, computes all of them that way).
    /// </summary>
    internal void RepairDegenerateNormals()
    {
        int vertexCount = VertexCount;
        var normals = Normals ?? new double[Coordinates.Length];
        bool anyDegenerate = Normals is null;
        for (int v = 0; v < vertexCount && !anyDegenerate; v++)
        {
            if (normals[3 * v] == 0.0 && normals[(3 * v) + 1] == 0.0 && normals[(3 * v) + 2] == 0.0)
                anyDegenerate = true;
        }

        if (!anyDegenerate)
            return;

        var faceSum = new double[Coordinates.Length];
        for (int f = 0; f < TriangleCount; f++)
        {
            int a = Triangles[3 * f], b = Triangles[(3 * f) + 1], c = Triangles[(3 * f) + 2];
            var (nx, ny, nz) = FaceCross(a, b, c);
            foreach (int v in new[] { a, b, c })
            {
                faceSum[3 * v] += nx;
                faceSum[(3 * v) + 1] += ny;
                faceSum[(3 * v) + 2] += nz;
            }
        }

        for (int v = 0; v < vertexCount; v++)
        {
            if (Normals is not null
                && (normals[3 * v] != 0.0 || normals[(3 * v) + 1] != 0.0 || normals[(3 * v) + 2] != 0.0))
                continue;

            double x = faceSum[3 * v], y = faceSum[(3 * v) + 1], z = faceSum[(3 * v) + 2];
            double length = Math.Sqrt((x * x) + (y * y) + (z * z));
            if (length > 0.0)
            {
                normals[3 * v] = x / length;
                normals[(3 * v) + 1] = y / length;
                normals[(3 * v) + 2] = z / length;
            }
        }

        Normals = normals;
    }

    /// <summary>
    /// The cross product (b - a) x (c - a): the face normal scaled by twice the triangle's area.
    /// </summary>
    internal (double X, double Y, double Z) FaceCross(int a, int b, int c)
    {
        double ux = Coordinates[3 * b] - Coordinates[3 * a];
        double uy = Coordinates[(3 * b) + 1] - Coordinates[(3 * a) + 1];
        double uz = Coordinates[(3 * b) + 2] - Coordinates[(3 * a) + 2];
        double vx = Coordinates[3 * c] - Coordinates[3 * a];
        double vy = Coordinates[(3 * c) + 1] - Coordinates[(3 * a) + 1];
        double vz = Coordinates[(3 * c) + 2] - Coordinates[(3 * a) + 2];
        return ((uy * vz) - (uz * vy), (uz * vx) - (ux * vz), (ux * vy) - (uy * vx));
    }

    /// <summary>
    /// The signed volume enclosed by the mesh (divergence theorem); positive when a closed mesh faces outward.
    /// </summary>
    internal double SignedVolume()
    {
        double volume = 0.0;
        for (int f = 0; f < TriangleCount; f++)
        {
            int a = 3 * Triangles[3 * f], b = 3 * Triangles[(3 * f) + 1], c = 3 * Triangles[(3 * f) + 2];
            double ax = Coordinates[a], ay = Coordinates[a + 1], az = Coordinates[a + 2];
            double bx = Coordinates[b], by = Coordinates[b + 1], bz = Coordinates[b + 2];
            double cx = Coordinates[c], cy = Coordinates[c + 1], cz = Coordinates[c + 2];
            volume += (ax * ((by * cz) - (bz * cy))) + (ay * ((bz * cx) - (bx * cz))) + (az * ((bx * cy) - (by * cx)));
        }

        return volume / 6.0;
    }

    /// <summary>
    /// Counts the directed edges whose reverse is missing (boundary or inconsistently wound edges) and the undirected
    /// edges used by more than two triangles. Both are zero for a closed, consistently oriented, manifold mesh.
    /// </summary>
    internal (int UnmatchedEdges, int NonManifoldEdges) EdgeDefects()
    {
        var directed = new Dictionary<long, int>();
        var undirected = new Dictionary<long, int>();
        long n = VertexCount;
        for (int f = 0; f < TriangleCount; f++)
        {
            for (int k = 0; k < 3; k++)
            {
                long a = Triangles[(3 * f) + k], b = Triangles[(3 * f) + ((k + 1) % 3)];
                long key = (a * n) + b;
                directed[key] = directed.TryGetValue(key, out int d) ? d + 1 : 1;
                long undirectedKey = (Math.Min(a, b) * n) + Math.Max(a, b);
                undirected[undirectedKey] = undirected.TryGetValue(undirectedKey, out int u) ? u + 1 : 1;
            }
        }

        int unmatched = 0;
        foreach (var entry in directed)
        {
            long a = entry.Key / n, b = entry.Key % n;
            int reverse = directed.TryGetValue((b * n) + a, out int r) ? r : 0;
            if (reverse != entry.Value)
                unmatched++;
        }

        int nonManifold = 0;
        foreach (var entry in undirected)
        {
            if (entry.Value > 2)
                nonManifold++;
        }

        return (unmatched, nonManifold);
    }

    /// <summary>
    /// Converts to a <see cref="Mesh3D{T}"/> with one bulk conversion of the coordinates (and normals) to T.
    /// </summary>
    internal Mesh3D<T> ToMesh<T>()
    {
        int vertexCount = VertexCount;
        int triangleCount = TriangleCount;
        var faces = new int[triangleCount, 3];
        for (int f = 0; f < triangleCount; f++)
        {
            faces[f, 0] = Triangles[3 * f];
            faces[f, 1] = Triangles[(3 * f) + 1];
            faces[f, 2] = Triangles[(3 * f) + 2];
        }

        if (vertexCount == 0)
            return new Mesh3D<T>(new Tensor<T>(new[] { 0, 3 }), faces) { Normals = new Tensor<T>(new[] { 0, 3 }) };

        var mesh = new Mesh3D<T>(ToTensor<T>(Coordinates, vertexCount), faces);
        if (Normals is not null)
            mesh.Normals = ToTensor<T>(Normals, vertexCount);
        return mesh;
    }

    private static Tensor<T> ToTensor<T>(double[] values, int rows)
    {
        return new Tensor<double>(new[] { rows, 3 }, new Vector<double>(values)).Cast<T>();
    }
}
