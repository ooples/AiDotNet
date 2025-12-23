using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using AiDotNet.Geometry.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Geometry.IO;

/// <summary>
/// STL mesh IO utilities.
/// </summary>
public static class StlMeshIO
{
    /// <summary>
    /// Loads a triangle mesh from an STL file.
    /// </summary>
    public static TriangleMeshData<T> LoadMesh<T>(string path, bool computeNormalsIfMissing = true)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        }
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("STL file not found.", path);
        }

        using var stream = File.OpenRead(path);
        return LoadMesh<T>(stream, computeNormalsIfMissing);
    }

    /// <summary>
    /// Saves a triangle mesh to an STL file.
    /// </summary>
    public static void SaveMesh<T>(TriangleMeshData<T> mesh, string path, bool binary = true, string? solidName = null)
    {
        if (mesh == null)
        {
            throw new ArgumentNullException(nameof(mesh));
        }
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        }

        using var stream = File.Create(path);
        SaveMesh(mesh, stream, binary, solidName);
    }

    private static TriangleMeshData<T> LoadMesh<T>(Stream stream, bool computeNormalsIfMissing)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (!stream.CanSeek)
        {
            using var memory = new MemoryStream();
            stream.CopyTo(memory);
            memory.Position = 0;
            return LoadMesh<T>(memory, computeNormalsIfMissing);
        }

        long start = stream.Position;
        long remaining = stream.Length - start;
        if (remaining >= 84)
        {
            var header = new byte[80];
            int headerRead = stream.Read(header, 0, header.Length);
            if (headerRead == header.Length)
            {
                var countBytes = new byte[4];
                int countRead = stream.Read(countBytes, 0, countBytes.Length);
                if (countRead == countBytes.Length)
                {
                    uint triangleCount = BitConverter.ToUInt32(countBytes, 0);
                    long expectedLength = 84L + 50L * triangleCount;
                    if (remaining == expectedLength)
                    {
                        if (triangleCount > int.MaxValue)
                        {
                            throw new FormatException("STL triangle count exceeds supported range.");
                        }

                        stream.Position = start + 84;
                        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
                        return ReadBinaryMesh(reader, numOps, (int)triangleCount, computeNormalsIfMissing);
                    }
                }
            }
        }

        stream.Position = start;
        return ReadAsciiMesh(stream, numOps, computeNormalsIfMissing);
    }

    private static void SaveMesh<T>(TriangleMeshData<T> mesh, Stream stream, bool binary, string? solidName)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        string resolvedName = ResolveSolidName(solidName);
        if (binary)
        {
            WriteBinaryMesh(mesh, stream, numOps, resolvedName);
        }
        else
        {
            WriteAsciiMesh(mesh, stream, numOps, resolvedName);
        }
    }

    private static TriangleMeshData<T> ReadBinaryMesh<T>(
        BinaryReader reader,
        INumericOperations<T> numOps,
        int triangleCount,
        bool computeNormalsIfMissing)
    {
        var positions = new List<T>(triangleCount * 9);
        var faceIndices = new List<int>(triangleCount * 3);
        var faceNormals = new List<T>(triangleCount * 3);
        var vertexLookup = new Dictionary<VertexKey, int>();
        bool hasNonZeroNormal = false;

        for (int i = 0; i < triangleCount; i++)
        {
            double nx = reader.ReadSingle();
            double ny = reader.ReadSingle();
            double nz = reader.ReadSingle();
            if (!IsZeroVector(nx, ny, nz))
            {
                hasNonZeroNormal = true;
            }

            int i0 = ReadBinaryVertex(reader, positions, vertexLookup, numOps);
            int i1 = ReadBinaryVertex(reader, positions, vertexLookup, numOps);
            int i2 = ReadBinaryVertex(reader, positions, vertexLookup, numOps);

            faceIndices.Add(i0);
            faceIndices.Add(i1);
            faceIndices.Add(i2);

            faceNormals.Add(numOps.FromDouble(nx));
            faceNormals.Add(numOps.FromDouble(ny));
            faceNormals.Add(numOps.FromDouble(nz));

            _ = reader.ReadUInt16();
        }

        return BuildMesh(positions, faceIndices, faceNormals, hasNonZeroNormal, computeNormalsIfMissing);
    }

    private static TriangleMeshData<T> ReadAsciiMesh<T>(
        Stream stream,
        INumericOperations<T> numOps,
        bool computeNormalsIfMissing)
    {
        var positions = new List<T>();
        var faceIndices = new List<int>();
        var faceNormals = new List<T>();
        var vertexLookup = new Dictionary<VertexKey, int>();
        var currentFace = new List<int>(3);
        double currentNx = 0.0;
        double currentNy = 0.0;
        double currentNz = 0.0;
        bool hasNonZeroNormal = false;
        bool hasNormals = false;

        void FinalizeFace()
        {
            if (currentFace.Count == 0)
            {
                return;
            }
            if (currentFace.Count < 3)
            {
                throw new FormatException("STL facet has fewer than 3 vertices.");
            }

            TriangulateFace(currentFace, faceIndices, faceNormals, numOps, currentNx, currentNy, currentNz);
            if (!IsZeroVector(currentNx, currentNy, currentNz))
            {
                hasNonZeroNormal = true;
            }

            currentFace.Clear();
        }

        using var reader = new StreamReader(stream, Encoding.ASCII, false, 1024, leaveOpen: true);
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0)
            {
                continue;
            }

            var tokens = trimmed.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length == 0)
            {
                continue;
            }

            if (tokens.Length >= 5
                && string.Equals(tokens[0], "facet", StringComparison.OrdinalIgnoreCase)
                && string.Equals(tokens[1], "normal", StringComparison.OrdinalIgnoreCase))
            {
                FinalizeFace();
                currentNx = ParseDouble(tokens[2], "normal X");
                currentNy = ParseDouble(tokens[3], "normal Y");
                currentNz = ParseDouble(tokens[4], "normal Z");
                hasNormals = true;
                continue;
            }

            if (tokens.Length >= 4 && string.Equals(tokens[0], "vertex", StringComparison.OrdinalIgnoreCase))
            {
                double x = ParseDouble(tokens[1], "vertex X");
                double y = ParseDouble(tokens[2], "vertex Y");
                double z = ParseDouble(tokens[3], "vertex Z");
                int index = GetOrAddVertex(positions, vertexLookup, numOps, x, y, z);
                currentFace.Add(index);
                continue;
            }

            if (string.Equals(tokens[0], "endfacet", StringComparison.OrdinalIgnoreCase))
            {
                FinalizeFace();
            }
        }

        FinalizeFace();

        if (!hasNormals)
        {
            hasNonZeroNormal = false;
        }

        return BuildMesh(positions, faceIndices, faceNormals, hasNonZeroNormal, computeNormalsIfMissing);
    }

    private static TriangleMeshData<T> BuildMesh<T>(
        List<T> positions,
        List<int> faceIndices,
        List<T> faceNormals,
        bool hasNonZeroNormal,
        bool computeNormalsIfMissing)
    {
        var verticesTensor = new Tensor<T>(positions.ToArray(), new[] { positions.Count / 3, 3 });
        var facesTensor = new Tensor<int>(faceIndices.ToArray(), new[] { faceIndices.Count / 3, 3 });

        Tensor<T>? faceNormalsTensor = null;
        if (faceNormals.Count == faceIndices.Count && faceIndices.Count > 0)
        {
            faceNormalsTensor = new Tensor<T>(faceNormals.ToArray(), new[] { faceIndices.Count / 3, 3 });
        }

        var mesh = new TriangleMeshData<T>(verticesTensor, facesTensor, faceNormals: faceNormalsTensor);
        if (faceNormalsTensor == null || !hasNonZeroNormal)
        {
            mesh.ComputeFaceNormals();
        }

        if (computeNormalsIfMissing)
        {
            mesh.ComputeVertexNormals();
        }

        return mesh;
    }

    private static void WriteBinaryMesh<T>(
        TriangleMeshData<T> mesh,
        Stream stream,
        INumericOperations<T> numOps,
        string solidName)
    {
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
        var headerBytes = Encoding.ASCII.GetBytes(solidName);
        var header = new byte[80];
        int headerCount = Math.Min(headerBytes.Length, header.Length);
        Array.Copy(headerBytes, header, headerCount);
        writer.Write(header);
        writer.Write((uint)mesh.NumFaces);

        var vertices = mesh.Vertices.Data;
        var faces = mesh.Faces.Data;
        var faceNormals = mesh.FaceNormals ?? mesh.ComputeFaceNormals();

        for (int f = 0; f < mesh.NumFaces; f++)
        {
            int faceOffset = f * 3;
            int normalOffset = f * 3;
            writer.Write((float)numOps.ToDouble(faceNormals.Data[normalOffset]));
            writer.Write((float)numOps.ToDouble(faceNormals.Data[normalOffset + 1]));
            writer.Write((float)numOps.ToDouble(faceNormals.Data[normalOffset + 2]));

            WriteVertexBinary(writer, vertices, numOps, faces[faceOffset]);
            WriteVertexBinary(writer, vertices, numOps, faces[faceOffset + 1]);
            WriteVertexBinary(writer, vertices, numOps, faces[faceOffset + 2]);

            writer.Write((ushort)0);
        }
    }

    private static void WriteAsciiMesh<T>(
        TriangleMeshData<T> mesh,
        Stream stream,
        INumericOperations<T> numOps,
        string solidName)
    {
        using var writer = new StreamWriter(stream, Encoding.ASCII, 1024, leaveOpen: true);
        writer.WriteLine($"solid {solidName}");

        var vertices = mesh.Vertices.Data;
        var faces = mesh.Faces.Data;
        var faceNormals = mesh.FaceNormals ?? mesh.ComputeFaceNormals();

        for (int f = 0; f < mesh.NumFaces; f++)
        {
            int faceOffset = f * 3;
            int normalOffset = f * 3;
            writer.WriteLine(string.Format(
                CultureInfo.InvariantCulture,
                "facet normal {0} {1} {2}",
                numOps.ToDouble(faceNormals.Data[normalOffset]),
                numOps.ToDouble(faceNormals.Data[normalOffset + 1]),
                numOps.ToDouble(faceNormals.Data[normalOffset + 2])));
            writer.WriteLine("outer loop");

            WriteVertexAscii(writer, vertices, numOps, faces[faceOffset]);
            WriteVertexAscii(writer, vertices, numOps, faces[faceOffset + 1]);
            WriteVertexAscii(writer, vertices, numOps, faces[faceOffset + 2]);

            writer.WriteLine("endloop");
            writer.WriteLine("endfacet");
        }

        writer.WriteLine($"endsolid {solidName}");
        writer.Flush();
    }

    private static int ReadBinaryVertex<T>(
        BinaryReader reader,
        List<T> positions,
        Dictionary<VertexKey, int> vertexLookup,
        INumericOperations<T> numOps)
    {
        double x = reader.ReadSingle();
        double y = reader.ReadSingle();
        double z = reader.ReadSingle();
        return GetOrAddVertex(positions, vertexLookup, numOps, x, y, z);
    }

    private static int GetOrAddVertex<T>(
        List<T> positions,
        Dictionary<VertexKey, int> vertexLookup,
        INumericOperations<T> numOps,
        double x,
        double y,
        double z)
    {
        var key = new VertexKey(x, y, z);
        if (!vertexLookup.TryGetValue(key, out int index))
        {
            index = positions.Count / 3;
            positions.Add(numOps.FromDouble(x));
            positions.Add(numOps.FromDouble(y));
            positions.Add(numOps.FromDouble(z));
            vertexLookup[key] = index;
        }

        return index;
    }

    private static void TriangulateFace<T>(
        List<int> polygon,
        List<int> indices,
        List<T> faceNormals,
        INumericOperations<T> numOps,
        double nx,
        double ny,
        double nz)
    {
        int first = polygon[0];
        for (int i = 1; i < polygon.Count - 1; i++)
        {
            indices.Add(first);
            indices.Add(polygon[i]);
            indices.Add(polygon[i + 1]);

            faceNormals.Add(numOps.FromDouble(nx));
            faceNormals.Add(numOps.FromDouble(ny));
            faceNormals.Add(numOps.FromDouble(nz));
        }
    }

    private static void WriteVertexBinary<T>(BinaryWriter writer, T[] vertices, INumericOperations<T> numOps, int index)
    {
        int offset = index * 3;
        writer.Write((float)numOps.ToDouble(vertices[offset]));
        writer.Write((float)numOps.ToDouble(vertices[offset + 1]));
        writer.Write((float)numOps.ToDouble(vertices[offset + 2]));
    }

    private static void WriteVertexAscii<T>(StreamWriter writer, T[] vertices, INumericOperations<T> numOps, int index)
    {
        int offset = index * 3;
        writer.WriteLine(string.Format(
            CultureInfo.InvariantCulture,
            "vertex {0} {1} {2}",
            numOps.ToDouble(vertices[offset]),
            numOps.ToDouble(vertices[offset + 1]),
            numOps.ToDouble(vertices[offset + 2])));
    }

    private static string ResolveSolidName(string? solidName)
    {
        return string.IsNullOrWhiteSpace(solidName) ? "AiDotNet" : solidName.Trim();
    }

    private static double ParseDouble(string token, string label)
    {
        if (!double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
        {
            throw new FormatException($"Invalid STL {label} value '{token}'.");
        }
        return value;
    }

    private static bool IsZeroVector(double x, double y, double z)
    {
        return x == 0.0 && y == 0.0 && z == 0.0;
    }

    private readonly struct VertexKey : IEquatable<VertexKey>
    {
        public VertexKey(double x, double y, double z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public double X { get; }
        public double Y { get; }
        public double Z { get; }

        public bool Equals(VertexKey other)
        {
            return X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z);
        }

        public override bool Equals(object? obj)
        {
            return obj is VertexKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(X, Y, Z);
        }
    }
}
