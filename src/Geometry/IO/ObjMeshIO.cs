using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using AiDotNet.Geometry.Data;
using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Geometry.IO;

/// <summary>
/// OBJ mesh and point cloud IO utilities.
/// </summary>
public static class ObjMeshIO
{
    /// <summary>
    /// Loads a triangle mesh from an OBJ file.
    /// </summary>
    public static TriangleMeshData<T> LoadMesh<T>(string path, bool computeNormalsIfMissing = true)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        }
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("OBJ file not found.", path);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var positions = new List<T>();
        var uvs = new List<T>();
        var normals = new List<T>();
        var vertexColors = new List<T>();
        int colorChannels = 0;

        var finalVertices = new List<T>();
        var finalUvs = new List<T>();
        var finalNormals = new List<T>();
        var finalColors = new List<T>();
        bool hasFaceUvs = false;
        bool hasFaceNormals = false;
        bool hasVertexColors = false;

        var faceIndices = new List<int>();
        var keyToIndex = new Dictionary<ObjVertexKey, int>();

        foreach (var rawLine in File.ReadLines(path))
        {
            var line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
            {
                continue;
            }

            if (line.StartsWith("v ", StringComparison.Ordinal))
            {
                var parts = SplitLine(line);
                if (parts.Length < 4)
                {
                    throw new FormatException("OBJ vertex line must have at least 3 coordinates.");
                }

                int vertexIndex = positions.Count / 3;
                var x = ParseDouble(parts[1], "vertex X");
                var y = ParseDouble(parts[2], "vertex Y");
                var z = ParseDouble(parts[3], "vertex Z");
                positions.Add(numOps.FromDouble(x));
                positions.Add(numOps.FromDouble(y));
                positions.Add(numOps.FromDouble(z));

                int detectedColorChannels = DetectColorChannels(parts.Length);
                if (detectedColorChannels > 0)
                {
                    if (colorChannels == 0)
                    {
                        colorChannels = detectedColorChannels;
                        EnsureColorStorage(vertexColors, vertexIndex, colorChannels, numOps);
                    }

                    if (detectedColorChannels != colorChannels)
                    {
                        throw new FormatException("OBJ vertex color channel count is inconsistent.");
                    }

                    hasVertexColors = true;
                    AppendVertexColors(vertexColors, parts, colorChannels, numOps);
                }
                else if (colorChannels > 0)
                {
                    EnsureColorStorage(vertexColors, vertexIndex, colorChannels, numOps);
                    AppendDefaultColors(vertexColors, colorChannels, numOps);
                }

                continue;
            }

            if (line.StartsWith("vn ", StringComparison.Ordinal))
            {
                var parts = SplitLine(line);
                if (parts.Length < 4)
                {
                    throw new FormatException("OBJ normal line must have 3 components.");
                }

                normals.Add(numOps.FromDouble(ParseDouble(parts[1], "normal X")));
                normals.Add(numOps.FromDouble(ParseDouble(parts[2], "normal Y")));
                normals.Add(numOps.FromDouble(ParseDouble(parts[3], "normal Z")));
                continue;
            }

            if (line.StartsWith("vt ", StringComparison.Ordinal))
            {
                var parts = SplitLine(line);
                if (parts.Length < 3)
                {
                    throw new FormatException("OBJ texture coordinate line must have at least 2 components.");
                }

                uvs.Add(numOps.FromDouble(ParseDouble(parts[1], "texture U")));
                uvs.Add(numOps.FromDouble(ParseDouble(parts[2], "texture V")));
                continue;
            }

            if (line.StartsWith("f ", StringComparison.Ordinal))
            {
                var parts = SplitLine(line);
                if (parts.Length < 4)
                {
                    throw new FormatException("OBJ face line must have at least 3 vertices.");
                }

                var polygon = new List<int>(parts.Length - 1);
                for (int i = 1; i < parts.Length; i++)
                {
                    var faceVertex = ParseFaceVertex(parts[i], positions.Count / 3, uvs.Count / 2, normals.Count / 3);
                    if (faceVertex.TextureIndex >= 0 && !hasFaceUvs)
                    {
                        hasFaceUvs = true;
                        EnsureUvStorage(finalUvs, finalVertices.Count / 3, numOps);
                    }
                    if (faceVertex.NormalIndex >= 0 && !hasFaceNormals)
                    {
                        hasFaceNormals = true;
                        EnsureNormalStorage(finalNormals, finalVertices.Count / 3, numOps);
                    }

                    int finalIndex;
                    var key = new ObjVertexKey(faceVertex.PositionIndex, faceVertex.TextureIndex, faceVertex.NormalIndex);
                    if (!keyToIndex.TryGetValue(key, out finalIndex))
                    {
                        finalIndex = finalVertices.Count / 3;
                        keyToIndex[key] = finalIndex;
                        AppendPosition(finalVertices, positions, faceVertex.PositionIndex);

                        if (hasFaceUvs)
                        {
                            if (faceVertex.TextureIndex >= 0)
                            {
                                AppendUv(finalUvs, uvs, faceVertex.TextureIndex);
                            }
                            else
                            {
                                AppendDefaultUv(finalUvs, numOps);
                            }
                        }

                        if (hasFaceNormals)
                        {
                            if (faceVertex.NormalIndex >= 0)
                            {
                                AppendNormal(finalNormals, normals, faceVertex.NormalIndex);
                            }
                            else
                            {
                                AppendDefaultNormal(finalNormals, numOps);
                            }
                        }

                        if (hasVertexColors && colorChannels > 0)
                        {
                            AppendColor(finalColors, vertexColors, faceVertex.PositionIndex, colorChannels);
                        }
                    }

                    polygon.Add(finalIndex);
                }

                TriangulateFace(polygon, faceIndices);
            }
        }

        var verticesTensor = new Tensor<T>(finalVertices.ToArray(), new[] { finalVertices.Count / 3, 3 });
        var facesTensor = new Tensor<int>(faceIndices.ToArray(), new[] { faceIndices.Count / 3, 3 });

        Tensor<T>? vertexNormals = null;
        Tensor<T>? vertexUvs = null;
        Tensor<T>? vertexColorsTensor = null;

        if (hasFaceNormals)
        {
            vertexNormals = new Tensor<T>(finalNormals.ToArray(), new[] { finalNormals.Count / 3, 3 });
        }
        if (hasFaceUvs)
        {
            vertexUvs = new Tensor<T>(finalUvs.ToArray(), new[] { finalUvs.Count / 2, 2 });
        }
        if (hasVertexColors && colorChannels > 0)
        {
            vertexColorsTensor = new Tensor<T>(finalColors.ToArray(), new[] { finalColors.Count / colorChannels, colorChannels });
        }

        var mesh = new TriangleMeshData<T>(verticesTensor, facesTensor, vertexNormals, vertexColorsTensor, vertexUvs);
        if (computeNormalsIfMissing && vertexNormals == null)
        {
            mesh.ComputeVertexNormals();
        }

        return mesh;
    }

    /// <summary>
    /// Loads a point cloud from an OBJ file (using vertex lines).
    /// </summary>
    public static PointCloudData<T> LoadPointCloud<T>(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        }
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("OBJ file not found.", path);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var points = new List<T>();
        var colors = new List<T>();
        int colorChannels = 0;

        foreach (var rawLine in File.ReadLines(path))
        {
            var line = rawLine.Trim();
            if (!line.StartsWith("v ", StringComparison.Ordinal))
            {
                continue;
            }

            var parts = SplitLine(line);
            if (parts.Length < 4)
            {
                continue;
            }

            points.Add(numOps.FromDouble(ParseDouble(parts[1], "vertex X")));
            points.Add(numOps.FromDouble(ParseDouble(parts[2], "vertex Y")));
            points.Add(numOps.FromDouble(ParseDouble(parts[3], "vertex Z")));

            int detectedColorChannels = DetectColorChannels(parts.Length);
            if (detectedColorChannels > 0)
            {
                if (colorChannels == 0)
                {
                    colorChannels = detectedColorChannels;
                }
                if (detectedColorChannels != colorChannels)
                {
                    throw new FormatException("OBJ vertex color channel count is inconsistent.");
                }

                AppendVertexColors(colors, parts, colorChannels, numOps);
            }
            else if (colorChannels > 0)
            {
                AppendDefaultColors(colors, colorChannels, numOps);
            }
        }

        int pointCount = points.Count / 3;
        int featureDim = 3 + colorChannels;
        var data = new T[pointCount * featureDim];

        for (int i = 0; i < pointCount; i++)
        {
            int pointOffset = i * 3;
            int outputOffset = i * featureDim;
            data[outputOffset] = points[pointOffset];
            data[outputOffset + 1] = points[pointOffset + 1];
            data[outputOffset + 2] = points[pointOffset + 2];

            if (colorChannels > 0)
            {
                int colorOffset = i * colorChannels;
                for (int c = 0; c < colorChannels; c++)
                {
                    data[outputOffset + 3 + c] = colors[colorOffset + c];
                }
            }
        }

        var tensor = new Tensor<T>(data, new[] { pointCount, featureDim });
        return new PointCloudData<T>(tensor);
    }

    /// <summary>
    /// Saves a triangle mesh to an OBJ file.
    /// </summary>
    public static void SaveMesh<T>(
        TriangleMeshData<T> mesh,
        string path,
        bool includeNormals = true,
        bool includeUvs = true,
        bool includeColors = false)
    {
        if (mesh == null)
        {
            throw new ArgumentNullException(nameof(mesh));
        }
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        using var writer = new StreamWriter(path);

        var vertices = mesh.Vertices.Data;
        var vertexCount = mesh.NumVertices;

        Tensor<T>? colors = includeColors ? mesh.VertexColors : null;
        int colorChannels = colors == null ? 0 : colors.Shape[1];

        for (int i = 0; i < vertexCount; i++)
        {
            int offset = i * 3;
            string line = string.Format(
                CultureInfo.InvariantCulture,
                "v {0} {1} {2}",
                numOps.ToDouble(vertices[offset]),
                numOps.ToDouble(vertices[offset + 1]),
                numOps.ToDouble(vertices[offset + 2]));

            if (colors != null && colorChannels > 0)
            {
                int colorOffset = i * colorChannels;
                for (int c = 0; c < colorChannels; c++)
                {
                    line += string.Format(
                        CultureInfo.InvariantCulture,
                        " {0}",
                        numOps.ToDouble(colors.Data[colorOffset + c]));
                }
            }

            writer.WriteLine(line);
        }

        Tensor<T>? normals = includeNormals ? mesh.VertexNormals : null;
        if (normals != null)
        {
            for (int i = 0; i < vertexCount; i++)
            {
                int offset = i * 3;
                writer.WriteLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "vn {0} {1} {2}",
                    numOps.ToDouble(normals.Data[offset]),
                    numOps.ToDouble(normals.Data[offset + 1]),
                    numOps.ToDouble(normals.Data[offset + 2])));
            }
        }

        Tensor<T>? uvs = includeUvs ? mesh.VertexUVs : null;
        if (uvs != null)
        {
            for (int i = 0; i < vertexCount; i++)
            {
                int offset = i * 2;
                writer.WriteLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "vt {0} {1}",
                    numOps.ToDouble(uvs.Data[offset]),
                    numOps.ToDouble(uvs.Data[offset + 1])));
            }
        }

        var faces = mesh.Faces.Data;
        var faceCount = mesh.NumFaces;
        bool hasNormals = normals != null;
        bool hasUvs = uvs != null;

        for (int f = 0; f < faceCount; f++)
        {
            int faceOffset = f * 3;
            int i0 = faces[faceOffset] + 1;
            int i1 = faces[faceOffset + 1] + 1;
            int i2 = faces[faceOffset + 2] + 1;

            if (hasUvs && hasNormals)
            {
                writer.WriteLine($"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}");
            }
            else if (hasUvs)
            {
                writer.WriteLine($"f {i0}/{i0} {i1}/{i1} {i2}/{i2}");
            }
            else if (hasNormals)
            {
                writer.WriteLine($"f {i0}//{i0} {i1}//{i1} {i2}//{i2}");
            }
            else
            {
                writer.WriteLine($"f {i0} {i1} {i2}");
            }
        }
    }

    /// <summary>
    /// Saves a point cloud to an OBJ file (vertex lines only).
    /// </summary>
    public static void SavePointCloud<T>(PointCloudData<T> pointCloud, string path, bool includeColors = false)
    {
        if (pointCloud == null)
        {
            throw new ArgumentNullException(nameof(pointCloud));
        }
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        int featureDim = pointCloud.NumFeatures;
        if (featureDim < 3)
        {
            throw new ArgumentException("Point cloud must have at least 3 features (XYZ).", nameof(pointCloud));
        }

        int colorOffset = includeColors && featureDim >= 6 ? 3 : -1;

        using var writer = new StreamWriter(path);
        for (int i = 0; i < pointCloud.NumPoints; i++)
        {
            int offset = i * featureDim;
            string line = string.Format(
                CultureInfo.InvariantCulture,
                "v {0} {1} {2}",
                numOps.ToDouble(pointCloud.Points.Data[offset]),
                numOps.ToDouble(pointCloud.Points.Data[offset + 1]),
                numOps.ToDouble(pointCloud.Points.Data[offset + 2]));

            if (colorOffset >= 0)
            {
                line += string.Format(
                    CultureInfo.InvariantCulture,
                    " {0} {1} {2}",
                    numOps.ToDouble(pointCloud.Points.Data[offset + colorOffset]),
                    numOps.ToDouble(pointCloud.Points.Data[offset + colorOffset + 1]),
                    numOps.ToDouble(pointCloud.Points.Data[offset + colorOffset + 2]));
            }

            writer.WriteLine(line);
        }
    }

    private static string[] SplitLine(string line)
    {
        return line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
    }

    private static double ParseDouble(string token, string label)
    {
        if (!double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
        {
            throw new FormatException($"Invalid {label} value '{token}'.");
        }
        return value;
    }

    private static int DetectColorChannels(int tokenCount)
    {
        if (tokenCount >= 8)
        {
            return 4;
        }
        if (tokenCount >= 7)
        {
            return 3;
        }
        return 0;
    }

    private static void EnsureColorStorage<T>(List<T> colors, int existingVertices, int channels, INumericOperations<T> numOps)
    {
        int target = existingVertices * channels;
        while (colors.Count < target)
        {
            colors.Add(numOps.Zero);
        }
    }

    private static void AppendVertexColors<T>(List<T> colors, string[] parts, int channels, INumericOperations<T> numOps)
    {
        int start = 4;
        for (int i = 0; i < channels; i++)
        {
            colors.Add(numOps.FromDouble(ParseDouble(parts[start + i], "vertex color")));
        }
    }

    private static void AppendDefaultColors<T>(List<T> colors, int channels, INumericOperations<T> numOps)
    {
        for (int i = 0; i < channels; i++)
        {
            colors.Add(numOps.Zero);
        }
    }

    private static ObjFaceVertex ParseFaceVertex(string token, int positionCount, int uvCount, int normalCount)
    {
        var parts = token.Split('/');
        if (parts.Length == 0 || parts[0].Length == 0)
        {
            throw new FormatException("OBJ face vertex is missing a position index.");
        }

        int positionIndex = ParseIndex(parts[0], positionCount, "position");
        int uvIndex = -1;
        int normalIndex = -1;

        if (parts.Length > 1 && parts[1].Length > 0)
        {
            uvIndex = ParseIndex(parts[1], uvCount, "texture coordinate");
        }
        if (parts.Length > 2 && parts[2].Length > 0)
        {
            normalIndex = ParseIndex(parts[2], normalCount, "normal");
        }

        return new ObjFaceVertex(positionIndex, uvIndex, normalIndex);
    }

    private static int ParseIndex(string token, int count, string label)
    {
        if (!int.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out var rawIndex))
        {
            throw new FormatException($"Invalid OBJ {label} index '{token}'.");
        }
        if (rawIndex == 0)
        {
            throw new FormatException($"OBJ {label} index cannot be zero.");
        }

        int index = rawIndex > 0 ? rawIndex - 1 : count + rawIndex;
        if (index < 0 || index >= count)
        {
            throw new ArgumentOutOfRangeException(nameof(token), $"OBJ {label} index {rawIndex} is out of range.");
        }

        return index;
    }

    private static void AppendPosition<T>(List<T> destination, List<T> positions, int index)
    {
        int offset = index * 3;
        destination.Add(positions[offset]);
        destination.Add(positions[offset + 1]);
        destination.Add(positions[offset + 2]);
    }

    private static void AppendUv<T>(List<T> destination, List<T> uvs, int index)
    {
        int offset = index * 2;
        destination.Add(uvs[offset]);
        destination.Add(uvs[offset + 1]);
    }

    private static void AppendNormal<T>(List<T> destination, List<T> normals, int index)
    {
        int offset = index * 3;
        destination.Add(normals[offset]);
        destination.Add(normals[offset + 1]);
        destination.Add(normals[offset + 2]);
    }

    private static void AppendColor<T>(List<T> destination, List<T> colors, int index, int channels)
    {
        int offset = index * channels;
        for (int i = 0; i < channels; i++)
        {
            destination.Add(colors[offset + i]);
        }
    }

    private static void EnsureUvStorage<T>(List<T> uvs, int existingVertices, INumericOperations<T> numOps)
    {
        int target = existingVertices * 2;
        while (uvs.Count < target)
        {
            uvs.Add(numOps.Zero);
        }
    }

    private static void EnsureNormalStorage<T>(List<T> normals, int existingVertices, INumericOperations<T> numOps)
    {
        int target = existingVertices * 3;
        while (normals.Count < target)
        {
            normals.Add(numOps.Zero);
        }
    }

    private static void AppendDefaultUv<T>(List<T> uvs, INumericOperations<T> numOps)
    {
        uvs.Add(numOps.Zero);
        uvs.Add(numOps.Zero);
    }

    private static void AppendDefaultNormal<T>(List<T> normals, INumericOperations<T> numOps)
    {
        normals.Add(numOps.Zero);
        normals.Add(numOps.Zero);
        normals.Add(numOps.Zero);
    }

    private static void TriangulateFace(List<int> polygon, List<int> indices)
    {
        if (polygon.Count < 3)
        {
            return;
        }

        int first = polygon[0];
        for (int i = 1; i < polygon.Count - 1; i++)
        {
            indices.Add(first);
            indices.Add(polygon[i]);
            indices.Add(polygon[i + 1]);
        }
    }

    private readonly struct ObjFaceVertex
    {
        public ObjFaceVertex(int positionIndex, int textureIndex, int normalIndex)
        {
            PositionIndex = positionIndex;
            TextureIndex = textureIndex;
            NormalIndex = normalIndex;
        }

        public int PositionIndex { get; }
        public int TextureIndex { get; }
        public int NormalIndex { get; }
    }

    private readonly struct ObjVertexKey : IEquatable<ObjVertexKey>
    {
        public ObjVertexKey(int positionIndex, int textureIndex, int normalIndex)
        {
            PositionIndex = positionIndex;
            TextureIndex = textureIndex;
            NormalIndex = normalIndex;
        }

        public int PositionIndex { get; }
        public int TextureIndex { get; }
        public int NormalIndex { get; }

        public bool Equals(ObjVertexKey other)
        {
            return PositionIndex == other.PositionIndex
                && TextureIndex == other.TextureIndex
                && NormalIndex == other.NormalIndex;
        }

        public override bool Equals(object? obj)
        {
            return obj is ObjVertexKey other && Equals(other);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 31 + PositionIndex;
                hash = hash * 31 + TextureIndex;
                hash = hash * 31 + NormalIndex;
                return hash;
            }
        }
    }
}
