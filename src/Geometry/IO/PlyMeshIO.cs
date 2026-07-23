using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using AiDotNet.Geometry.Data;
using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Geometry.IO;

/// <summary>
/// PLY mesh and point cloud IO utilities.
/// </summary>
public static class PlyMeshIO
{
    /// <summary>
    /// Loads a triangle mesh from a PLY file.
    /// </summary>
    public static TriangleMeshData<T> LoadMesh<T>(string path, bool computeNormalsIfMissing = true)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        }
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("PLY file not found.", path);
        }

        using var stream = File.OpenRead(path);
        return LoadMesh<T>(stream, computeNormalsIfMissing);
    }

    /// <summary>
    /// Loads a point cloud from a PLY file (vertex element only).
    /// </summary>
    public static PointCloudData<T> LoadPointCloud<T>(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        }
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("PLY file not found.", path);
        }

        using var stream = File.OpenRead(path);
        return LoadPointCloud<T>(stream);
    }

    /// <summary>
    /// Saves a triangle mesh to a PLY file.
    /// </summary>
    public static void SaveMesh<T>(TriangleMeshData<T> mesh, string path, bool binary = false)
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
        SaveMesh(mesh, stream, binary);
    }

    /// <summary>
    /// Saves a point cloud to a PLY file.
    /// </summary>
    public static void SavePointCloud<T>(
        PointCloudData<T> pointCloud,
        string path,
        bool binary = false,
        bool includeColors = false,
        bool includeNormals = false,
        bool includeUvs = false)
    {
        if (pointCloud == null)
        {
            throw new ArgumentNullException(nameof(pointCloud));
        }
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        }

        using var stream = File.Create(path);
        SavePointCloud(pointCloud, stream, binary, includeColors, includeNormals, includeUvs);
    }

    private static TriangleMeshData<T> LoadMesh<T>(Stream stream, bool computeNormalsIfMissing)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var header = ReadHeader(stream);
        var vertexElement = FindElement(header.Elements, "vertex");
        if (vertexElement == null)
        {
            throw new FormatException("PLY file is missing a vertex element.");
        }

        var vertexProperties = vertexElement.Properties;
        var vertexFlags = AnalyzeVertexProperties(vertexProperties);
        var positions = new List<T>(vertexElement.Count * 3);
        var normals = vertexFlags.HasNormals ? new List<T>(vertexElement.Count * 3) : new List<T>();
        var colors = vertexFlags.ColorChannels > 0 ? new List<T>(vertexElement.Count * vertexFlags.ColorChannels) : new List<T>();
        var uvs = vertexFlags.HasUvs ? new List<T>(vertexElement.Count * 2) : new List<T>();

        var faceIndices = new List<int>();
        var faceNormals = new List<T>();

        if (header.Format == PlyFormat.Ascii)
        {
            using var reader = new StreamReader(stream, Encoding.ASCII, false, 1024, leaveOpen: true);
            foreach (var element in header.Elements)
            {
                if (string.Equals(element.Name, "vertex", StringComparison.Ordinal))
                {
                    ReadAsciiVertices(reader, element, vertexFlags, numOps, positions, normals, colors, uvs);
                }
                else if (string.Equals(element.Name, "face", StringComparison.Ordinal))
                {
                    ReadAsciiFaces(reader, element, faceIndices, faceNormals, numOps);
                }
                else
                {
                    SkipAsciiElement(reader, element.Count);
                }
            }
        }
        else if (header.Format == PlyFormat.BinaryLittleEndian)
        {
            using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
            foreach (var element in header.Elements)
            {
                if (string.Equals(element.Name, "vertex", StringComparison.Ordinal))
                {
                    ReadBinaryVertices(reader, element, vertexFlags, numOps, positions, normals, colors, uvs);
                }
                else if (string.Equals(element.Name, "face", StringComparison.Ordinal))
                {
                    ReadBinaryFaces(reader, element, faceIndices, faceNormals, numOps);
                }
                else
                {
                    SkipBinaryElement(reader, element);
                }
            }
        }
        else
        {
            throw new NotSupportedException("Binary big endian PLY is not supported.");
        }

        var verticesTensor = new Tensor<T>(positions.ToArray(), new[] { positions.Count / 3, 3 });
        var facesTensor = new Tensor<int>(faceIndices.ToArray(), new[] { faceIndices.Count / 3, 3 });

        Tensor<T>? vertexNormals = vertexFlags.HasNormals ? new Tensor<T>(normals.ToArray(), new[] { normals.Count / 3, 3 }) : null;
        Tensor<T>? vertexColors = vertexFlags.ColorChannels > 0 ? new Tensor<T>(colors.ToArray(), new[] { colors.Count / vertexFlags.ColorChannels, vertexFlags.ColorChannels }) : null;
        Tensor<T>? vertexUvs = vertexFlags.HasUvs ? new Tensor<T>(uvs.ToArray(), new[] { uvs.Count / 2, 2 }) : null;
        Tensor<T>? faceNormalsTensor = faceNormals.Count > 0 ? new Tensor<T>(faceNormals.ToArray(), new[] { faceNormals.Count / 3, 3 }) : null;

        var mesh = new TriangleMeshData<T>(verticesTensor, facesTensor, vertexNormals, vertexColors, vertexUvs, faceNormalsTensor);
        if (computeNormalsIfMissing && vertexNormals == null)
        {
            mesh.ComputeVertexNormals();
        }

        return mesh;
    }
    private static PointCloudData<T> LoadPointCloud<T>(Stream stream)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var header = ReadHeader(stream);
        var vertexElement = FindElement(header.Elements, "vertex");
        if (vertexElement == null)
        {
            throw new FormatException("PLY file is missing a vertex element.");
        }

        var vertexFlags = AnalyzeVertexProperties(vertexElement.Properties);
        var positions = new List<T>(vertexElement.Count * 3);
        var normals = vertexFlags.HasNormals ? new List<T>(vertexElement.Count * 3) : new List<T>();
        var colors = vertexFlags.ColorChannels > 0 ? new List<T>(vertexElement.Count * vertexFlags.ColorChannels) : new List<T>();
        var uvs = vertexFlags.HasUvs ? new List<T>(vertexElement.Count * 2) : new List<T>();

        if (header.Format == PlyFormat.Ascii)
        {
            using var reader = new StreamReader(stream, Encoding.ASCII, false, 1024, leaveOpen: true);
            foreach (var element in header.Elements)
            {
                if (string.Equals(element.Name, "vertex", StringComparison.Ordinal))
                {
                    ReadAsciiVertices(reader, element, vertexFlags, numOps, positions, normals, colors, uvs);
                }
                else
                {
                    SkipAsciiElement(reader, element.Count);
                }
            }
        }
        else if (header.Format == PlyFormat.BinaryLittleEndian)
        {
            using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
            foreach (var element in header.Elements)
            {
                if (string.Equals(element.Name, "vertex", StringComparison.Ordinal))
                {
                    ReadBinaryVertices(reader, element, vertexFlags, numOps, positions, normals, colors, uvs);
                }
                else
                {
                    SkipBinaryElement(reader, element);
                }
            }
        }
        else
        {
            throw new NotSupportedException("Binary big endian PLY is not supported.");
        }

        int pointCount = positions.Count / 3;
        int featureDim = 3 + (vertexFlags.ColorChannels > 0 ? vertexFlags.ColorChannels : 0)
            + (vertexFlags.HasNormals ? 3 : 0)
            + (vertexFlags.HasUvs ? 2 : 0);

        var data = new T[pointCount * featureDim];

        for (int i = 0; i < pointCount; i++)
        {
            int outputOffset = i * featureDim;
            int posOffset = i * 3;
            data[outputOffset] = positions[posOffset];
            data[outputOffset + 1] = positions[posOffset + 1];
            data[outputOffset + 2] = positions[posOffset + 2];

            int featureOffset = outputOffset + 3;
            if (vertexFlags.ColorChannels > 0)
            {
                int colorOffset = i * vertexFlags.ColorChannels;
                for (int c = 0; c < vertexFlags.ColorChannels; c++)
                {
                    data[featureOffset + c] = colors[colorOffset + c];
                }
                featureOffset += vertexFlags.ColorChannels;
            }

            if (vertexFlags.HasNormals)
            {
                int normalOffset = i * 3;
                data[featureOffset] = normals[normalOffset];
                data[featureOffset + 1] = normals[normalOffset + 1];
                data[featureOffset + 2] = normals[normalOffset + 2];
                featureOffset += 3;
            }

            if (vertexFlags.HasUvs)
            {
                int uvOffset = i * 2;
                data[featureOffset] = uvs[uvOffset];
                data[featureOffset + 1] = uvs[uvOffset + 1];
            }
        }

        var tensor = new Tensor<T>(data, new[] { pointCount, featureDim });
        return new PointCloudData<T>(tensor);
    }

    private static void SaveMesh<T>(TriangleMeshData<T> mesh, Stream stream, bool binary)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var hasNormals = mesh.VertexNormals != null;
        var hasUvs = mesh.VertexUVs != null;
        var hasColors = mesh.VertexColors != null;
        int colorChannels = 0;
        if (mesh.VertexColors != null)
        {
            colorChannels = mesh.VertexColors.Shape[1];
        }

        WriteMeshHeader(mesh, stream, binary, hasNormals, hasUvs, colorChannels, mesh.FaceNormals != null);

        if (binary)
        {
            using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
            WriteMeshBinary(mesh, writer, numOps, hasNormals, hasUvs, colorChannels);
        }
        else
        {
            using var writer = new StreamWriter(stream, Encoding.ASCII, 1024, leaveOpen: true);
            WriteMeshAscii(mesh, writer, numOps, hasNormals, hasUvs, colorChannels);
        }
    }

    private static void SavePointCloud<T>(
        PointCloudData<T> pointCloud,
        Stream stream,
        bool binary,
        bool includeColors,
        bool includeNormals,
        bool includeUvs)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int featureDim = pointCloud.NumFeatures;
        if (featureDim < 3)
        {
            throw new ArgumentException("Point cloud must have at least 3 features (XYZ).", nameof(pointCloud));
        }

        int colorChannels = 0;
        int colorOffset = -1;
        int normalOffset = -1;
        int uvOffset = -1;
        int offset = 3;

        if (includeColors)
        {
            if (featureDim < offset + 3)
            {
                throw new ArgumentException("Point cloud does not contain color features.", nameof(pointCloud));
            }
            colorChannels = featureDim >= offset + 4 ? 4 : 3;
            colorOffset = offset;
            offset += colorChannels;
        }

        if (includeNormals)
        {
            if (featureDim < offset + 3)
            {
                throw new ArgumentException("Point cloud does not contain normal features.", nameof(pointCloud));
            }
            normalOffset = offset;
            offset += 3;
        }

        if (includeUvs)
        {
            if (featureDim < offset + 2)
            {
                throw new ArgumentException("Point cloud does not contain UV features.", nameof(pointCloud));
            }
            uvOffset = offset;
        }

        WritePointCloudHeader(pointCloud, stream, binary, includeNormals, includeUvs, colorChannels);

        if (binary)
        {
            using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
            WritePointCloudBinary(pointCloud, writer, numOps, includeNormals, includeUvs, colorChannels, colorOffset, normalOffset, uvOffset);
        }
        else
        {
            using var writer = new StreamWriter(stream, Encoding.ASCII, 1024, leaveOpen: true);
            WritePointCloudAscii(pointCloud, writer, numOps, includeNormals, includeUvs, colorChannels, colorOffset, normalOffset, uvOffset);
        }
    }

    private static PlyHeader ReadHeader(Stream stream)
    {
        var lines = new List<string>();
        var buffer = new List<byte>();
        int value;

        while ((value = stream.ReadByte()) != -1)
        {
            if (value == '\n')
            {
                string line = Encoding.ASCII.GetString(buffer.ToArray()).TrimEnd('\r');
                buffer.Clear();
                lines.Add(line);
                if (string.Equals(line, "end_header", StringComparison.Ordinal))
                {
                    break;
                }
            }
            else
            {
                buffer.Add((byte)value);
            }
        }

        if (lines.Count == 0 || !string.Equals(lines[0], "ply", StringComparison.Ordinal))
        {
            throw new FormatException("PLY header is missing the 'ply' magic string.");
        }

        return ParseHeader(lines);
    }

    private static PlyHeader ParseHeader(List<string> lines)
    {
        PlyFormat format = PlyFormat.Ascii;
        var elements = new List<PlyElement>();
        PlyElement? current = null;

        for (int i = 1; i < lines.Count; i++)
        {
            var line = lines[i];
            if (line.Length == 0)
            {
                continue;
            }

            var parts = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
            {
                continue;
            }

            if (string.Equals(parts[0], "format", StringComparison.Ordinal))
            {
                if (parts.Length < 2)
                {
                    throw new FormatException("PLY format line is invalid.");
                }
                format = ParseFormat(parts[1]);
            }
            else if (string.Equals(parts[0], "element", StringComparison.Ordinal))
            {
                if (parts.Length < 3)
                {
                    throw new FormatException("PLY element line is invalid.");
                }
                if (!int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out var count))
                {
                    throw new FormatException("PLY element count is invalid.");
                }

                current = new PlyElement(parts[1], count);
                elements.Add(current);
            }
            else if (string.Equals(parts[0], "property", StringComparison.Ordinal))
            {
                if (current == null)
                {
                    throw new FormatException("PLY property defined before any element.");
                }
                if (parts.Length < 3)
                {
                    throw new FormatException("PLY property line is invalid.");
                }

                if (string.Equals(parts[1], "list", StringComparison.Ordinal))
                {
                    if (parts.Length < 5)
                    {
                        throw new FormatException("PLY list property line is invalid.");
                    }
                    var countType = ParseScalarType(parts[2]);
                    var itemType = ParseScalarType(parts[3]);
                    var name = parts[4].ToLowerInvariant();
                    current.Properties.Add(new PlyProperty(name, PlyScalarType.Int8, true, countType, itemType));
                }
                else
                {
                    var type = ParseScalarType(parts[1]);
                    var name = parts[2].ToLowerInvariant();
                    current.Properties.Add(new PlyProperty(name, type, false, PlyScalarType.Int8, PlyScalarType.Int8));
                }
            }
        }

        return new PlyHeader(format, elements);
    }

    private static PlyFormat ParseFormat(string token)
    {
        return token switch
        {
            "ascii" => PlyFormat.Ascii,
            "binary_little_endian" => PlyFormat.BinaryLittleEndian,
            "binary_big_endian" => PlyFormat.BinaryBigEndian,
            _ => throw new FormatException($"Unsupported PLY format '{token}'.")
        };
    }

    private static PlyScalarType ParseScalarType(string token)
    {
        return token switch
        {
            "char" => PlyScalarType.Int8,
            "uchar" => PlyScalarType.UInt8,
            "short" => PlyScalarType.Int16,
            "ushort" => PlyScalarType.UInt16,
            "int" => PlyScalarType.Int32,
            "uint" => PlyScalarType.UInt32,
            "float" => PlyScalarType.Float32,
            "double" => PlyScalarType.Float64,
            _ => throw new FormatException($"Unsupported PLY scalar type '{token}'.")
        };
    }

    private static PlyElement? FindElement(List<PlyElement> elements, string name)
    {
        foreach (var element in elements)
        {
            if (string.Equals(element.Name, name, StringComparison.Ordinal))
            {
                return element;
            }
        }
        return null;
    }

    private static VertexPropertyFlags AnalyzeVertexProperties(List<PlyProperty> properties)
    {
        bool hasNormals = HasProperties(properties, "nx", "ny", "nz");
        bool hasColorLong = HasProperties(properties, "red", "green", "blue");
        bool hasColorShort = HasProperties(properties, "r", "g", "b");
        bool hasAlphaLong = HasProperty(properties, "alpha");
        bool hasAlphaShort = HasProperty(properties, "a");
        int colorChannels = hasColorLong || hasColorShort ? (hasAlphaLong || hasAlphaShort ? 4 : 3) : 0;

        bool hasUv = HasProperty(properties, "u") && HasProperty(properties, "v");
        bool hasSt = HasProperty(properties, "s") && HasProperty(properties, "t");
        string uName = hasUv ? "u" : "s";
        string vName = hasUv ? "v" : "t";

        return new VertexPropertyFlags(hasNormals, colorChannels, hasUv || hasSt, uName, vName, hasColorLong, hasAlphaLong);
    }

    private static bool HasProperty(List<PlyProperty> properties, string name)
    {
        foreach (var property in properties)
        {
            if (string.Equals(property.Name, name, StringComparison.Ordinal))
            {
                return true;
            }
        }
        return false;
    }

    private static bool HasProperties(List<PlyProperty> properties, string first, string second, string third)
    {
        return HasProperty(properties, first) && HasProperty(properties, second) && HasProperty(properties, third);
    }
    private static void ReadAsciiVertices<T>(
        StreamReader reader,
        PlyElement element,
        VertexPropertyFlags flags,
        INumericOperations<T> numOps,
        List<T> positions,
        List<T> normals,
        List<T> colors,
        List<T> uvs)
    {
        for (int i = 0; i < element.Count; i++)
        {
            var line = reader.ReadLine();
            if (line == null)
            {
                throw new EndOfStreamException("Unexpected end of PLY vertex data.");
            }

            var tokens = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            int tokenIndex = 0;
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            double nx = 0.0;
            double ny = 0.0;
            double nz = 0.0;
            double u = 0.0;
            double v = 0.0;
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            double a = 0.0;

            foreach (var property in element.Properties)
            {
                if (property.IsList)
                {
                    int listCount = ParseScalarAsInt(tokens, ref tokenIndex, property.ListCountType);
                    tokenIndex += listCount;
                    continue;
                }

                double value = ParseScalar(tokens, ref tokenIndex, property.Type);
                switch (property.Name)
                {
                    case "x":
                        x = value;
                        break;
                    case "y":
                        y = value;
                        break;
                    case "z":
                        z = value;
                        break;
                    case "nx":
                        nx = value;
                        break;
                    case "ny":
                        ny = value;
                        break;
                    case "nz":
                        nz = value;
                        break;
                    case "u":
                    case "s":
                        u = value;
                        break;
                    case "v":
                    case "t":
                        v = value;
                        break;
                    case "red":
                    case "r":
                        r = value;
                        break;
                    case "green":
                    case "g":
                        g = value;
                        break;
                    case "blue":
                    case "b":
                        b = value;
                        break;
                    case "alpha":
                    case "a":
                        a = value;
                        break;
                }
            }

            positions.Add(numOps.FromDouble(x));
            positions.Add(numOps.FromDouble(y));
            positions.Add(numOps.FromDouble(z));

            if (flags.HasNormals)
            {
                normals.Add(numOps.FromDouble(nx));
                normals.Add(numOps.FromDouble(ny));
                normals.Add(numOps.FromDouble(nz));
            }

            if (flags.ColorChannels > 0)
            {
                colors.Add(numOps.FromDouble(r));
                colors.Add(numOps.FromDouble(g));
                colors.Add(numOps.FromDouble(b));
                if (flags.ColorChannels == 4)
                {
                    colors.Add(numOps.FromDouble(a));
                }
            }

            if (flags.HasUvs)
            {
                uvs.Add(numOps.FromDouble(u));
                uvs.Add(numOps.FromDouble(v));
            }
        }
    }

    private static void ReadBinaryVertices<T>(
        BinaryReader reader,
        PlyElement element,
        VertexPropertyFlags flags,
        INumericOperations<T> numOps,
        List<T> positions,
        List<T> normals,
        List<T> colors,
        List<T> uvs)
    {
        for (int i = 0; i < element.Count; i++)
        {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            double nx = 0.0;
            double ny = 0.0;
            double nz = 0.0;
            double u = 0.0;
            double v = 0.0;
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            double a = 0.0;

            foreach (var property in element.Properties)
            {
                if (property.IsList)
                {
                    int listCount = ReadScalarAsInt(reader, property.ListCountType);
                    for (int j = 0; j < listCount; j++)
                    {
                        _ = ReadScalar(reader, property.ListItemType);
                    }
                    continue;
                }

                double value = ReadScalar(reader, property.Type);
                switch (property.Name)
                {
                    case "x":
                        x = value;
                        break;
                    case "y":
                        y = value;
                        break;
                    case "z":
                        z = value;
                        break;
                    case "nx":
                        nx = value;
                        break;
                    case "ny":
                        ny = value;
                        break;
                    case "nz":
                        nz = value;
                        break;
                    case "u":
                    case "s":
                        u = value;
                        break;
                    case "v":
                    case "t":
                        v = value;
                        break;
                    case "red":
                    case "r":
                        r = value;
                        break;
                    case "green":
                    case "g":
                        g = value;
                        break;
                    case "blue":
                    case "b":
                        b = value;
                        break;
                    case "alpha":
                    case "a":
                        a = value;
                        break;
                }
            }

            positions.Add(numOps.FromDouble(x));
            positions.Add(numOps.FromDouble(y));
            positions.Add(numOps.FromDouble(z));

            if (flags.HasNormals)
            {
                normals.Add(numOps.FromDouble(nx));
                normals.Add(numOps.FromDouble(ny));
                normals.Add(numOps.FromDouble(nz));
            }

            if (flags.ColorChannels > 0)
            {
                colors.Add(numOps.FromDouble(r));
                colors.Add(numOps.FromDouble(g));
                colors.Add(numOps.FromDouble(b));
                if (flags.ColorChannels == 4)
                {
                    colors.Add(numOps.FromDouble(a));
                }
            }

            if (flags.HasUvs)
            {
                uvs.Add(numOps.FromDouble(u));
                uvs.Add(numOps.FromDouble(v));
            }
        }
    }

    private static void ReadAsciiFaces<T>(
        StreamReader reader,
        PlyElement element,
        List<int> indices,
        List<T> faceNormals,
        INumericOperations<T> numOps)
    {
        var propertyNames = element.Properties;
        var indexProperty = FindFaceIndexProperty(propertyNames);
        bool hasNormals = HasProperties(propertyNames, "nx", "ny", "nz");

        for (int i = 0; i < element.Count; i++)
        {
            var line = reader.ReadLine();
            if (line == null)
            {
                throw new EndOfStreamException("Unexpected end of PLY face data.");
            }

            var tokens = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            int tokenIndex = 0;
            double nx = 0.0;
            double ny = 0.0;
            double nz = 0.0;
            var polygon = new List<int>();

            foreach (var property in element.Properties)
            {
                if (property.IsList)
                {
                    int listCount = ParseScalarAsInt(tokens, ref tokenIndex, property.ListCountType);
                    if (property.Name == indexProperty.Name)
                    {
                        for (int j = 0; j < listCount; j++)
                        {
                            int idx = ParseScalarAsInt(tokens, ref tokenIndex, property.ListItemType);
                            polygon.Add(idx);
                        }
                    }
                    else
                    {
                        tokenIndex += listCount;
                    }
                    continue;
                }

                double value = ParseScalar(tokens, ref tokenIndex, property.Type);
                if (property.Name == "nx")
                {
                    nx = value;
                }
                else if (property.Name == "ny")
                {
                    ny = value;
                }
                else if (property.Name == "nz")
                {
                    nz = value;
                }
            }

            if (polygon.Count >= 3)
            {
                TriangulateFace(polygon, indices, faceNormals, hasNormals, numOps, nx, ny, nz);
            }
        }
    }

    private static void ReadBinaryFaces<T>(
        BinaryReader reader,
        PlyElement element,
        List<int> indices,
        List<T> faceNormals,
        INumericOperations<T> numOps)
    {
        var propertyNames = element.Properties;
        var indexProperty = FindFaceIndexProperty(propertyNames);
        bool hasNormals = HasProperties(propertyNames, "nx", "ny", "nz");

        for (int i = 0; i < element.Count; i++)
        {
            double nx = 0.0;
            double ny = 0.0;
            double nz = 0.0;
            var polygon = new List<int>();

            foreach (var property in element.Properties)
            {
                if (property.IsList)
                {
                    int listCount = ReadScalarAsInt(reader, property.ListCountType);
                    if (property.Name == indexProperty.Name)
                    {
                        for (int j = 0; j < listCount; j++)
                        {
                            int idx = ReadScalarAsInt(reader, property.ListItemType);
                            polygon.Add(idx);
                        }
                    }
                    else
                    {
                        for (int j = 0; j < listCount; j++)
                        {
                            _ = ReadScalar(reader, property.ListItemType);
                        }
                    }
                    continue;
                }

                double value = ReadScalar(reader, property.Type);
                if (property.Name == "nx")
                {
                    nx = value;
                }
                else if (property.Name == "ny")
                {
                    ny = value;
                }
                else if (property.Name == "nz")
                {
                    nz = value;
                }
            }

            if (polygon.Count >= 3)
            {
                TriangulateFace(polygon, indices, faceNormals, hasNormals, numOps, nx, ny, nz);
            }
        }
    }

    private static void SkipAsciiElement(StreamReader reader, int count)
    {
        for (int i = 0; i < count; i++)
        {
            if (reader.ReadLine() == null)
            {
                throw new EndOfStreamException("Unexpected end of PLY data.");
            }
        }
    }

    private static void SkipBinaryElement(BinaryReader reader, PlyElement element)
    {
        for (int i = 0; i < element.Count; i++)
        {
            foreach (var property in element.Properties)
            {
                if (property.IsList)
                {
                    int listCount = ReadScalarAsInt(reader, property.ListCountType);
                    for (int j = 0; j < listCount; j++)
                    {
                        _ = ReadScalar(reader, property.ListItemType);
                    }
                }
                else
                {
                    _ = ReadScalar(reader, property.Type);
                }
            }
        }
    }

    private static PlyProperty FindFaceIndexProperty(List<PlyProperty> properties)
    {
        foreach (var property in properties)
        {
            if (property.IsList && (property.Name == "vertex_indices" || property.Name == "vertex_index"))
            {
                return property;
            }
        }

        foreach (var property in properties)
        {
            if (property.IsList)
            {
                return property;
            }
        }

        throw new FormatException("PLY face element is missing a vertex index list property.");
    }

    private static double ParseScalar(string[] tokens, ref int index, PlyScalarType type)
    {
        if (index >= tokens.Length)
        {
            throw new FormatException("PLY data line has fewer values than expected.");
        }

        string token = tokens[index++];
        if (type == PlyScalarType.Float32 || type == PlyScalarType.Float64)
        {
            if (!double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
            {
                throw new FormatException($"Invalid PLY float value '{token}'.");
            }
            return value;
        }

        if (!long.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out var intValue))
        {
            throw new FormatException($"Invalid PLY integer value '{token}'.");
        }
        return intValue;
    }

    private static int ParseScalarAsInt(string[] tokens, ref int index, PlyScalarType type)
    {
        double value = ParseScalar(tokens, ref index, type);
        return Convert.ToInt32(value, CultureInfo.InvariantCulture);
    }

    private static double ReadScalar(BinaryReader reader, PlyScalarType type)
    {
        return type switch
        {
            PlyScalarType.Int8 => reader.ReadSByte(),
            PlyScalarType.UInt8 => reader.ReadByte(),
            PlyScalarType.Int16 => reader.ReadInt16(),
            PlyScalarType.UInt16 => reader.ReadUInt16(),
            PlyScalarType.Int32 => reader.ReadInt32(),
            PlyScalarType.UInt32 => reader.ReadUInt32(),
            PlyScalarType.Float32 => reader.ReadSingle(),
            PlyScalarType.Float64 => reader.ReadDouble(),
            _ => throw new FormatException("Unsupported PLY scalar type.")
        };
    }

    private static int ReadScalarAsInt(BinaryReader reader, PlyScalarType type)
    {
        return type switch
        {
            PlyScalarType.Int8 => reader.ReadSByte(),
            PlyScalarType.UInt8 => reader.ReadByte(),
            PlyScalarType.Int16 => reader.ReadInt16(),
            PlyScalarType.UInt16 => reader.ReadUInt16(),
            PlyScalarType.Int32 => reader.ReadInt32(),
            PlyScalarType.UInt32 => checked((int)reader.ReadUInt32()),
            _ => throw new FormatException("PLY list count/index must be an integer type.")
        };
    }

    private static void TriangulateFace<T>(
        List<int> polygon,
        List<int> indices,
        List<T> faceNormals,
        bool hasNormals,
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

            if (hasNormals)
            {
                faceNormals.Add(numOps.FromDouble(nx));
                faceNormals.Add(numOps.FromDouble(ny));
                faceNormals.Add(numOps.FromDouble(nz));
            }
        }
    }
    private static void WriteMeshHeader<T>(
        TriangleMeshData<T> mesh,
        Stream stream,
        bool binary,
        bool includeNormals,
        bool includeUvs,
        int colorChannels,
        bool includeFaceNormals)
    {
        using var writer = new StreamWriter(stream, Encoding.ASCII, 1024, leaveOpen: true);
        writer.WriteLine("ply");
        writer.WriteLine(binary ? "format binary_little_endian 1.0" : "format ascii 1.0");
        writer.WriteLine($"element vertex {mesh.NumVertices}");
        writer.WriteLine("property float x");
        writer.WriteLine("property float y");
        writer.WriteLine("property float z");

        if (includeNormals)
        {
            writer.WriteLine("property float nx");
            writer.WriteLine("property float ny");
            writer.WriteLine("property float nz");
        }

        if (colorChannels > 0)
        {
            writer.WriteLine("property uchar red");
            writer.WriteLine("property uchar green");
            writer.WriteLine("property uchar blue");
            if (colorChannels == 4)
            {
                writer.WriteLine("property uchar alpha");
            }
        }

        if (includeUvs)
        {
            writer.WriteLine("property float u");
            writer.WriteLine("property float v");
        }

        writer.WriteLine($"element face {mesh.NumFaces}");
        writer.WriteLine("property list uchar int vertex_indices");
        if (includeFaceNormals)
        {
            writer.WriteLine("property float nx");
            writer.WriteLine("property float ny");
            writer.WriteLine("property float nz");
        }
        writer.WriteLine("end_header");
        writer.Flush();
    }

    private static void WritePointCloudHeader<T>(
        PointCloudData<T> pointCloud,
        Stream stream,
        bool binary,
        bool includeNormals,
        bool includeUvs,
        int colorChannels)
    {
        using var writer = new StreamWriter(stream, Encoding.ASCII, 1024, leaveOpen: true);
        writer.WriteLine("ply");
        writer.WriteLine(binary ? "format binary_little_endian 1.0" : "format ascii 1.0");
        writer.WriteLine($"element vertex {pointCloud.NumPoints}");
        writer.WriteLine("property float x");
        writer.WriteLine("property float y");
        writer.WriteLine("property float z");

        if (colorChannels > 0)
        {
            writer.WriteLine("property uchar red");
            writer.WriteLine("property uchar green");
            writer.WriteLine("property uchar blue");
            if (colorChannels == 4)
            {
                writer.WriteLine("property uchar alpha");
            }
        }

        if (includeNormals)
        {
            writer.WriteLine("property float nx");
            writer.WriteLine("property float ny");
            writer.WriteLine("property float nz");
        }

        if (includeUvs)
        {
            writer.WriteLine("property float u");
            writer.WriteLine("property float v");
        }

        writer.WriteLine("end_header");
        writer.Flush();
    }

    private static void WriteMeshAscii<T>(
        TriangleMeshData<T> mesh,
        StreamWriter writer,
        INumericOperations<T> numOps,
        bool includeNormals,
        bool includeUvs,
        int colorChannels)
    {
        var vertices = mesh.Vertices.Data.Span;
        var normals = includeNormals && mesh.VertexNormals != null ? mesh.VertexNormals.Data.Span : default;
        var uvs = includeUvs && mesh.VertexUVs != null ? mesh.VertexUVs.Data.Span : default;
        var colors = colorChannels > 0 && mesh.VertexColors != null ? mesh.VertexColors.Data.Span : default;

        for (int i = 0; i < mesh.NumVertices; i++)
        {
            int offset = i * 3;
            writer.Write(string.Format(CultureInfo.InvariantCulture, "{0} {1} {2}",
                numOps.ToDouble(vertices[offset]),
                numOps.ToDouble(vertices[offset + 1]),
                numOps.ToDouble(vertices[offset + 2])));

            if (includeNormals && normals.Length > 0)
            {
                int normalOffset = i * 3;
                writer.Write(string.Format(CultureInfo.InvariantCulture, " {0} {1} {2}",
                    numOps.ToDouble(normals[normalOffset]),
                    numOps.ToDouble(normals[normalOffset + 1]),
                    numOps.ToDouble(normals[normalOffset + 2])));
            }

            if (colorChannels > 0 && colors.Length > 0)
            {
                int colorOffset = i * colorChannels;
                for (int c = 0; c < colorChannels; c++)
                {
                    writer.Write(string.Format(CultureInfo.InvariantCulture, " {0}",
                        ConvertColorToByte(numOps.ToDouble(colors[colorOffset + c]))));
                }
            }

            if (includeUvs && uvs.Length > 0)
            {
                int uvOffset = i * 2;
                writer.Write(string.Format(CultureInfo.InvariantCulture, " {0} {1}",
                    numOps.ToDouble(uvs[uvOffset]),
                    numOps.ToDouble(uvs[uvOffset + 1])));
            }

            writer.WriteLine();
        }

        WriteFacesAscii(mesh, writer, numOps);
    }

    private static void WriteMeshBinary<T>(
        TriangleMeshData<T> mesh,
        BinaryWriter writer,
        INumericOperations<T> numOps,
        bool includeNormals,
        bool includeUvs,
        int colorChannels)
    {
        var vertices = mesh.Vertices.Data.Span;
        var normals = includeNormals && mesh.VertexNormals != null ? mesh.VertexNormals.Data.Span : default;
        var uvs = includeUvs && mesh.VertexUVs != null ? mesh.VertexUVs.Data.Span : default;
        var colors = colorChannels > 0 && mesh.VertexColors != null ? mesh.VertexColors.Data.Span : default;

        for (int i = 0; i < mesh.NumVertices; i++)
        {
            int offset = i * 3;
            writer.Write((float)numOps.ToDouble(vertices[offset]));
            writer.Write((float)numOps.ToDouble(vertices[offset + 1]));
            writer.Write((float)numOps.ToDouble(vertices[offset + 2]));

            if (includeNormals && normals.Length > 0)
            {
                int normalOffset = i * 3;
                writer.Write((float)numOps.ToDouble(normals[normalOffset]));
                writer.Write((float)numOps.ToDouble(normals[normalOffset + 1]));
                writer.Write((float)numOps.ToDouble(normals[normalOffset + 2]));
            }

            if (colorChannels > 0 && colors.Length > 0)
            {
                int colorOffset = i * colorChannels;
                for (int c = 0; c < colorChannels; c++)
                {
                    writer.Write(ConvertColorToByte(numOps.ToDouble(colors[colorOffset + c])));
                }
            }

            if (includeUvs && uvs.Length > 0)
            {
                int uvOffset = i * 2;
                writer.Write((float)numOps.ToDouble(uvs[uvOffset]));
                writer.Write((float)numOps.ToDouble(uvs[uvOffset + 1]));
            }
        }

        WriteFacesBinary(mesh, writer, numOps);
    }

    private static void WriteFacesAscii<T>(TriangleMeshData<T> mesh, StreamWriter writer, INumericOperations<T> numOps)
    {
        var faces = mesh.Faces.Data.Span;
        var faceNormals = mesh.FaceNormals;
        for (int i = 0; i < mesh.NumFaces; i++)
        {
            int faceOffset = i * 3;
            int i0 = faces[faceOffset];
            int i1 = faces[faceOffset + 1];
            int i2 = faces[faceOffset + 2];
            writer.Write($"3 {i0} {i1} {i2}");

            if (faceNormals != null)
            {
                int normalOffset = i * 3;
                writer.Write(string.Format(CultureInfo.InvariantCulture, " {0} {1} {2}",
                    numOps.ToDouble(faceNormals.Data.Span[normalOffset]),
                    numOps.ToDouble(faceNormals.Data.Span[normalOffset + 1]),
                    numOps.ToDouble(faceNormals.Data.Span[normalOffset + 2])));
            }
            writer.WriteLine();
        }
    }

    private static void WriteFacesBinary<T>(TriangleMeshData<T> mesh, BinaryWriter writer, INumericOperations<T> numOps)
    {
        var faces = mesh.Faces.Data.Span;
        var faceNormals = mesh.FaceNormals;
        for (int i = 0; i < mesh.NumFaces; i++)
        {
            int faceOffset = i * 3;
            writer.Write((byte)3);
            writer.Write(faces[faceOffset]);
            writer.Write(faces[faceOffset + 1]);
            writer.Write(faces[faceOffset + 2]);

            if (faceNormals != null)
            {
                int normalOffset = i * 3;
                writer.Write((float)numOps.ToDouble(faceNormals.Data.Span[normalOffset]));
                writer.Write((float)numOps.ToDouble(faceNormals.Data.Span[normalOffset + 1]));
                writer.Write((float)numOps.ToDouble(faceNormals.Data.Span[normalOffset + 2]));
            }
        }
    }

    private static void WritePointCloudAscii<T>(
        PointCloudData<T> pointCloud,
        StreamWriter writer,
        INumericOperations<T> numOps,
        bool includeNormals,
        bool includeUvs,
        int colorChannels,
        int colorOffset,
        int normalOffset,
        int uvOffset)
    {
        int featureDim = pointCloud.NumFeatures;
        var data = pointCloud.Points.Data.Span;
        for (int i = 0; i < pointCloud.NumPoints; i++)
        {
            int offset = i * featureDim;
            writer.Write(string.Format(CultureInfo.InvariantCulture, "{0} {1} {2}",
                numOps.ToDouble(data[offset]),
                numOps.ToDouble(data[offset + 1]),
                numOps.ToDouble(data[offset + 2])));

            if (colorChannels > 0 && colorOffset >= 0)
            {
                for (int c = 0; c < colorChannels; c++)
                {
                    writer.Write(string.Format(CultureInfo.InvariantCulture, " {0}",
                        ConvertColorToByte(numOps.ToDouble(data[offset + colorOffset + c]))));
                }
            }

            if (includeNormals && normalOffset >= 0)
            {
                writer.Write(string.Format(CultureInfo.InvariantCulture, " {0} {1} {2}",
                    numOps.ToDouble(data[offset + normalOffset]),
                    numOps.ToDouble(data[offset + normalOffset + 1]),
                    numOps.ToDouble(data[offset + normalOffset + 2])));
            }

            if (includeUvs && uvOffset >= 0)
            {
                writer.Write(string.Format(CultureInfo.InvariantCulture, " {0} {1}",
                    numOps.ToDouble(data[offset + uvOffset]),
                    numOps.ToDouble(data[offset + uvOffset + 1])));
            }

            writer.WriteLine();
        }
    }

    private static void WritePointCloudBinary<T>(
        PointCloudData<T> pointCloud,
        BinaryWriter writer,
        INumericOperations<T> numOps,
        bool includeNormals,
        bool includeUvs,
        int colorChannels,
        int colorOffset,
        int normalOffset,
        int uvOffset)
    {
        int featureDim = pointCloud.NumFeatures;
        var data = pointCloud.Points.Data.Span;
        for (int i = 0; i < pointCloud.NumPoints; i++)
        {
            int offset = i * featureDim;
            writer.Write((float)numOps.ToDouble(data[offset]));
            writer.Write((float)numOps.ToDouble(data[offset + 1]));
            writer.Write((float)numOps.ToDouble(data[offset + 2]));

            if (colorChannels > 0 && colorOffset >= 0)
            {
                for (int c = 0; c < colorChannels; c++)
                {
                    writer.Write(ConvertColorToByte(numOps.ToDouble(data[offset + colorOffset + c])));
                }
            }

            if (includeNormals && normalOffset >= 0)
            {
                writer.Write((float)numOps.ToDouble(data[offset + normalOffset]));
                writer.Write((float)numOps.ToDouble(data[offset + normalOffset + 1]));
                writer.Write((float)numOps.ToDouble(data[offset + normalOffset + 2]));
            }

            if (includeUvs && uvOffset >= 0)
            {
                writer.Write((float)numOps.ToDouble(data[offset + uvOffset]));
                writer.Write((float)numOps.ToDouble(data[offset + uvOffset + 1]));
            }
        }
    }

    private static byte ConvertColorToByte(double value)
    {
        double scaled = value <= 1.0 ? value * 255.0 : value;
        if (scaled < 0.0)
        {
            scaled = 0.0;
        }
        if (scaled > 255.0)
        {
            scaled = 255.0;
        }
        return (byte)Math.Round(scaled);
    }

    private sealed class PlyHeader
    {
        public PlyHeader(PlyFormat format, List<PlyElement> elements)
        {
            Format = format;
            Elements = elements;
        }

        public PlyFormat Format { get; }
        public List<PlyElement> Elements { get; }
    }

    private sealed class PlyElement
    {
        public PlyElement(string name, int count)
        {
            Name = name;
            Count = count;
            Properties = new List<PlyProperty>();
        }

        public string Name { get; }
        public int Count { get; }
        public List<PlyProperty> Properties { get; }
    }

    private sealed class PlyProperty
    {
        public PlyProperty(string name, PlyScalarType type, bool isList, PlyScalarType listCountType, PlyScalarType listItemType)
        {
            Name = name;
            Type = type;
            IsList = isList;
            ListCountType = listCountType;
            ListItemType = listItemType;
        }

        public string Name { get; }
        public PlyScalarType Type { get; }
        public bool IsList { get; }
        public PlyScalarType ListCountType { get; }
        public PlyScalarType ListItemType { get; }
    }

    private readonly struct VertexPropertyFlags
    {
        public VertexPropertyFlags(
            bool hasNormals,
            int colorChannels,
            bool hasUvs,
            string uName,
            string vName,
            bool usesLongColorNames,
            bool usesLongAlphaName)
        {
            HasNormals = hasNormals;
            ColorChannels = colorChannels;
            HasUvs = hasUvs;
            UName = uName;
            VName = vName;
            UsesLongColorNames = usesLongColorNames;
            UsesLongAlphaName = usesLongAlphaName;
        }

        public bool HasNormals { get; }
        public int ColorChannels { get; }
        public bool HasUvs { get; }
        public string UName { get; }
        public string VName { get; }
        public bool UsesLongColorNames { get; }
        public bool UsesLongAlphaName { get; }
    }

    private enum PlyFormat
    {
        Ascii,
        BinaryLittleEndian,
        BinaryBigEndian
    }

    private enum PlyScalarType
    {
        Int8,
        UInt8,
        Int16,
        UInt16,
        Int32,
        UInt32,
        Float32,
        Float64
    }
}
