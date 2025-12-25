using System;
using System.Collections.Generic;
using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Geometry.Data;

/// <summary>
/// Represents a triangle mesh with vertex and face attributes.
/// </summary>
/// <typeparam name="T">The numeric type used for geometry values.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A triangle mesh is a surface defined by vertices
/// (points in 3D space) and faces (triangles that connect three vertices).
/// Meshes are a common way to represent 3D objects in graphics and AI.
/// </remarks>
public sealed class TriangleMeshData<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Vertex positions of shape [numVertices, 3].
    /// </summary>
    public Tensor<T> Vertices { get; }

    /// <summary>
    /// Triangle indices of shape [numFaces, 3].
    /// </summary>
    public Tensor<int> Faces { get; }

    /// <summary>
    /// Optional per-vertex normals of shape [numVertices, 3].
    /// </summary>
    public Tensor<T>? VertexNormals { get; private set; }

    /// <summary>
    /// Optional per-face normals of shape [numFaces, 3].
    /// </summary>
    public Tensor<T>? FaceNormals { get; private set; }

    /// <summary>
    /// Optional per-vertex colors of shape [numVertices, 3] or [numVertices, 4].
    /// </summary>
    public Tensor<T>? VertexColors { get; }

    /// <summary>
    /// Optional per-vertex UV coordinates of shape [numVertices, 2].
    /// </summary>
    public Tensor<T>? VertexUVs { get; }

    /// <summary>
    /// Optional metadata associated with the mesh (units, source, labels, etc.).
    /// </summary>
    public Dictionary<string, object>? Metadata { get; set; }

    /// <summary>
    /// Number of vertices in the mesh.
    /// </summary>
    public int NumVertices => Vertices.Shape[0];

    /// <summary>
    /// Number of faces in the mesh.
    /// </summary>
    public int NumFaces => Faces.Shape[0];

    /// <summary>
    /// Initializes a new instance of the TriangleMeshData class.
    /// </summary>
    public TriangleMeshData(
        Tensor<T> vertices,
        Tensor<int> faces,
        Tensor<T>? vertexNormals = null,
        Tensor<T>? vertexColors = null,
        Tensor<T>? vertexUVs = null,
        Tensor<T>? faceNormals = null,
        bool validateIndices = true)
    {
        ValidateVertices(vertices);
        ValidateFaces(faces);
        ValidateVertexNormals(vertices, vertexNormals);
        ValidateFaceNormals(faces, faceNormals);
        ValidateVertexColors(vertices, vertexColors);
        ValidateVertexUVs(vertices, vertexUVs);

        Vertices = vertices;
        Faces = faces;
        VertexNormals = vertexNormals;
        VertexColors = vertexColors;
        VertexUVs = vertexUVs;
        FaceNormals = faceNormals;

        if (NumVertices == 0 && NumFaces > 0)
        {
            throw new ArgumentException("Faces cannot be specified when there are no vertices.", nameof(faces));
        }

        if (validateIndices)
        {
            ValidateFaceIndices(faces, NumVertices);
        }
    }

    /// <summary>
    /// Computes the axis-aligned bounding box for the mesh.
    /// </summary>
    public (Vector<T> min, Vector<T> max) ComputeBounds()
    {
        if (NumVertices == 0)
        {
            var zero = NumOps.Zero;
            return (new Vector<T>(new[] { zero, zero, zero }), new Vector<T>(new[] { zero, zero, zero }));
        }

        var minX = NumOps.MaxValue;
        var minY = NumOps.MaxValue;
        var minZ = NumOps.MaxValue;
        var maxX = NumOps.MinValue;
        var maxY = NumOps.MinValue;
        var maxZ = NumOps.MinValue;

        var data = Vertices.Data;
        for (int i = 0; i < NumVertices; i++)
        {
            int offset = i * 3;
            var x = data[offset];
            var y = data[offset + 1];
            var z = data[offset + 2];

            if (NumOps.LessThan(x, minX)) minX = x;
            if (NumOps.LessThan(y, minY)) minY = y;
            if (NumOps.LessThan(z, minZ)) minZ = z;
            if (NumOps.GreaterThan(x, maxX)) maxX = x;
            if (NumOps.GreaterThan(y, maxY)) maxY = y;
            if (NumOps.GreaterThan(z, maxZ)) maxZ = z;
        }

        return (new Vector<T>(new[] { minX, minY, minZ }), new Vector<T>(new[] { maxX, maxY, maxZ }));
    }

    /// <summary>
    /// Computes and stores per-face normals.
    /// </summary>
    public Tensor<T> ComputeFaceNormals()
    {
        if (NumFaces == 0)
        {
            FaceNormals = new Tensor<T>(new T[0], new[] { 0, 3 });
            return FaceNormals;
        }

        var normals = new T[NumFaces * 3];
        var vertices = Vertices.Data;
        var faces = Faces.Data;

        for (int f = 0; f < NumFaces; f++)
        {
            int faceOffset = f * 3;
            int i0 = faces[faceOffset];
            int i1 = faces[faceOffset + 1];
            int i2 = faces[faceOffset + 2];

            int v0 = i0 * 3;
            int v1 = i1 * 3;
            int v2 = i2 * 3;

            var v0x = vertices[v0];
            var v0y = vertices[v0 + 1];
            var v0z = vertices[v0 + 2];
            var v1x = vertices[v1];
            var v1y = vertices[v1 + 1];
            var v1z = vertices[v1 + 2];
            var v2x = vertices[v2];
            var v2y = vertices[v2 + 1];
            var v2z = vertices[v2 + 2];

            var e1x = NumOps.Subtract(v1x, v0x);
            var e1y = NumOps.Subtract(v1y, v0y);
            var e1z = NumOps.Subtract(v1z, v0z);
            var e2x = NumOps.Subtract(v2x, v0x);
            var e2y = NumOps.Subtract(v2y, v0y);
            var e2z = NumOps.Subtract(v2z, v0z);

            var nx = NumOps.Subtract(NumOps.Multiply(e1y, e2z), NumOps.Multiply(e1z, e2y));
            var ny = NumOps.Subtract(NumOps.Multiply(e1z, e2x), NumOps.Multiply(e1x, e2z));
            var nz = NumOps.Subtract(NumOps.Multiply(e1x, e2y), NumOps.Multiply(e1y, e2x));

            var lenSq = NumOps.Add(
                NumOps.Add(NumOps.Multiply(nx, nx), NumOps.Multiply(ny, ny)),
                NumOps.Multiply(nz, nz));

            if (NumOps.GreaterThan(lenSq, NumOps.Zero))
            {
                var invLen = NumOps.Divide(NumOps.One, NumOps.Sqrt(lenSq));
                nx = NumOps.Multiply(nx, invLen);
                ny = NumOps.Multiply(ny, invLen);
                nz = NumOps.Multiply(nz, invLen);
            }
            else
            {
                nx = NumOps.Zero;
                ny = NumOps.Zero;
                nz = NumOps.Zero;
            }

            normals[faceOffset] = nx;
            normals[faceOffset + 1] = ny;
            normals[faceOffset + 2] = nz;
        }

        FaceNormals = new Tensor<T>(normals, new[] { NumFaces, 3 });
        return FaceNormals;
    }

    /// <summary>
    /// Computes and stores per-vertex normals by averaging adjacent face normals.
    /// </summary>
    public Tensor<T> ComputeVertexNormals()
    {
        if (NumVertices == 0)
        {
            VertexNormals = new Tensor<T>(new T[0], new[] { 0, 3 });
            return VertexNormals;
        }

        var faceNormals = FaceNormals ?? ComputeFaceNormals();
        var normals = new T[NumVertices * 3];
        var faces = Faces.Data;

        for (int f = 0; f < NumFaces; f++)
        {
            int faceOffset = f * 3;
            var nx = faceNormals.Data[faceOffset];
            var ny = faceNormals.Data[faceOffset + 1];
            var nz = faceNormals.Data[faceOffset + 2];

            int i0 = faces[faceOffset];
            int i1 = faces[faceOffset + 1];
            int i2 = faces[faceOffset + 2];

            AddNormal(normals, i0, nx, ny, nz);
            AddNormal(normals, i1, nx, ny, nz);
            AddNormal(normals, i2, nx, ny, nz);
        }

        for (int v = 0; v < NumVertices; v++)
        {
            int offset = v * 3;
            var nx = normals[offset];
            var ny = normals[offset + 1];
            var nz = normals[offset + 2];

            var lenSq = NumOps.Add(
                NumOps.Add(NumOps.Multiply(nx, nx), NumOps.Multiply(ny, ny)),
                NumOps.Multiply(nz, nz));

            if (NumOps.GreaterThan(lenSq, NumOps.Zero))
            {
                var invLen = NumOps.Divide(NumOps.One, NumOps.Sqrt(lenSq));
                normals[offset] = NumOps.Multiply(nx, invLen);
                normals[offset + 1] = NumOps.Multiply(ny, invLen);
                normals[offset + 2] = NumOps.Multiply(nz, invLen);
            }
            else
            {
                normals[offset] = NumOps.Zero;
                normals[offset + 1] = NumOps.Zero;
                normals[offset + 2] = NumOps.Zero;
            }
        }

        VertexNormals = new Tensor<T>(normals, new[] { NumVertices, 3 });
        return VertexNormals;
    }

    /// <summary>
    /// Converts the mesh to a point cloud using vertex positions and optional attributes.
    /// </summary>
    public PointCloudData<T> ToPointCloud(
        bool includeColors = false,
        bool includeNormals = false,
        bool includeUVs = false)
    {
        if (!includeColors && !includeNormals && !includeUVs)
        {
            return new PointCloudData<T>(Vertices);
        }

        var normals = VertexNormals;
        if (includeNormals && normals == null)
        {
            normals = ComputeVertexNormals();
        }

        var colorsTensor = VertexColors;
        if (includeColors && colorsTensor == null)
        {
            throw new InvalidOperationException("VertexColors are required to include color features.");
        }
        if (includeUVs && VertexUVs == null)
        {
            throw new InvalidOperationException("VertexUVs are required to include UV features.");
        }

        int colorDim = includeColors && colorsTensor != null ? colorsTensor.Shape[1] : 0;
        int normalDim = includeNormals ? 3 : 0;
        int uvDim = includeUVs ? 2 : 0;
        int featureDim = colorDim + normalDim + uvDim;

        var data = new T[NumVertices * (3 + featureDim)];
        var vertices = Vertices.Data;
        var colors = colorsTensor?.Data;
        var uv = VertexUVs?.Data;
        var normalData = normals?.Data;

        for (int v = 0; v < NumVertices; v++)
        {
            int srcOffset = v * 3;
            int dstOffset = v * (3 + featureDim);

            data[dstOffset] = vertices[srcOffset];
            data[dstOffset + 1] = vertices[srcOffset + 1];
            data[dstOffset + 2] = vertices[srcOffset + 2];

            int featureOffset = dstOffset + 3;
            if (includeColors && colors != null)
            {
                int colorOffset = v * colorDim;
                for (int c = 0; c < colorDim; c++)
                {
                    data[featureOffset + c] = colors[colorOffset + c];
                }
                featureOffset += colorDim;
            }

            if (includeNormals && normalData != null)
            {
                int normalOffset = v * 3;
                data[featureOffset] = normalData[normalOffset];
                data[featureOffset + 1] = normalData[normalOffset + 1];
                data[featureOffset + 2] = normalData[normalOffset + 2];
                featureOffset += 3;
            }

            if (includeUVs && uv != null)
            {
                int uvOffset = v * 2;
                data[featureOffset] = uv[uvOffset];
                data[featureOffset + 1] = uv[uvOffset + 1];
            }
        }

        var tensor = new Tensor<T>(data, new[] { NumVertices, 3 + featureDim });
        return new PointCloudData<T>(tensor);
    }

    private static void AddNormal(T[] normals, int vertexIndex, T nx, T ny, T nz)
    {
        int offset = vertexIndex * 3;
        normals[offset] = NumOps.Add(normals[offset], nx);
        normals[offset + 1] = NumOps.Add(normals[offset + 1], ny);
        normals[offset + 2] = NumOps.Add(normals[offset + 2], nz);
    }

    private static void ValidateVertices(Tensor<T> vertices)
    {
        if (vertices == null)
        {
            throw new ArgumentNullException(nameof(vertices));
        }
        if (vertices.Shape.Length != 2 || vertices.Shape[1] != 3)
        {
            throw new ArgumentException("Vertices must have shape [numVertices, 3].", nameof(vertices));
        }
    }

    private static void ValidateFaces(Tensor<int> faces)
    {
        if (faces == null)
        {
            throw new ArgumentNullException(nameof(faces));
        }
        if (faces.Shape.Length != 2 || faces.Shape[1] != 3)
        {
            throw new ArgumentException("Faces must have shape [numFaces, 3].", nameof(faces));
        }
    }

    private static void ValidateFaceIndices(Tensor<int> faces, int numVertices)
    {
        var data = faces.Data;
        for (int i = 0; i < data.Length; i++)
        {
            int index = data[i];
            if (index < 0 || index >= numVertices)
            {
                throw new ArgumentOutOfRangeException(nameof(faces), $"Face index {index} is out of range for {numVertices} vertices.");
            }
        }
    }

    private static void ValidateVertexNormals(Tensor<T> vertices, Tensor<T>? vertexNormals)
    {
        if (vertexNormals == null)
        {
            return;
        }
        if (vertexNormals.Shape.Length != 2 || vertexNormals.Shape[0] != vertices.Shape[0] || vertexNormals.Shape[1] != 3)
        {
            throw new ArgumentException("VertexNormals must have shape [numVertices, 3].", nameof(vertexNormals));
        }
    }

    private static void ValidateFaceNormals(Tensor<int> faces, Tensor<T>? faceNormals)
    {
        if (faceNormals == null)
        {
            return;
        }
        if (faceNormals.Shape.Length != 2 || faceNormals.Shape[0] != faces.Shape[0] || faceNormals.Shape[1] != 3)
        {
            throw new ArgumentException("FaceNormals must have shape [numFaces, 3].", nameof(faceNormals));
        }
    }

    private static void ValidateVertexColors(Tensor<T> vertices, Tensor<T>? vertexColors)
    {
        if (vertexColors == null)
        {
            return;
        }
        if (vertexColors.Shape.Length != 2 || vertexColors.Shape[0] != vertices.Shape[0])
        {
            throw new ArgumentException("VertexColors must have shape [numVertices, 3] or [numVertices, 4].", nameof(vertexColors));
        }
        if (vertexColors.Shape[1] != 3 && vertexColors.Shape[1] != 4)
        {
            throw new ArgumentException("VertexColors must have 3 (RGB) or 4 (RGBA) channels.", nameof(vertexColors));
        }
    }

    private static void ValidateVertexUVs(Tensor<T> vertices, Tensor<T>? vertexUVs)
    {
        if (vertexUVs == null)
        {
            return;
        }
        if (vertexUVs.Shape.Length != 2 || vertexUVs.Shape[0] != vertices.Shape[0] || vertexUVs.Shape[1] != 2)
        {
            throw new ArgumentException("VertexUVs must have shape [numVertices, 2].", nameof(vertexUVs));
        }
    }
}
