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

        var (min, max) = GeometryEngineOps<T>.ColumnBounds(Vertices);
        return (min.ToVector(), max.ToVector());
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

        var crossProducts = GeometryEngineOps<T>.FaceCrossProducts(Vertices, Faces);
        FaceNormals = GeometryEngineOps<T>.NormalizeRowsOrZero(crossProducts);
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
        var sums = GeometryEngineOps<T>.AccumulateToVertices(faceNormals, Faces, NumVertices);
        VertexNormals = GeometryEngineOps<T>.NormalizeRowsOrZero(sums);
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

        var parts = new List<Tensor<T>> { Vertices };
        if (includeColors && colorsTensor != null)
        {
            parts.Add(colorsTensor);
        }

        if (includeNormals && normals != null)
        {
            parts.Add(normals);
        }

        if (includeUVs && VertexUVs != null)
        {
            parts.Add(VertexUVs);
        }

        var tensor = AiDotNetEngine.Current.TensorConcatenate(parts.ToArray(), 1);
        return new PointCloudData<T>(tensor);
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
        var data = faces.Data.Span;
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
