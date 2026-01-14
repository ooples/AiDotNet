using System;
using System.Globalization;
using System.IO;
using System.Text;
using AiDotNet.Geometry.Data;
using AiDotNet.Geometry.IO;
using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Geometry;

public class GeometryIOIntegrationTests
{
    [Fact]
    public void ObjMeshIO_RoundTripsMeshWithAttributes()
    {
        string tempDir = CreateTempDirectory();
        try
        {
            string path = Path.Combine(tempDir, "mesh.obj");
            var vertices = new Tensor<double>(new[]
            {
                0.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0
            }, new[] { 3, 3 });
            var faces = new Tensor<int>(new[] { 0, 1, 2 }, new[] { 1, 3 });
            var normals = new Tensor<double>(new[]
            {
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0
            }, new[] { 3, 3 });
            var uvs = new Tensor<double>(new[]
            {
                0.0, 0.0,
                1.0, 0.0,
                0.0, 1.0
            }, new[] { 3, 2 });
            var colors = new Tensor<double>(new[]
            {
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            }, new[] { 3, 3 });

            var mesh = new TriangleMeshData<double>(vertices, faces, normals, colors, uvs);
            ObjMeshIO.SaveMesh(mesh, path, includeNormals: true, includeUvs: true, includeColors: true);

            var loaded = ObjMeshIO.LoadMesh<double>(path, computeNormalsIfMissing: false);

            Assert.Equal(3, loaded.NumVertices);
            Assert.Equal(1, loaded.NumFaces);
            Assert.NotNull(loaded.VertexNormals);
            Assert.NotNull(loaded.VertexUVs);
            Assert.NotNull(loaded.VertexColors);

            Assert.Equal(0.0, loaded.Vertices.Data.Span[0], 6);
            Assert.Equal(1.0, loaded.Vertices.Data.Span[3], 6);
            Assert.Equal(1.0, loaded.VertexNormals.Data.Span[2], 6);
            Assert.Equal(1.0, loaded.VertexUVs.Data.Span[2], 6);
            Assert.Equal(1.0, loaded.VertexColors.Data.Span[0], 6);
        }
        finally
        {
            CleanupTempDirectory(tempDir);
        }
    }

    [Fact]
    public void ObjMeshIO_RoundTripsPointCloudWithColors()
    {
        string tempDir = CreateTempDirectory();
        try
        {
            string path = Path.Combine(tempDir, "cloud.obj");
            var data = new Tensor<double>(new[]
            {
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.0, 1.0
            }, new[] { 3, 6 });
            var cloud = new PointCloudData<double>(data);

            ObjMeshIO.SavePointCloud(cloud, path, includeColors: true);
            var loaded = ObjMeshIO.LoadPointCloud<double>(path);

            Assert.Equal(3, loaded.NumPoints);
            Assert.Equal(6, loaded.NumFeatures);
            Assert.Equal(1.0, loaded.Points.Data.Span[3], 6);
            Assert.Equal(1.0, loaded.Points.Data.Span[10], 6);
            Assert.Equal(1.0, loaded.Points.Data.Span[17], 6);
        }
        finally
        {
            CleanupTempDirectory(tempDir);
        }
    }

    [Fact]
    public void PlyMeshIO_RoundTripsMeshAscii()
    {
        string tempDir = CreateTempDirectory();
        try
        {
            string path = Path.Combine(tempDir, "mesh.ply");
            var vertices = new Tensor<double>(new[]
            {
                0.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0
            }, new[] { 3, 3 });
            var faces = new Tensor<int>(new[] { 0, 1, 2 }, new[] { 1, 3 });
            var normals = new Tensor<double>(new[]
            {
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0
            }, new[] { 3, 3 });
            var uvs = new Tensor<double>(new[]
            {
                0.0, 0.0,
                1.0, 0.0,
                0.0, 1.0
            }, new[] { 3, 2 });
            var colors = new Tensor<double>(new[]
            {
                255.0, 0.0, 0.0,
                0.0, 255.0, 0.0,
                0.0, 0.0, 255.0
            }, new[] { 3, 3 });

            var mesh = new TriangleMeshData<double>(vertices, faces, normals, colors, uvs);
            mesh.ComputeFaceNormals();
            PlyMeshIO.SaveMesh(mesh, path, binary: false);

            var loaded = PlyMeshIO.LoadMesh<double>(path, computeNormalsIfMissing: false);

            Assert.Equal(3, loaded.NumVertices);
            Assert.Equal(1, loaded.NumFaces);
            Assert.NotNull(loaded.VertexNormals);
            Assert.NotNull(loaded.VertexUVs);
            Assert.NotNull(loaded.VertexColors);
            Assert.NotNull(loaded.FaceNormals);

            Assert.Equal(255.0, loaded.VertexColors.Data.Span[0], 6);
            Assert.Equal(1.0, loaded.VertexUVs.Data.Span[2], 6);
            Assert.Equal(1.0, loaded.FaceNormals.Data.Span[2], 6);
        }
        finally
        {
            CleanupTempDirectory(tempDir);
        }
    }

    [Fact]
    public void PlyMeshIO_RoundTripsPointCloudBinary()
    {
        string tempDir = CreateTempDirectory();
        try
        {
            string path = Path.Combine(tempDir, "cloud.ply");
            var data = new Tensor<double>(new[]
            {
                0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 128.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0, 255.0, 0.0, 128.0, 0.0, 0.0, 1.0, 1.0, 0.0
            }, new[] { 2, 12 });
            var cloud = new PointCloudData<double>(data);

            PlyMeshIO.SavePointCloud(cloud, path, binary: true, includeColors: true, includeNormals: true, includeUvs: true);
            var loaded = PlyMeshIO.LoadPointCloud<double>(path);

            Assert.Equal(2, loaded.NumPoints);
            Assert.Equal(12, loaded.NumFeatures);
            Assert.Equal(255.0, loaded.Points.Data.Span[3], 6);
            Assert.Equal(1.0, loaded.Points.Data.Span[22], 6);
        }
        finally
        {
            CleanupTempDirectory(tempDir);
        }
    }

    [Fact]
    public void StlMeshIO_RoundTripsBinary()
    {
        string tempDir = CreateTempDirectory();
        try
        {
            string path = Path.Combine(tempDir, "mesh.stl");
            var vertices = new Tensor<double>(new[]
            {
                0.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                1.0, 1.0, 0.0,
                0.0, 1.0, 0.0
            }, new[] { 4, 3 });
            var faces = new Tensor<int>(new[] { 0, 1, 2, 0, 2, 3 }, new[] { 2, 3 });
            var mesh = new TriangleMeshData<double>(vertices, faces);

            StlMeshIO.SaveMesh(mesh, path, binary: true, solidName: "binary_test");
            var loaded = StlMeshIO.LoadMesh<double>(path, computeNormalsIfMissing: false);

            Assert.Equal(4, loaded.NumVertices);
            Assert.Equal(2, loaded.NumFaces);
            Assert.NotNull(loaded.FaceNormals);
            Assert.Equal(1.0, loaded.Vertices.Data.Span[3], 6);
        }
        finally
        {
            CleanupTempDirectory(tempDir);
        }
    }

    [Fact]
    public void StlMeshIO_LoadsAscii()
    {
        string tempDir = CreateTempDirectory();
        try
        {
            string path = Path.Combine(tempDir, "mesh_ascii.stl");
            string ascii = string.Join(Environment.NewLine, new[]
            {
                "solid ascii_test",
                "facet normal 0 0 1",
                "outer loop",
                "vertex 0 0 0",
                "vertex 1 0 0",
                "vertex 0 1 0",
                "endloop",
                "endfacet",
                "endsolid ascii_test"
            });
            File.WriteAllText(path, ascii, Encoding.ASCII);

            var loaded = StlMeshIO.LoadMesh<double>(path, computeNormalsIfMissing: false);

            Assert.Equal(3, loaded.NumVertices);
            Assert.Equal(1, loaded.NumFaces);
            Assert.NotNull(loaded.FaceNormals);
            Assert.Equal(1.0, loaded.FaceNormals.Data.Span[2], 6);
        }
        finally
        {
            CleanupTempDirectory(tempDir);
        }
    }

    private static string CreateTempDirectory()
    {
        string directory = Path.Combine(Path.GetTempPath(), "aidotnet_geom_io_" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void CleanupTempDirectory(string directory)
    {
        if (Directory.Exists(directory))
        {
            Directory.Delete(directory, true);
        }
    }
}
