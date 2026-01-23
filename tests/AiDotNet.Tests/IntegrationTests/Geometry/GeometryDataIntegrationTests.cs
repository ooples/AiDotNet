using AiDotNet.Geometry.Data;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Geometry;

public class GeometryDataIntegrationTests
{
    [Fact]
    public void TriangleMeshData_ComputesNormalsAndBounds()
    {
        var vertices = new Tensor<double>(new[]
        {
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        }, new[] { 3, 3 });
        var faces = new Tensor<int>(new[] { 0, 1, 2 }, new[] { 1, 3 });

        var mesh = new TriangleMeshData<double>(vertices, faces);

        var bounds = mesh.ComputeBounds();
        Assert.Equal(0.0, bounds.min[0], 6);
        Assert.Equal(0.0, bounds.min[1], 6);
        Assert.Equal(0.0, bounds.min[2], 6);
        Assert.Equal(1.0, bounds.max[0], 6);
        Assert.Equal(1.0, bounds.max[1], 6);
        Assert.Equal(0.0, bounds.max[2], 6);

        var faceNormals = mesh.ComputeFaceNormals();
        Assert.Equal(new[] { 1, 3 }, faceNormals.Shape);
        Assert.Equal(0.0, faceNormals[0], 6);
        Assert.Equal(0.0, faceNormals[1], 6);
        Assert.Equal(1.0, faceNormals[2], 6);

        var vertexNormals = mesh.ComputeVertexNormals();
        Assert.Equal(new[] { 3, 3 }, vertexNormals.Shape);
        for (int i = 0; i < 3; i++)
        {
            int offset = i * 3;
            Assert.Equal(0.0, vertexNormals[offset], 6);
            Assert.Equal(0.0, vertexNormals[offset + 1], 6);
            Assert.Equal(1.0, vertexNormals[offset + 2], 6);
        }
    }

    [Fact]
    public void TriangleMeshData_ToPointCloud_IncludesFeatures()
    {
        var vertices = new Tensor<double>(new[]
        {
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        }, new[] { 3, 3 });
        var faces = new Tensor<int>(new[] { 0, 1, 2 }, new[] { 1, 3 });
        var colors = new Tensor<double>(new[]
        {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        }, new[] { 3, 3 });

        var mesh = new TriangleMeshData<double>(vertices, faces, vertexColors: colors);
        var cloud = mesh.ToPointCloud(includeColors: true, includeNormals: true);

        Assert.Equal(3, cloud.NumPoints);
        Assert.Equal(9, cloud.NumFeatures);
        Assert.Equal(3, cloud.Points.Shape[0]);
        Assert.Equal(9, cloud.Points.Shape[1]);
    }

    [Fact]
    public void VoxelGridData_ToPointCloud_ThresholdsAndCenters()
    {
        var data = new[]
        {
            1.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.75
        };
        var voxels = new Tensor<double>(data, new[] { 2, 2, 2 });
        var grid = new VoxelGridData<double>(voxels);

        var cloud = grid.ToPointCloud(0.5);

        Assert.Equal(2, cloud.NumPoints);
        Assert.Equal(3, cloud.NumFeatures);

        var points = cloud.Points.AsSpan();
        Assert.Equal(0.5, points[0], 6);
        Assert.Equal(0.5, points[1], 6);
        Assert.Equal(0.5, points[2], 6);
        Assert.Equal(1.5, points[3], 6);
        Assert.Equal(1.5, points[4], 6);
        Assert.Equal(1.5, points[5], 6);
    }
}
