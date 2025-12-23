using AiDotNet.Geometry.Data;
using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Geometry.Preprocessing;

/// <summary>
/// Provides voxelization utilities for converting point clouds and meshes to voxel grids.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class Voxelization<T>
{
    /// <summary>
    /// Clamps a value between a minimum and maximum. .NET 4.7.1 compatible.
    /// </summary>
    private static int ClampValue(int value, int min, int max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Converts a point cloud to an occupancy voxel grid.
    /// </summary>
    public static VoxelGridData<T> VoxelizePointCloud(
        PointCloudData<T> pointCloud,
        int resolution,
        double padding = 0.1)
    {
        int numPoints = pointCloud.NumPoints;
        if (numPoints == 0 || resolution <= 0)
        {
            return CreateEmptyGrid(resolution);
        }

        double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;

        for (int i = 0; i < numPoints; i++)
        {
            double x = NumOps.ToDouble(pointCloud.Points[i, 0]);
            double y = NumOps.ToDouble(pointCloud.Points[i, 1]);
            double z = NumOps.ToDouble(pointCloud.Points[i, 2]);

            minX = Math.Min(minX, x);
            minY = Math.Min(minY, y);
            minZ = Math.Min(minZ, z);
            maxX = Math.Max(maxX, x);
            maxY = Math.Max(maxY, y);
            maxZ = Math.Max(maxZ, z);
        }

        double rangeX = maxX - minX;
        double rangeY = maxY - minY;
        double rangeZ = maxZ - minZ;
        double maxRange = Math.Max(Math.Max(rangeX, rangeY), rangeZ);
        
        if (maxRange < 1e-10)
        {
            maxRange = 1.0;
        }

        double padAmount = maxRange * padding;
        minX -= padAmount;
        minY -= padAmount;
        minZ -= padAmount;

        double voxelSize = (maxRange + 2 * padAmount) / resolution;

        var voxels = new T[resolution * resolution * resolution];

        for (int i = 0; i < numPoints; i++)
        {
            double x = NumOps.ToDouble(pointCloud.Points[i, 0]);
            double y = NumOps.ToDouble(pointCloud.Points[i, 1]);
            double z = NumOps.ToDouble(pointCloud.Points[i, 2]);

            int vx = (int)Math.Floor((x - minX) / voxelSize);
            int vy = (int)Math.Floor((y - minY) / voxelSize);
            int vz = (int)Math.Floor((z - minZ) / voxelSize);

            vx = ClampValue(vx, 0, resolution - 1);
            vy = ClampValue(vy, 0, resolution - 1);
            vz = ClampValue(vz, 0, resolution - 1);

            int idx = vz * resolution * resolution + vy * resolution + vx;
            voxels[idx] = NumOps.One;
        }

        var voxelTensor = new Tensor<T>(voxels, [resolution, resolution, resolution]);
        var origin = new Vector<T>([NumOps.FromDouble(minX), NumOps.FromDouble(minY), NumOps.FromDouble(minZ)]);
        var voxelSizeVec = new Vector<T>([NumOps.FromDouble(voxelSize), NumOps.FromDouble(voxelSize), NumOps.FromDouble(voxelSize)]);
        
        return new VoxelGridData<T>(voxelTensor, origin, voxelSizeVec);
    }

    /// <summary>
    /// Converts a triangle mesh to an occupancy voxel grid using surface voxelization.
    /// </summary>
    public static VoxelGridData<T> VoxelizeMeshSurface(
        TriangleMeshData<T> mesh,
        int resolution,
        double padding = 0.1)
    {
        int numVertices = mesh.NumVertices;
        int numFaces = mesh.NumFaces;

        if (numVertices == 0 || numFaces == 0 || resolution <= 0)
        {
            return CreateEmptyGrid(resolution);
        }

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

        double rangeX = maxX - minX;
        double rangeY = maxY - minY;
        double rangeZ = maxZ - minZ;
        double maxRange = Math.Max(Math.Max(rangeX, rangeY), rangeZ);

        if (maxRange < 1e-10)
        {
            maxRange = 1.0;
        }

        double padAmount = maxRange * padding;
        minX -= padAmount;
        minY -= padAmount;
        minZ -= padAmount;

        double voxelSize = (maxRange + 2 * padAmount) / resolution;
        var voxels = new T[resolution * resolution * resolution];

        int samplesPerEdge = Math.Max(1, resolution / 4);
        
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

            for (int i = 0; i <= samplesPerEdge; i++)
            {
                for (int j = 0; j <= samplesPerEdge - i; j++)
                {
                    double u = (double)i / samplesPerEdge;
                    double v = (double)j / samplesPerEdge;
                    double w = 1 - u - v;

                    double px = w * x0 + u * x1 + v * x2;
                    double py = w * y0 + u * y1 + v * y2;
                    double pz = w * z0 + u * z1 + v * z2;

                    int vx = (int)Math.Floor((px - minX) / voxelSize);
                    int vy = (int)Math.Floor((py - minY) / voxelSize);
                    int vz = (int)Math.Floor((pz - minZ) / voxelSize);

                    vx = ClampValue(vx, 0, resolution - 1);
                    vy = ClampValue(vy, 0, resolution - 1);
                    vz = ClampValue(vz, 0, resolution - 1);

                    int idx = vz * resolution * resolution + vy * resolution + vx;
                    voxels[idx] = NumOps.One;
                }
            }
        }

        var voxelTensor = new Tensor<T>(voxels, [resolution, resolution, resolution]);
        var origin = new Vector<T>([NumOps.FromDouble(minX), NumOps.FromDouble(minY), NumOps.FromDouble(minZ)]);
        var voxelSizeVec = new Vector<T>([NumOps.FromDouble(voxelSize), NumOps.FromDouble(voxelSize), NumOps.FromDouble(voxelSize)]);
        
        return new VoxelGridData<T>(voxelTensor, origin, voxelSizeVec);
    }

    /// <summary>
    /// Computes the Intersection over Union (IoU) between two voxel grids.
    /// </summary>
    public static double IntersectionOverUnion(
        VoxelGridData<T> prediction,
        VoxelGridData<T> groundTruth,
        double threshold = 0.5)
    {
        if (prediction.Depth != groundTruth.Depth ||
            prediction.Height != groundTruth.Height ||
            prediction.Width != groundTruth.Width)
        {
            throw new ArgumentException("Voxel grids must have the same dimensions");
        }

        T thresholdT = NumOps.FromDouble(threshold);

        int intersection = 0;
        int union = 0;

        for (int z = 0; z < prediction.Depth; z++)
        {
            for (int y = 0; y < prediction.Height; y++)
            {
                for (int x = 0; x < prediction.Width; x++)
                {
                    bool predOccupied = NumOps.GreaterThan(prediction.Voxels[z, y, x], thresholdT);
                    bool gtOccupied = NumOps.GreaterThan(groundTruth.Voxels[z, y, x], thresholdT);

                    if (predOccupied && gtOccupied)
                    {
                        intersection++;
                    }
                    if (predOccupied || gtOccupied)
                    {
                        union++;
                    }
                }
            }
        }

        return union > 0 ? (double)intersection / union : 0;
    }

    /// <summary>
    /// Applies morphological dilation to a voxel grid.
    /// </summary>
    public static VoxelGridData<T> Dilate(VoxelGridData<T> voxelGrid, int radius = 1)
    {
        int depth = voxelGrid.Depth;
        int height = voxelGrid.Height;
        int width = voxelGrid.Width;
        var result = new T[depth * height * width];

        for (int z = 0; z < depth; z++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    bool occupied = false;

                    for (int dz = -radius; dz <= radius && !occupied; dz++)
                    {
                        for (int dy = -radius; dy <= radius && !occupied; dy++)
                        {
                            for (int dx = -radius; dx <= radius && !occupied; dx++)
                            {
                                int nz = z + dz;
                                int ny = y + dy;
                                int nx = x + dx;

                                if (nz >= 0 && nz < depth &&
                                    ny >= 0 && ny < height &&
                                    nx >= 0 && nx < width)
                                {
                                    if (NumOps.GreaterThan(voxelGrid.Voxels[nz, ny, nx], NumOps.Zero))
                                    {
                                        occupied = true;
                                    }
                                }
                            }
                        }
                    }

                    int idx = z * height * width + y * width + x;
                    result[idx] = occupied ? NumOps.One : NumOps.Zero;
                }
            }
        }

        var voxelTensor = new Tensor<T>(result, [depth, height, width]);
        return new VoxelGridData<T>(voxelTensor, voxelGrid.Origin, voxelGrid.VoxelSize);
    }

    /// <summary>
    /// Applies morphological erosion to a voxel grid.
    /// </summary>
    public static VoxelGridData<T> Erode(VoxelGridData<T> voxelGrid, int radius = 1)
    {
        int depth = voxelGrid.Depth;
        int height = voxelGrid.Height;
        int width = voxelGrid.Width;
        var result = new T[depth * height * width];

        for (int z = 0; z < depth; z++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    bool allOccupied = true;

                    for (int dz = -radius; dz <= radius && allOccupied; dz++)
                    {
                        for (int dy = -radius; dy <= radius && allOccupied; dy++)
                        {
                            for (int dx = -radius; dx <= radius && allOccupied; dx++)
                            {
                                int nz = z + dz;
                                int ny = y + dy;
                                int nx = x + dx;

                                if (nz >= 0 && nz < depth &&
                                    ny >= 0 && ny < height &&
                                    nx >= 0 && nx < width)
                                {
                                    if (!NumOps.GreaterThan(voxelGrid.Voxels[nz, ny, nx], NumOps.Zero))
                                    {
                                        allOccupied = false;
                                    }
                                }
                                else
                                {
                                    allOccupied = false;
                                }
                            }
                        }
                    }

                    int idx = z * height * width + y * width + x;
                    result[idx] = allOccupied ? NumOps.One : NumOps.Zero;
                }
            }
        }

        var voxelTensor = new Tensor<T>(result, [depth, height, width]);
        return new VoxelGridData<T>(voxelTensor, voxelGrid.Origin, voxelGrid.VoxelSize);
    }

    private static VoxelGridData<T> CreateEmptyGrid(int resolution)
    {
        resolution = Math.Max(1, resolution);
        var voxelTensor = new Tensor<T>(new T[resolution * resolution * resolution], [resolution, resolution, resolution]);
        var origin = new Vector<T>([NumOps.Zero, NumOps.Zero, NumOps.Zero]);
        var voxelSize = new Vector<T>([NumOps.One, NumOps.One, NumOps.One]);
        return new VoxelGridData<T>(voxelTensor, origin, voxelSize);
    }
}
