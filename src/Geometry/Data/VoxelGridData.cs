using System;
using System.Collections.Generic;
using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Geometry.Data;

/// <summary>
/// Represents a voxel grid with world-space metadata.
/// </summary>
/// <typeparam name="T">The numeric type used for voxel values.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A voxel grid is like a 3D image made of cubes
/// (voxels). Each voxel stores a value, such as occupancy or density.
/// </remarks>
public sealed class VoxelGridData<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Voxel values of shape [depth, height, width] or [depth, height, width, channels].
    /// </summary>
    public Tensor<T> Voxels { get; }

    /// <summary>
    /// World-space origin of the voxel grid (corner of voxel [0,0,0]).
    /// </summary>
    public Vector<T> Origin { get; }

    /// <summary>
    /// Size of a voxel along each axis in world units.
    /// </summary>
    public Vector<T> VoxelSize { get; }

    /// <summary>
    /// Optional metadata associated with the voxel grid.
    /// </summary>
    public Dictionary<string, object>? Metadata { get; set; }

    /// <summary>
    /// Depth (Z dimension) of the grid.
    /// </summary>
    public int Depth => Voxels.Shape[0];

    /// <summary>
    /// Height (Y dimension) of the grid.
    /// </summary>
    public int Height => Voxels.Shape[1];

    /// <summary>
    /// Width (X dimension) of the grid.
    /// </summary>
    public int Width => Voxels.Shape[2];

    /// <summary>
    /// Number of channels per voxel (1 if no channel dimension).
    /// </summary>
    public int Channels => Voxels.Shape.Length == 4 ? Voxels.Shape[3] : 1;

    /// <summary>
    /// Initializes a new instance of the VoxelGridData class.
    /// </summary>
    public VoxelGridData(
        Tensor<T> voxels,
        Vector<T>? origin = null,
        Vector<T>? voxelSize = null)
    {
        ValidateVoxels(voxels);

        Voxels = voxels;
        Origin = origin ?? new Vector<T>(new[] { NumOps.Zero, NumOps.Zero, NumOps.Zero });
        VoxelSize = voxelSize ?? new Vector<T>(new[] { NumOps.One, NumOps.One, NumOps.One });

        ValidateVector(Origin, nameof(origin));
        ValidateVector(VoxelSize, nameof(voxelSize));
        ValidateVoxelSize(VoxelSize);
    }

    /// <summary>
    /// Computes the axis-aligned bounds of the grid in world coordinates.
    /// </summary>
    public (Vector<T> min, Vector<T> max) ComputeBounds()
    {
        var maxX = NumOps.Add(Origin[0], NumOps.Multiply(VoxelSize[0], NumOps.FromDouble(Width)));
        var maxY = NumOps.Add(Origin[1], NumOps.Multiply(VoxelSize[1], NumOps.FromDouble(Height)));
        var maxZ = NumOps.Add(Origin[2], NumOps.Multiply(VoxelSize[2], NumOps.FromDouble(Depth)));

        return (new Vector<T>(new[] { Origin[0], Origin[1], Origin[2] }),
            new Vector<T>(new[] { maxX, maxY, maxZ }));
    }

    /// <summary>
    /// Gets the world-space center of a voxel at the given indices.
    /// </summary>
    public Vector<T> GetVoxelCenter(int x, int y, int z)
    {
        ValidateIndices(x, y, z);

        var half = NumOps.FromDouble(0.5);
        var cx = NumOps.Add(Origin[0], NumOps.Multiply(VoxelSize[0], NumOps.Add(NumOps.FromDouble(x), half)));
        var cy = NumOps.Add(Origin[1], NumOps.Multiply(VoxelSize[1], NumOps.Add(NumOps.FromDouble(y), half)));
        var cz = NumOps.Add(Origin[2], NumOps.Multiply(VoxelSize[2], NumOps.Add(NumOps.FromDouble(z), half)));

        return new Vector<T>(new[] { cx, cy, cz });
    }

    /// <summary>
    /// Converts occupied voxels to a point cloud using voxel centers.
    /// </summary>
    public PointCloudData<T> ToPointCloud(
        T occupancyThreshold,
        int occupancyChannel = 0,
        bool includeOccupancyAsFeature = false)
    {
        if (Voxels.Shape.Length == 4 && (occupancyChannel < 0 || occupancyChannel >= Channels))
        {
            throw new ArgumentOutOfRangeException(nameof(occupancyChannel), "Occupancy channel is out of range.");
        }

        var xCenters = BuildAxisCenters(Origin[0], VoxelSize[0], Width);
        var yCenters = BuildAxisCenters(Origin[1], VoxelSize[1], Height);
        var zCenters = BuildAxisCenters(Origin[2], VoxelSize[2], Depth);

        int stride = includeOccupancyAsFeature ? 4 : 3;
        var points = new List<T>();
        var data = Voxels.Data.Span;

        int channelStride = Channels;
        int sliceStride = Height * Width * channelStride;
        int rowStride = Width * channelStride;

        for (int z = 0; z < Depth; z++)
        {
            int zOffset = z * sliceStride;
            for (int y = 0; y < Height; y++)
            {
                int yOffset = zOffset + y * rowStride;
                for (int x = 0; x < Width; x++)
                {
                    int baseIndex = yOffset + x * channelStride;
                    var value = data[baseIndex + (Voxels.Shape.Length == 4 ? occupancyChannel : 0)];

                    if (!NumOps.GreaterThan(value, occupancyThreshold))
                    {
                        continue;
                    }

                    points.Add(xCenters[x]);
                    points.Add(yCenters[y]);
                    points.Add(zCenters[z]);

                    if (includeOccupancyAsFeature)
                    {
                        points.Add(value);
                    }
                }
            }
        }

        int pointCount = points.Count / stride;
        var tensor = new Tensor<T>(points.ToArray(), new[] { pointCount, stride });
        return new PointCloudData<T>(tensor);
    }

    private void ValidateIndices(int x, int y, int z)
    {
        if (x < 0 || x >= Width)
        {
            throw new ArgumentOutOfRangeException(nameof(x), "Voxel X index is out of range.");
        }
        if (y < 0 || y >= Height)
        {
            throw new ArgumentOutOfRangeException(nameof(y), "Voxel Y index is out of range.");
        }
        if (z < 0 || z >= Depth)
        {
            throw new ArgumentOutOfRangeException(nameof(z), "Voxel Z index is out of range.");
        }
    }

    private static T[] BuildAxisCenters(T origin, T voxelSize, int count)
    {
        var centers = new T[count];
        var half = NumOps.FromDouble(0.5);

        for (int i = 0; i < count; i++)
        {
            var offset = NumOps.Add(NumOps.FromDouble(i), half);
            centers[i] = NumOps.Add(origin, NumOps.Multiply(voxelSize, offset));
        }

        return centers;
    }

    private static void ValidateVoxels(Tensor<T> voxels)
    {
        if (voxels == null)
        {
            throw new ArgumentNullException(nameof(voxels));
        }
        if (voxels.Shape.Length != 3 && voxels.Shape.Length != 4)
        {
            throw new ArgumentException("Voxels must have shape [D, H, W] or [D, H, W, C].", nameof(voxels));
        }
        if (voxels.Shape[0] <= 0 || voxels.Shape[1] <= 0 || voxels.Shape[2] <= 0)
        {
            throw new ArgumentException("Voxel dimensions must be positive.", nameof(voxels));
        }
        if (voxels.Shape.Length == 4 && voxels.Shape[3] <= 0)
        {
            throw new ArgumentException("Voxel channel dimension must be positive.", nameof(voxels));
        }
    }

    private static void ValidateVector(Vector<T> vector, string paramName)
    {
        if (vector == null)
        {
            throw new ArgumentNullException(paramName);
        }
        if (vector.Length != 3)
        {
            throw new ArgumentException("Vector must have length 3.", paramName);
        }
    }

    private static void ValidateVoxelSize(Vector<T> voxelSize)
    {
        for (int i = 0; i < 3; i++)
        {
            if (!NumOps.GreaterThan(voxelSize[i], NumOps.Zero))
            {
                throw new ArgumentOutOfRangeException(nameof(voxelSize), "VoxelSize components must be positive.");
            }
        }
    }
}
