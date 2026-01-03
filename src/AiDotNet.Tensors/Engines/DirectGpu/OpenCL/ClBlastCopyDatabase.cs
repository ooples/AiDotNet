// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

internal readonly struct ClBlastCopyParameters
{
    public int DimX { get; init; }
    public int DimY { get; init; }
    public int VectorWidth { get; init; }
    public int WorkPerThread { get; init; }
}

internal static class ClBlastCopyDatabase
{
    public static ClBlastCopyParameters GetParameters(ClBlastDeviceInfo device)
    {
        if (!ClBlastDatabaseSearch.TryGetParameters(ClBlastCopyDatabaseData.Vendors, device, out var parameters) ||
            parameters.Length < 4)
        {
            return GetDefaults();
        }

        int dimX = parameters[0] > 0 ? parameters[0] : 8;
        int dimY = parameters[1] > 0 ? parameters[1] : 8;
        int vectorWidth = parameters[2] > 0 ? parameters[2] : 1;
        int workPerThread = parameters[3] > 0 ? parameters[3] : 1;

        return new ClBlastCopyParameters
        {
            DimX = dimX,
            DimY = dimY,
            VectorWidth = vectorWidth,
            WorkPerThread = workPerThread
        };
    }

    private static ClBlastCopyParameters GetDefaults()
    {
        return new ClBlastCopyParameters
        {
            DimX = 8,
            DimY = 8,
            VectorWidth = 1,
            WorkPerThread = 1
        };
    }
}
