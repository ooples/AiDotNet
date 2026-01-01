// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

internal readonly struct ClBlastPadParameters
{
    public int DimX { get; init; }
    public int DimY { get; init; }
    public int WorkPerThreadX { get; init; }
    public int WorkPerThreadY { get; init; }
}

internal static class ClBlastPadDatabase
{
    public static ClBlastPadParameters GetParameters(ClBlastDeviceInfo device)
    {
        if (!ClBlastDatabaseSearch.TryGetParameters(ClBlastPadDatabaseData.Vendors, device, out var parameters) ||
            parameters.Length < 4)
        {
            return GetDefaults();
        }

        int dimX = parameters[0] > 0 ? parameters[0] : 8;
        int dimY = parameters[1] > 0 ? parameters[1] : 8;
        int wptX = parameters[2] > 0 ? parameters[2] : 1;
        int wptY = parameters[3] > 0 ? parameters[3] : 1;

        return new ClBlastPadParameters
        {
            DimX = dimX,
            DimY = dimY,
            WorkPerThreadX = wptX,
            WorkPerThreadY = wptY
        };
    }

    private static ClBlastPadParameters GetDefaults()
    {
        return new ClBlastPadParameters
        {
            DimX = 8,
            DimY = 8,
            WorkPerThreadX = 1,
            WorkPerThreadY = 1
        };
    }
}
