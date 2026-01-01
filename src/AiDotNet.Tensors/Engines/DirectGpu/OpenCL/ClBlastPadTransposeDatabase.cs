// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

internal readonly struct ClBlastPadTransposeParameters
{
    public int Pad { get; init; }
    public int Tile { get; init; }
    public int WorkPerThread { get; init; }
}

internal static class ClBlastPadTransposeDatabase
{
    public static ClBlastPadTransposeParameters GetParameters(ClBlastDeviceInfo device)
    {
        if (!ClBlastDatabaseSearch.TryGetParameters(ClBlastPadTransposeDatabaseData.Vendors, device, out var parameters) ||
            parameters.Length < 3)
        {
            return GetDefaults();
        }

        int pad = parameters[0] >= 0 ? parameters[0] : 0;
        int tile = parameters[1] > 0 ? parameters[1] : 8;
        int wpt = parameters[2] > 0 ? parameters[2] : 1;

        return new ClBlastPadTransposeParameters
        {
            Pad = pad,
            Tile = tile,
            WorkPerThread = wpt
        };
    }

    private static ClBlastPadTransposeParameters GetDefaults()
    {
        return new ClBlastPadTransposeParameters
        {
            Pad = 0,
            Tile = 8,
            WorkPerThread = 1
        };
    }
}
