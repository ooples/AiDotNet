// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

internal readonly struct ClBlastTransposeParameters
{
    public int Dim { get; init; }
    public int WorkPerThread { get; init; }
    public int Pad { get; init; }
    public int Shuffle { get; init; }
}

internal static class ClBlastTransposeDatabase
{
    public static ClBlastTransposeParameters GetParameters(ClBlastDeviceInfo device)
    {
        if (!ClBlastDatabaseSearch.TryGetParameters(ClBlastTransposeDatabaseData.Vendors, device, out var parameters) ||
            parameters.Length < 4)
        {
            return GetDefaults();
        }

        int dim = parameters[0] > 0 ? parameters[0] : 8;
        int workPerThread = parameters[1] > 0 ? parameters[1] : 1;
        int pad = parameters[2] >= 0 ? parameters[2] : 0;
        int shuffle = parameters[3] >= 0 ? parameters[3] : 0;

        return new ClBlastTransposeParameters
        {
            Dim = dim,
            WorkPerThread = workPerThread,
            Pad = pad,
            Shuffle = shuffle
        };
    }

    private static ClBlastTransposeParameters GetDefaults()
    {
        return new ClBlastTransposeParameters
        {
            Dim = 8,
            WorkPerThread = 1,
            Pad = 0,
            Shuffle = 0
        };
    }
}
