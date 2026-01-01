// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

internal static class ClBlastGemmRoutineDatabase
{
    public static int GetXgemmMinIndirectSize(ClBlastDeviceInfo device)
    {
        if (!ClBlastDatabaseSearch.TryGetParameters(ClBlastGemmRoutineDatabaseData.Vendors, device, out var parameters) ||
            parameters.Length < 1)
        {
            return 0;
        }

        return parameters[0];
    }
}
