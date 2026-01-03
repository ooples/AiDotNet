// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

internal readonly struct ClBlastXgemmDirectParameters
{
    public int Kwid { get; init; }
    public int MdimAd { get; init; }
    public int MdimCd { get; init; }
    public int NdimBd { get; init; }
    public int NdimCd { get; init; }
    public int PadA { get; init; }
    public int PadB { get; init; }
    public int Vwm { get; init; }
    public int Vwn { get; init; }
    public int Wgd { get; init; }
}

internal static class ClBlastXgemmDirectDatabase
{
    public static ClBlastXgemmDirectParameters GetParameters(ClBlastDeviceInfo device)
    {
        if (!ClBlastDatabaseSearch.TryGetParameters(ClBlastXgemmDirectDatabaseData.Vendors, device, out var parameters) ||
            parameters.Length < 10)
        {
            return GetDefaults();
        }

        int kwid = parameters[0] > 0 ? parameters[0] : 1;
        int mdimAd = parameters[1] > 0 ? parameters[1] : 8;
        int mdimCd = parameters[2] > 0 ? parameters[2] : 8;
        int ndimBd = parameters[3] > 0 ? parameters[3] : 8;
        int ndimCd = parameters[4] > 0 ? parameters[4] : 8;
        int padA = parameters[5] >= 0 ? parameters[5] : 1;
        int padB = parameters[6] >= 0 ? parameters[6] : 1;
        int vwmD = parameters[7] > 0 ? parameters[7] : 1;
        int vwnD = parameters[8] > 0 ? parameters[8] : 1;
        int wgd = parameters[9] > 0 ? parameters[9] : 8;

        return new ClBlastXgemmDirectParameters
        {
            Kwid = kwid,
            MdimAd = mdimAd,
            MdimCd = mdimCd,
            NdimBd = ndimBd,
            NdimCd = ndimCd,
            PadA = padA,
            PadB = padB,
            Vwm = vwmD,
            Vwn = vwnD,
            Wgd = wgd
        };
    }

    private static ClBlastXgemmDirectParameters GetDefaults()
    {
        return new ClBlastXgemmDirectParameters
        {
            Kwid = 1,
            MdimAd = 8,
            MdimCd = 8,
            NdimBd = 8,
            NdimCd = 8,
            PadA = 1,
            PadB = 1,
            Vwm = 1,
            Vwn = 1,
            Wgd = 8
        };
    }
}
