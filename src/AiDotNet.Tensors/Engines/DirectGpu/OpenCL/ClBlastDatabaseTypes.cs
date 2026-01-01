// Copyright (c) AiDotNet. All rights reserved.
// CLBlast database types and search helpers (Apache 2.0 compatible data).
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

internal readonly struct ClBlastDeviceEntry
{
    public readonly string Name;
    public readonly short[] Parameters;

    public ClBlastDeviceEntry(string name, short[] parameters)
    {
        Name = name;
        Parameters = parameters;
    }
}

internal readonly struct ClBlastArchitectureEntry
{
    public readonly string Name;
    public readonly ClBlastDeviceEntry[] Devices;

    public ClBlastArchitectureEntry(string name, ClBlastDeviceEntry[] devices)
    {
        Name = name;
        Devices = devices;
    }
}

internal readonly struct ClBlastVendorEntry
{
    public readonly string Type;
    public readonly string Vendor;
    public readonly ClBlastArchitectureEntry[] Architectures;

    public ClBlastVendorEntry(string type, string vendor, ClBlastArchitectureEntry[] architectures)
    {
        Type = type;
        Vendor = vendor;
        Architectures = architectures;
    }
}

internal static class ClBlastDatabaseSearch
{
    public static bool TryGetParameters(
        ClBlastVendorEntry[] vendors,
        ClBlastDeviceInfo device,
        out short[] parameters)
    {
        if (TryFindVendorType(vendors, device.Vendor, device.Type, device.Architecture, device.DeviceName, out parameters))
            return true;

        return TryFindVendorType(vendors, "default", "default", device.Architecture, device.DeviceName, out parameters);
    }

    private static bool TryFindVendorType(
        ClBlastVendorEntry[] vendors,
        string vendor,
        string type,
        string architecture,
        string deviceName,
        out short[] parameters)
    {
        foreach (var entry in vendors)
        {
            if (!entry.Vendor.Equals(vendor, System.StringComparison.Ordinal) ||
                !entry.Type.Equals(type, System.StringComparison.Ordinal))
                continue;

            if (TryFindArchitecture(entry.Architectures, architecture, deviceName, out parameters))
                return true;

            return TryFindArchitecture(entry.Architectures, "default", deviceName, out parameters);
        }

        parameters = System.Array.Empty<short>();
        return false;
    }

    private static bool TryFindArchitecture(
        ClBlastArchitectureEntry[] architectures,
        string architecture,
        string deviceName,
        out short[] parameters)
    {
        foreach (var entry in architectures)
        {
            if (!entry.Name.Equals(architecture, System.StringComparison.Ordinal))
                continue;

            if (TryFindDevice(entry.Devices, deviceName, out parameters))
                return true;

            return TryFindDevice(entry.Devices, "default", out parameters);
        }

        parameters = System.Array.Empty<short>();
        return false;
    }

    private static bool TryFindDevice(
        ClBlastDeviceEntry[] devices,
        string deviceName,
        out short[] parameters)
    {
        var target = deviceName.Length > 50 ? deviceName.Substring(0, 50) : deviceName;
        foreach (var entry in devices)
        {
            if (entry.Name.Equals(target, System.StringComparison.Ordinal))
            {
                parameters = entry.Parameters;
                return true;
            }
        }

        parameters = System.Array.Empty<short>();
        return false;
    }
}
