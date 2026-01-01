// Copyright (c) AiDotNet. All rights reserved.
// CLBlast device info mapping (vendor/architecture/device name normalization).
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

internal readonly struct ClBlastDeviceInfo
{
    public string Type { get; }
    public string Vendor { get; }
    public string Architecture { get; }
    public string DeviceName { get; }

    private ClBlastDeviceInfo(string type, string vendor, string architecture, string deviceName)
    {
        Type = type;
        Vendor = vendor;
        Architecture = architecture;
        DeviceName = deviceName;
    }

    public static ClBlastDeviceInfo FromContext(DirectOpenClContext context)
    {
        string vendor = NormalizeVendor(context.DeviceVendor);
        string type = NormalizeDeviceType(context.DeviceType);

        bool hasAmdAttributes = HasExtension(context.Extensions, "cl_amd_device_attribute_query");
        bool hasNvAttributes = HasExtension(context.Extensions, "cl_nv_device_attribute_query");

        string rawDeviceName = context.DeviceName ?? string.Empty;
        string architecture = string.Empty;
        if (hasNvAttributes)
        {
            uint major = OpenClNativeBindings.GetDeviceInfoUInt(context.Device, OpenClNativeBindings.CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV);
            uint minor = OpenClNativeBindings.GetDeviceInfoUInt(context.Device, OpenClNativeBindings.CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV);
            if (major > 0 || minor > 0)
                architecture = $"SM{major}.{minor}";
        }
        else if (hasAmdAttributes)
        {
            architecture = rawDeviceName;
        }

        architecture = NormalizeArchitecture(architecture);

        string deviceName = rawDeviceName;
        if (hasAmdAttributes && !string.IsNullOrWhiteSpace(context.DeviceBoardName))
            deviceName = context.DeviceBoardName;

        deviceName = NormalizeDeviceName(deviceName);
        return new ClBlastDeviceInfo(type, vendor, architecture, deviceName);
    }

    private static bool HasExtension(string extensions, string extension)
    {
        return extensions.IndexOf(extension, StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private static string NormalizeVendor(string vendor)
    {
        return vendor switch
        {
            "Intel(R) Corporation" => "Intel",
            "GenuineIntel" => "Intel",
            "Advanced Micro Devices, Inc." => "AMD",
            "NVIDIA Corporation" => "NVIDIA",
            _ => string.IsNullOrWhiteSpace(vendor) ? "default" : vendor
        };
    }

    private static string NormalizeArchitecture(string architecture)
    {
        if (string.IsNullOrWhiteSpace(architecture))
            return string.Empty;

        return architecture switch
        {
            "gfx803" => "Fiji",
            "gfx900" => "Vega",
            _ => architecture
        };
    }

    private static string NormalizeDeviceName(string deviceName)
    {
        if (string.IsNullOrEmpty(deviceName))
            return deviceName;

        if (deviceName.Contains("pthread-", StringComparison.Ordinal))
            deviceName = deviceName.Replace("pthread-", string.Empty, StringComparison.Ordinal);

        return deviceName.Trim();
    }

    private static string NormalizeDeviceType(ulong deviceType)
    {
        if ((deviceType & OpenClNativeBindings.CL_DEVICE_TYPE_GPU) != 0)
            return "GPU";
        if ((deviceType & OpenClNativeBindings.CL_DEVICE_TYPE_CPU) != 0)
            return "CPU";
        return "default";
    }
}
