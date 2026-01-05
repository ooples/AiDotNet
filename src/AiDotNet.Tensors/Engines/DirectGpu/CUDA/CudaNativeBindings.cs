// Copyright (c) AiDotNet. All rights reserved.
// CUDA Driver API bindings for direct kernel execution.
using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// CUDA device attributes for querying device properties.
/// Values align with CUdevice_attribute in the CUDA driver API.
/// </summary>
public enum CudaDeviceAttribute
{
    MaxSharedMemoryPerBlock = 8,
    MultiprocessorCount = 16,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76
}

internal static class CudaNativeBindings
{
#if WINDOWS
    private const string CudaLibrary = "nvcuda";
#else
    private const string CudaLibrary = "libcuda";
#endif

    private static bool _isAvailable;
    private static bool _availabilityChecked;

    /// <summary>
    /// Gets whether the CUDA driver API is available on this system.
    /// </summary>
    public static bool IsAvailable
    {
        get
        {
            if (!_availabilityChecked)
            {
                _availabilityChecked = true;
                try
                {
                    var result = CuBlasNative.cuInit(0);
                    if (result != CudaResult.Success)
                    {
                        _isAvailable = false;
                    }
                    else
                    {
                        result = CuBlasNative.cuDeviceGetCount(out int count);
                        _isAvailable = result == CudaResult.Success && count > 0;
                    }
                }
                catch (DllNotFoundException)
                {
                    _isAvailable = false;
                }
                catch (EntryPointNotFoundException)
                {
                    _isAvailable = false;
                }
                catch
                {
                    _isAvailable = false;
                }
            }
            return _isAvailable;
        }
    }

    [DllImport(CudaLibrary, EntryPoint = "cuDeviceTotalMem_v2")]
    public static extern CudaResult cuDeviceTotalMem(out ulong bytes, int device);

    [DllImport(CudaLibrary, EntryPoint = "cuModuleLoadData")]
    public static extern CudaResult cuModuleLoadData(out IntPtr module, IntPtr image);

    [DllImport(CudaLibrary, EntryPoint = "cuModuleUnload")]
    public static extern CudaResult cuModuleUnload(IntPtr module);

    [DllImport(CudaLibrary, EntryPoint = "cuModuleGetFunction")]
    public static extern CudaResult cuModuleGetFunction(out IntPtr function, IntPtr module, string name);

    [DllImport(CudaLibrary, EntryPoint = "cuLaunchKernel")]
    public static extern CudaResult cuLaunchKernel(
        IntPtr function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes,
        IntPtr stream,
        IntPtr kernelParams,
        IntPtr extra);

    [DllImport(CudaLibrary, EntryPoint = "cuStreamCreate")]
    public static extern CudaResult cuStreamCreate(out IntPtr stream, uint flags);

    [DllImport(CudaLibrary, EntryPoint = "cuStreamDestroy_v2")]
    public static extern CudaResult cuStreamDestroy(IntPtr stream);

    [DllImport(CudaLibrary, EntryPoint = "cuStreamSynchronize")]
    public static extern CudaResult cuStreamSynchronize(IntPtr stream);

    [DllImport(CudaLibrary, EntryPoint = "cuStreamQuery")]
    public static extern CudaResult cuStreamQuery(IntPtr stream);

    [DllImport(CudaLibrary, EntryPoint = "cuStreamCreateWithPriority")]
    public static extern CudaResult cuStreamCreateWithPriority(out IntPtr stream, uint flags, int priority);

    [DllImport(CudaLibrary, EntryPoint = "cuStreamWaitEvent")]
    public static extern CudaResult cuStreamWaitEvent(IntPtr stream, IntPtr hEvent, uint flags);

    // Event APIs
    [DllImport(CudaLibrary, EntryPoint = "cuEventCreate")]
    public static extern CudaResult cuEventCreate(out IntPtr phEvent, uint flags);

    [DllImport(CudaLibrary, EntryPoint = "cuEventDestroy_v2")]
    public static extern CudaResult cuEventDestroy(IntPtr hEvent);

    [DllImport(CudaLibrary, EntryPoint = "cuEventRecord")]
    public static extern CudaResult cuEventRecord(IntPtr hEvent, IntPtr stream);

    [DllImport(CudaLibrary, EntryPoint = "cuEventSynchronize")]
    public static extern CudaResult cuEventSynchronize(IntPtr hEvent);

    [DllImport(CudaLibrary, EntryPoint = "cuEventQuery")]
    public static extern CudaResult cuEventQuery(IntPtr hEvent);

    [DllImport(CudaLibrary, EntryPoint = "cuEventElapsedTime")]
    public static extern CudaResult cuEventElapsedTime(out float pMilliseconds, IntPtr hStart, IntPtr hEnd);

    // Async memory transfer APIs
    [DllImport(CudaLibrary, EntryPoint = "cuMemcpyHtoDAsync_v2")]
    public static extern CudaResult cuMemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, ulong byteCount, IntPtr stream);

    [DllImport(CudaLibrary, EntryPoint = "cuMemcpyDtoHAsync_v2")]
    public static extern CudaResult cuMemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, ulong byteCount, IntPtr stream);

    [DllImport(CudaLibrary, EntryPoint = "cuMemcpyDtoDAsync_v2")]
    public static extern CudaResult cuMemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, ulong byteCount, IntPtr stream);

    // Event flag constants
    public const uint CU_EVENT_DEFAULT = 0;
    public const uint CU_EVENT_BLOCKING_SYNC = 1;
    public const uint CU_EVENT_DISABLE_TIMING = 2;
    public const uint CU_EVENT_INTERPROCESS = 4;

    // Stream flag constants
    public const uint CU_STREAM_DEFAULT = 0;
    public const uint CU_STREAM_NON_BLOCKING = 1;
}
