// Copyright (c) AiDotNet. All rights reserved.
// Pure P/Invoke OpenCL context - NO ILGPU dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// OpenCL context wrapper using pure P/Invoke. No ILGPU dependency.
    /// </summary>
    internal sealed class DirectOpenClContext : IDisposable
    {
        private IntPtr _context;
        private IntPtr _commandQueue;
        private IntPtr _profilingCommandQueue;
        private IntPtr _device;
        private IntPtr _platform;
        private bool _disposed;

        public IntPtr Context => _context;
        public IntPtr CommandQueue => _commandQueue;

        /// <summary>
        /// Gets a profiling-enabled command queue for performance measurements.
        /// Uses CL_QUEUE_PROFILING_ENABLE to allow clGetEventProfilingInfo.
        /// </summary>
        public IntPtr ProfilingCommandQueue => _profilingCommandQueue;

        /// <summary>
        /// Gets whether profiling is enabled on this context.
        /// </summary>
        public bool IsProfilingEnabled => _profilingCommandQueue != IntPtr.Zero;

        public IntPtr Device => _device;

        public string DeviceName { get; private set; } = string.Empty;
        public string DeviceVendor { get; private set; } = string.Empty;
        public string DeviceBoardName { get; private set; } = string.Empty;
        public string DriverVersion { get; private set; } = string.Empty;
        public string OpenClVersion { get; private set; } = string.Empty;
        public ulong DeviceType { get; private set; }
        public uint MaxComputeUnits { get; private set; }
        public ulong GlobalMemSize { get; private set; }
        public ulong LocalMemSize { get; private set; }

        /// <summary>
        /// GPU memory bandwidth in bytes/second (approximate).
        /// </summary>
        public ulong MemoryBandwidth { get; private set; }

        /// <summary>
        /// GPU clock frequency in MHz.
        /// </summary>
        public uint ClockFrequencyMHz { get; private set; }

        /// <summary>
        /// Maximum total work items per work group (e.g., 256, 1024).
        /// </summary>
        public ulong MaxWorkGroupSize { get; private set; }

        /// <summary>
        /// Maximum number of work item dimensions (typically 3).
        /// </summary>
        public uint MaxWorkItemDimensions { get; private set; }

        /// <summary>
        /// Maximum work items per dimension (e.g., [1024, 1024, 64] or [256, 256, 256]).
        /// </summary>
        public ulong[] MaxWorkItemSizes { get; private set; } = Array.Empty<ulong>();

        /// <summary>
        /// Device extensions string (for capability detection).
        /// </summary>
        public string Extensions { get; private set; } = string.Empty;

        /// <summary>
        /// Whether cl_khr_fp16 (half precision) is supported.
        /// </summary>
        public bool SupportsFp16 { get; private set; }

        /// <summary>
        /// Whether cl_khr_subgroups is supported.
        /// </summary>
        public bool SupportsSubgroups { get; private set; }

        /// <summary>
        /// Gets whether OpenCL is available on this system.
        /// </summary>
        public static bool IsAvailable => OpenClNativeBindings.IsAvailable;

        public DirectOpenClContext()
        {
            Initialize();
        }

        private void Initialize()
        {
            // Get platforms
            int err = OpenClNativeBindings.GetPlatformIDs(0, null, out uint numPlatforms);
            if (err != OpenClNativeBindings.CL_SUCCESS || numPlatforms == 0)
                throw new InvalidOperationException("No OpenCL platforms found");

            var platforms = new IntPtr[numPlatforms];
            err = OpenClNativeBindings.GetPlatformIDs(numPlatforms, platforms, out _);
            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to get OpenCL platforms: {err}");

            // Find a GPU device
            foreach (var platform in platforms)
            {
                err = OpenClNativeBindings.GetDeviceIDs(platform, OpenClNativeBindings.CL_DEVICE_TYPE_GPU, 0, null, out uint numDevices);
                if (err == OpenClNativeBindings.CL_SUCCESS && numDevices > 0)
                {
                    var devices = new IntPtr[numDevices];
                    err = OpenClNativeBindings.GetDeviceIDs(platform, OpenClNativeBindings.CL_DEVICE_TYPE_GPU, numDevices, devices, out _);
                    if (err == OpenClNativeBindings.CL_SUCCESS)
                    {
                        _platform = platform;
                        _device = devices[0]; // Use first GPU
                        break;
                    }
                }
            }

            if (_device == IntPtr.Zero)
                throw new InvalidOperationException("No GPU devices found");

            // Get device info
            DeviceName = OpenClNativeBindings.GetDeviceInfoString(_device, OpenClNativeBindings.CL_DEVICE_NAME);
            DeviceVendor = OpenClNativeBindings.GetDeviceInfoString(_device, OpenClNativeBindings.CL_DEVICE_VENDOR);
            DeviceType = OpenClNativeBindings.GetDeviceInfoULong(_device, OpenClNativeBindings.CL_DEVICE_TYPE);
            DriverVersion = OpenClNativeBindings.GetDeviceInfoString(_device, OpenClNativeBindings.CL_DEVICE_DRIVER_VERSION);
            OpenClVersion = OpenClNativeBindings.GetDeviceInfoString(_device, OpenClNativeBindings.CL_DEVICE_VERSION);
            MaxComputeUnits = OpenClNativeBindings.GetDeviceInfoUInt(_device, OpenClNativeBindings.CL_DEVICE_MAX_COMPUTE_UNITS);
            GlobalMemSize = OpenClNativeBindings.GetDeviceInfoULong(_device, OpenClNativeBindings.CL_DEVICE_GLOBAL_MEM_SIZE);
            LocalMemSize = OpenClNativeBindings.GetDeviceInfoULong(_device, OpenClNativeBindings.CL_DEVICE_LOCAL_MEM_SIZE);

            // Get work group capabilities
            MaxWorkGroupSize = (ulong)OpenClNativeBindings.GetDeviceInfoSizeT(_device, OpenClNativeBindings.CL_DEVICE_MAX_WORK_GROUP_SIZE);
            MaxWorkItemDimensions = OpenClNativeBindings.GetDeviceInfoUInt(_device, OpenClNativeBindings.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);

            // Get max work item sizes per dimension
            if (MaxWorkItemDimensions > 0)
            {
                var sizes = OpenClNativeBindings.GetDeviceInfoSizeTArray(_device, OpenClNativeBindings.CL_DEVICE_MAX_WORK_ITEM_SIZES, (int)MaxWorkItemDimensions);
                MaxWorkItemSizes = new ulong[sizes.Length];
                for (int i = 0; i < sizes.Length; i++)
                {
                    MaxWorkItemSizes[i] = (ulong)sizes[i];
                }
            }

            // Get extensions for capability detection
            Extensions = OpenClNativeBindings.GetDeviceInfoString(_device, OpenClNativeBindings.CL_DEVICE_EXTENSIONS);
            SupportsFp16 = Extensions.Contains("cl_khr_fp16");
            SupportsSubgroups = Extensions.Contains("cl_khr_subgroups");
            if (Extensions.Contains("cl_amd_device_attribute_query", StringComparison.OrdinalIgnoreCase))
            {
                DeviceBoardName = OpenClNativeBindings.GetDeviceInfoString(
                    _device, OpenClNativeBindings.CL_DEVICE_BOARD_NAME_AMD);
            }

            // Get clock frequency for theoretical GFLOPS calculation
            ClockFrequencyMHz = OpenClNativeBindings.GetDeviceInfoUInt(_device, OpenClNativeBindings.CL_DEVICE_MAX_CLOCK_FREQUENCY);

            // Create context
            var deviceArray = new IntPtr[] { _device };
            _context = OpenClNativeBindings.CreateContext(IntPtr.Zero, 1, deviceArray, IntPtr.Zero, IntPtr.Zero, out err);
            if (err != OpenClNativeBindings.CL_SUCCESS || _context == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL context: {err}");

            // Create command queue (without profiling for normal operations)
            _commandQueue = OpenClNativeBindings.CreateCommandQueue(_context, _device, 0, out err);
            if (err != OpenClNativeBindings.CL_SUCCESS || _commandQueue == IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseContext(_context);
                throw new InvalidOperationException($"Failed to create command queue: {err}");
            }

            // Create profiling-enabled command queue for diagnostics
            // This queue has overhead but provides accurate GPU timestamps
            _profilingCommandQueue = OpenClNativeBindings.CreateCommandQueue(
                _context, _device, OpenClNativeBindings.CL_QUEUE_PROFILING_ENABLE, out err);
            if (err != OpenClNativeBindings.CL_SUCCESS)
            {
                // Profiling queue is optional - continue without it
                _profilingCommandQueue = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Waits for all enqueued commands to complete.
        /// </summary>
        public void Finish()
        {
            if (_commandQueue != IntPtr.Zero)
            {
                OpenClNativeBindings.Finish(_commandQueue);
            }
        }

        /// <summary>
        /// Waits for all commands on the profiling queue to complete.
        /// </summary>
        public void FinishProfiling()
        {
            if (_profilingCommandQueue != IntPtr.Zero)
            {
                OpenClNativeBindings.Finish(_profilingCommandQueue);
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            if (_profilingCommandQueue != IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseCommandQueue(_profilingCommandQueue);
                _profilingCommandQueue = IntPtr.Zero;
            }

            if (_commandQueue != IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseCommandQueue(_commandQueue);
                _commandQueue = IntPtr.Zero;
            }

            if (_context != IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseContext(_context);
                _context = IntPtr.Zero;
            }

            _disposed = true;
        }
    }
}
