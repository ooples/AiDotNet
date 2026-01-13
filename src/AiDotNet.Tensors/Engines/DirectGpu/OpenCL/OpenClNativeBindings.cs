// Copyright (c) AiDotNet. All rights reserved.
// Pure P/Invoke OpenCL bindings - no managed GPU runtime dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// Pure P/Invoke bindings for OpenCL. No managed GPU runtime dependencies.
    /// This allows DirectGpu to work on .NET Framework 4.6.2+.
    /// </summary>
    internal static class OpenClNativeBindings
    {
        // OpenCL library name varies by platform
        private const string OpenClLibrary = "OpenCL";

        #region Error Codes

        public const int CL_SUCCESS = 0;
        public const int CL_DEVICE_NOT_FOUND = -1;
        public const int CL_BUILD_PROGRAM_FAILURE = -11;
        public const int CL_OUT_OF_HOST_MEMORY = -6;
        public const int CL_INVALID_VALUE = -30;

        #endregion

        #region Device Types

        public const ulong CL_DEVICE_TYPE_CPU = 2;
        public const ulong CL_DEVICE_TYPE_GPU = 4;
        public const ulong CL_DEVICE_TYPE_ALL = 0xFFFFFFFF;

        #endregion

        #region Memory Flags

        public const ulong CL_MEM_READ_WRITE = 1;
        public const ulong CL_MEM_WRITE_ONLY = 2;
        public const ulong CL_MEM_READ_ONLY = 4;
        public const ulong CL_MEM_COPY_HOST_PTR = 32;

        #endregion

        #region Command Queue Properties

        /// <summary>
        /// Enable profiling on a command queue.
        /// Required for clGetEventProfilingInfo to work.
        /// </summary>
        public const ulong CL_QUEUE_PROFILING_ENABLE = 2;

        #endregion

        #region Event Profiling Info

        /// <summary>
        /// Timestamp when command was queued.
        /// </summary>
        public const uint CL_PROFILING_COMMAND_QUEUED = 0x1280;

        /// <summary>
        /// Timestamp when command was submitted to device.
        /// </summary>
        public const uint CL_PROFILING_COMMAND_SUBMIT = 0x1281;

        /// <summary>
        /// Timestamp when command started execution on device.
        /// </summary>
        public const uint CL_PROFILING_COMMAND_START = 0x1282;

        /// <summary>
        /// Timestamp when command finished execution on device.
        /// </summary>
        public const uint CL_PROFILING_COMMAND_END = 0x1283;

        /// <summary>
        /// Timestamp when command was completed (OpenCL 2.0+).
        /// </summary>
        public const uint CL_PROFILING_COMMAND_COMPLETE = 0x1284;

        #endregion

        #region Device Info

        public const uint CL_DEVICE_TYPE = 0x1000;
        public const uint CL_DEVICE_NAME = 0x102B;
        public const uint CL_DEVICE_VENDOR = 0x102C;
        public const uint CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
        public const uint CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
        public const uint CL_DEVICE_LOCAL_MEM_SIZE = 0x1023;
        public const uint CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
        public const uint CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003;
        public const uint CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005;
        public const uint CL_DEVICE_DRIVER_VERSION = 0x102D;
        public const uint CL_DEVICE_VERSION = 0x102F;
        public const uint CL_DEVICE_EXTENSIONS = 0x1030;
        public const uint CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C;
        public const uint CL_DEVICE_BOARD_NAME_AMD = 0x4038;
        public const uint CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV = 0x4000;
        public const uint CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV = 0x4001;

        #endregion

        #region Kernel Info

        public const uint CL_KERNEL_WORK_GROUP_SIZE = 0x11B0;
        public const uint CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11B3;

        #endregion

        #region Platform Functions

        [DllImport(OpenClLibrary, EntryPoint = "clGetPlatformIDs")]
        public static extern int GetPlatformIDs(
            uint numEntries,
            [Out] IntPtr[]? platforms,
            out uint numPlatforms);

        #endregion

        #region Device Functions

        [DllImport(OpenClLibrary, EntryPoint = "clGetDeviceIDs")]
        public static extern int GetDeviceIDs(
            IntPtr platform,
            ulong deviceType,
            uint numEntries,
            [Out] IntPtr[]? devices,
            out uint numDevices);

        [DllImport(OpenClLibrary, EntryPoint = "clGetDeviceInfo")]
        public static extern int GetDeviceInfo(
            IntPtr device,
            uint paramName,
            UIntPtr paramValueSize,
            IntPtr paramValue,
            out UIntPtr paramValueSizeRet);

        #endregion

        #region Context Functions

        [DllImport(OpenClLibrary, EntryPoint = "clCreateContext")]
        public static extern IntPtr CreateContext(
            IntPtr properties,
            uint numDevices,
            [In] IntPtr[] devices,
            IntPtr pfnNotify,
            IntPtr userData,
            out int errcode);

        [DllImport(OpenClLibrary, EntryPoint = "clReleaseContext")]
        public static extern int ReleaseContext(IntPtr context);

        #endregion

        #region Command Queue Functions

        [DllImport(OpenClLibrary, EntryPoint = "clCreateCommandQueue")]
        public static extern IntPtr CreateCommandQueue(
            IntPtr context,
            IntPtr device,
            ulong properties,
            out int errcode);

        [DllImport(OpenClLibrary, EntryPoint = "clReleaseCommandQueue")]
        public static extern int ReleaseCommandQueue(IntPtr commandQueue);

        [DllImport(OpenClLibrary, EntryPoint = "clFinish")]
        public static extern int Finish(IntPtr commandQueue);

        [DllImport(OpenClLibrary, EntryPoint = "clFlush")]
        public static extern int Flush(IntPtr commandQueue);

        #endregion

        #region Event Functions

        [DllImport(OpenClLibrary, EntryPoint = "clWaitForEvents")]
        public static extern int WaitForEvents(
            uint numEvents,
            [In] IntPtr[] eventList);

        [DllImport(OpenClLibrary, EntryPoint = "clGetEventProfilingInfo")]
        public static extern int GetEventProfilingInfo(
            IntPtr eventHandle,
            uint paramName,
            UIntPtr paramValueSize,
            IntPtr paramValue,
            out UIntPtr paramValueSizeRet);

        [DllImport(OpenClLibrary, EntryPoint = "clReleaseEvent")]
        public static extern int ReleaseEvent(IntPtr eventHandle);

        [DllImport(OpenClLibrary, EntryPoint = "clRetainEvent")]
        public static extern int RetainEvent(IntPtr eventHandle);

        [DllImport(OpenClLibrary, EntryPoint = "clGetEventInfo")]
        public static extern int GetEventInfo(
            IntPtr eventHandle,
            uint paramName,
            UIntPtr paramValueSize,
            IntPtr paramValue,
            out UIntPtr paramValueSizeRet);

        [DllImport(OpenClLibrary, EntryPoint = "clEnqueueMarkerWithWaitList")]
        public static extern int EnqueueMarkerWithWaitList(
            IntPtr commandQueue,
            uint numEventsInWaitList,
            [In] IntPtr[]? eventWaitList,
            out IntPtr eventOut);

        // Event info constants
        public const uint CL_EVENT_COMMAND_EXECUTION_STATUS = 0x11D3;

        // Event execution status values
        public const int CL_QUEUED = 3;
        public const int CL_SUBMITTED = 2;
        public const int CL_RUNNING = 1;
        public const int CL_COMPLETE = 0;

        #endregion

        #region Memory Functions

        [DllImport(OpenClLibrary, EntryPoint = "clCreateBuffer")]
        public static extern IntPtr CreateBuffer(
            IntPtr context,
            ulong flags,
            UIntPtr size,
            IntPtr hostPtr,
            out int errcode);

        [DllImport(OpenClLibrary, EntryPoint = "clReleaseMemObject")]
        public static extern int ReleaseMemObject(IntPtr memObj);

        [DllImport(OpenClLibrary, EntryPoint = "clEnqueueReadBuffer")]
        public static extern int EnqueueReadBuffer(
            IntPtr commandQueue,
            IntPtr buffer,
            uint blockingRead,
            UIntPtr offset,
            UIntPtr size,
            IntPtr ptr,
            uint numEventsInWaitList,
            IntPtr eventWaitList,
            IntPtr eventOut);

        [DllImport(OpenClLibrary, EntryPoint = "clEnqueueWriteBuffer")]
        public static extern int EnqueueWriteBuffer(
            IntPtr commandQueue,
            IntPtr buffer,
            uint blockingWrite,
            UIntPtr offset,
            UIntPtr size,
            IntPtr ptr,
            uint numEventsInWaitList,
            IntPtr eventWaitList,
            IntPtr eventOut);

        [DllImport(OpenClLibrary, EntryPoint = "clEnqueueCopyBuffer")]
        public static extern int EnqueueCopyBuffer(
            IntPtr commandQueue,
            IntPtr srcBuffer,
            IntPtr dstBuffer,
            UIntPtr srcOffset,
            UIntPtr dstOffset,
            UIntPtr size,
            uint numEventsInWaitList,
            IntPtr eventWaitList,
            IntPtr eventOut);

        #endregion

        #region Program Functions

        [DllImport(OpenClLibrary, EntryPoint = "clCreateProgramWithSource")]
        public static extern IntPtr CreateProgramWithSource(
            IntPtr context,
            uint count,
            [In] string[] strings,
            [In] UIntPtr[] lengths,
            out int errcode);

        [DllImport(OpenClLibrary, EntryPoint = "clBuildProgram")]
        public static extern int BuildProgram(
            IntPtr program,
            uint numDevices,
            [In] IntPtr[] deviceList,
            string options,
            IntPtr pfnNotify,
            IntPtr userData);

        [DllImport(OpenClLibrary, EntryPoint = "clGetProgramBuildInfo")]
        public static extern int GetProgramBuildInfo(
            IntPtr program,
            IntPtr device,
            uint paramName,
            UIntPtr paramValueSize,
            IntPtr paramValue,
            out UIntPtr paramValueSizeRet);

        [DllImport(OpenClLibrary, EntryPoint = "clReleaseProgram")]
        public static extern int ReleaseProgram(IntPtr program);

        // Build info constants
        public const uint CL_PROGRAM_BUILD_LOG = 0x1183;

        #endregion

        #region Kernel Functions

        [DllImport(OpenClLibrary, EntryPoint = "clCreateKernel")]
        public static extern IntPtr CreateKernel(
            IntPtr program,
            string kernelName,
            out int errcode);

        [DllImport(OpenClLibrary, EntryPoint = "clReleaseKernel")]
        public static extern int ReleaseKernel(IntPtr kernel);

        [DllImport(OpenClLibrary, EntryPoint = "clSetKernelArg")]
        public static extern int SetKernelArg(
            IntPtr kernel,
            uint argIndex,
            UIntPtr argSize,
            IntPtr argValue);

        [DllImport(OpenClLibrary, EntryPoint = "clEnqueueNDRangeKernel")]
        public static extern int EnqueueNDRangeKernel(
            IntPtr commandQueue,
            IntPtr kernel,
            uint workDim,
            [In] UIntPtr[]? globalWorkOffset,
            [In] UIntPtr[] globalWorkSize,
            [In] UIntPtr[] localWorkSize,
            uint numEventsInWaitList,
            IntPtr eventWaitList,
            IntPtr eventOut);

        [DllImport(OpenClLibrary, EntryPoint = "clGetKernelWorkGroupInfo")]
        public static extern int GetKernelWorkGroupInfo(
            IntPtr kernel,
            IntPtr device,
            uint paramName,
            UIntPtr paramValueSize,
            IntPtr paramValue,
            out UIntPtr paramValueSizeRet);

        #endregion

        #region Helper Methods

        /// <summary>
        /// Prints diagnostic information about OpenCL DLL search paths.
        /// Call this to debug DLL loading issues.
        /// </summary>
        public static void PrintDllSearchDiagnostics()
        {
            if (Environment.GetEnvironmentVariable("AIDOTNET_OPENCL_DIAGNOSTICS") != "1")
                return;
            Console.WriteLine("[OpenCL DLL Diagnostics] Searching for OpenCL.dll...");

            // Check common Windows locations for OpenCL.dll
            var searchPaths = new[]
            {
                System.IO.Path.Combine(Environment.SystemDirectory, "OpenCL.dll"),  // C:\Windows\System32\OpenCL.dll
                System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), "SysWOW64", "OpenCL.dll"),  // C:\Windows\SysWOW64\OpenCL.dll
                System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "OpenCL.dll"),  // Application directory
            };

            foreach (var path in searchPaths)
            {
                bool exists = System.IO.File.Exists(path);
                Console.WriteLine($"[OpenCL DLL Diagnostics] {path} - {(exists ? "FOUND" : "not found")}");
            }

            // Check PATH environment variable
            var pathEnv = Environment.GetEnvironmentVariable("PATH") ?? "";
            Console.WriteLine("[OpenCL DLL Diagnostics] Checking PATH directories...");

            var pathDirs = pathEnv.Split(';');
            bool foundInPath = false;
            foreach (var dir in pathDirs)
            {
                if (string.IsNullOrWhiteSpace(dir)) continue;
                try
                {
                    var openClPath = System.IO.Path.Combine(dir.Trim(), "OpenCL.dll");
                    if (System.IO.File.Exists(openClPath))
                    {
                        Console.WriteLine($"[OpenCL DLL Diagnostics] Found in PATH: {openClPath}");
                        foundInPath = true;
                    }
                }
                catch
                {
                    // Skip invalid paths
                }
            }

            if (!foundInPath)
            {
                Console.WriteLine("[OpenCL DLL Diagnostics] OpenCL.dll NOT found in any PATH directory");
            }

            Console.WriteLine("[OpenCL DLL Diagnostics] If OpenCL.dll is missing, install GPU drivers:");
            Console.WriteLine("[OpenCL DLL Diagnostics]   - NVIDIA: Install GeForce or CUDA drivers");
            Console.WriteLine("[OpenCL DLL Diagnostics]   - AMD: Install Adrenalin drivers");
            Console.WriteLine("[OpenCL DLL Diagnostics]   - Intel: Install Intel Graphics drivers or OpenCL Runtime");
        }

        /// <summary>
        /// Gets a string device info value.
        /// </summary>
        public static string GetDeviceInfoString(IntPtr device, uint paramName)
        {
            // First get the size
            int err = GetDeviceInfo(device, paramName, UIntPtr.Zero, IntPtr.Zero, out UIntPtr size);
            if (err != CL_SUCCESS || size == UIntPtr.Zero)
                return string.Empty;

            // Allocate and get the value
            IntPtr buffer = Marshal.AllocHGlobal((int)size);
            try
            {
                err = GetDeviceInfo(device, paramName, size, buffer, out _);
                if (err != CL_SUCCESS)
                    return string.Empty;

                return Marshal.PtrToStringAnsi(buffer) ?? string.Empty;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        /// <summary>
        /// Gets a ulong device info value.
        /// </summary>
        public static ulong GetDeviceInfoULong(IntPtr device, uint paramName)
        {
            IntPtr buffer = Marshal.AllocHGlobal(8);
            try
            {
                int err = GetDeviceInfo(device, paramName, (UIntPtr)8, buffer, out _);
                if (err != CL_SUCCESS)
                    return 0;

                return (ulong)Marshal.ReadInt64(buffer);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        /// <summary>
        /// Gets a uint device info value.
        /// </summary>
        public static uint GetDeviceInfoUInt(IntPtr device, uint paramName)
        {
            IntPtr buffer = Marshal.AllocHGlobal(4);
            try
            {
                int err = GetDeviceInfo(device, paramName, (UIntPtr)4, buffer, out _);
                if (err != CL_SUCCESS)
                    return 0;

                return (uint)Marshal.ReadInt32(buffer);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        /// <summary>
        /// Gets a size_t (UIntPtr) device info value.
        /// </summary>
        public static UIntPtr GetDeviceInfoSizeT(IntPtr device, uint paramName)
        {
            int ptrSize = IntPtr.Size;
            IntPtr buffer = Marshal.AllocHGlobal(ptrSize);
            try
            {
                int err = GetDeviceInfo(device, paramName, (UIntPtr)ptrSize, buffer, out _);
                if (err != CL_SUCCESS)
                    return UIntPtr.Zero;

                if (ptrSize == 8)
                    return (UIntPtr)(ulong)Marshal.ReadInt64(buffer);
                else
                    return (UIntPtr)(uint)Marshal.ReadInt32(buffer);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        /// <summary>
        /// Gets an array of size_t device info values (e.g., max work item sizes).
        /// </summary>
        public static UIntPtr[] GetDeviceInfoSizeTArray(IntPtr device, uint paramName, int count)
        {
            int ptrSize = IntPtr.Size;
            int bufferSize = ptrSize * count;
            IntPtr buffer = Marshal.AllocHGlobal(bufferSize);
            try
            {
                int err = GetDeviceInfo(device, paramName, (UIntPtr)bufferSize, buffer, out _);
                if (err != CL_SUCCESS)
                    return new UIntPtr[count];

                var result = new UIntPtr[count];
                for (int i = 0; i < count; i++)
                {
                    if (ptrSize == 8)
                        result[i] = (UIntPtr)(ulong)Marshal.ReadInt64(buffer, i * ptrSize);
                    else
                        result[i] = (UIntPtr)(uint)Marshal.ReadInt32(buffer, i * ptrSize);
                }
                return result;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        /// <summary>
        /// Gets a size_t kernel work group info value.
        /// </summary>
        public static UIntPtr GetKernelWorkGroupInfoSizeT(IntPtr kernel, IntPtr device, uint paramName)
        {
            int ptrSize = IntPtr.Size;
            IntPtr buffer = Marshal.AllocHGlobal(ptrSize);
            try
            {
                int err = GetKernelWorkGroupInfo(kernel, device, paramName, (UIntPtr)ptrSize, buffer, out _);
                if (err != CL_SUCCESS)
                    return UIntPtr.Zero;

                if (ptrSize == 8)
                    return (UIntPtr)(ulong)Marshal.ReadInt64(buffer);
                else
                    return (UIntPtr)(uint)Marshal.ReadInt32(buffer);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        /// <summary>
        /// Gets the build log for a program.
        /// </summary>
        public static string GetBuildLog(IntPtr program, IntPtr device)
        {
            // First get the size
            int err = GetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, UIntPtr.Zero, IntPtr.Zero, out UIntPtr size);
            if (err != CL_SUCCESS || size == UIntPtr.Zero)
                return string.Empty;

            // Allocate and get the log
            IntPtr buffer = Marshal.AllocHGlobal((int)size);
            try
            {
                err = GetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, buffer, out _);
                if (err != CL_SUCCESS)
                    return string.Empty;

                return Marshal.PtrToStringAnsi(buffer) ?? string.Empty;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        /// <summary>
        /// Checks if OpenCL is available on this system.
        /// </summary>
        public static bool IsAvailable
        {
            get
            {
                bool diagnosticsEnabled = Environment.GetEnvironmentVariable("AIDOTNET_OPENCL_DIAGNOSTICS") == "1";
                try
                {
                    int err = GetPlatformIDs(0, null, out uint numPlatforms);
                    bool available = err == CL_SUCCESS && numPlatforms > 0;
                    if (diagnosticsEnabled)
                    {
                        Console.WriteLine($"[OpenCL Diagnostics] GetPlatformIDs returned error code: {err}, platforms found: {numPlatforms}, available: {available}");
                    }
                    return available;
                }
                catch (DllNotFoundException ex)
                {
                    if (diagnosticsEnabled)
                    {
                        Console.WriteLine($"[OpenCL Diagnostics] DllNotFoundException: {ex.Message}");
                        PrintDllSearchDiagnostics();
                    }
                    return false;
                }
                catch (Exception ex)
                {
                    if (diagnosticsEnabled)
                    {
                        Console.WriteLine($"[OpenCL Diagnostics] Exception during OpenCL availability check: {ex.GetType().Name}: {ex.Message}");
                    }
                    return false;
                }
            }
        }

        /// <summary>
        /// Gets a ulong profiling info value from an event.
        /// Timestamps are in nanoseconds.
        /// </summary>
        public static ulong GetEventProfilingInfoULong(IntPtr eventHandle, uint paramName)
        {
            IntPtr buffer = Marshal.AllocHGlobal(8);
            try
            {
                int err = GetEventProfilingInfo(eventHandle, paramName, (UIntPtr)8, buffer, out _);
                if (err != CL_SUCCESS)
                    return 0;

                return (ulong)Marshal.ReadInt64(buffer);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        #endregion
    }
}
