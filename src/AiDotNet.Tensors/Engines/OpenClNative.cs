#if !NET462
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// OpenCL P/Invoke bindings for cross-platform GPU acceleration.
/// Supports AMD, Intel, and NVIDIA GPUs through the OpenCL API.
/// </summary>
/// <remarks>
/// <para><b>Performance Targets:</b></para>
/// <list type="bullet">
/// <item>AMD RX 7900 XTX: ~25,000 GFLOPS (FP32)</item>
/// <item>Intel Arc A770: ~18,000 GFLOPS (FP32)</item>
/// <item>NVIDIA RTX 4090: ~30,000 GFLOPS (OpenCL, ~82,000 via CUDA)</item>
/// </list>
/// <para><b>Why OpenCL:</b></para>
/// <list type="bullet">
/// <item>Cross-vendor: Works on AMD, Intel, NVIDIA, Apple, ARM Mali, Qualcomm Adreno</item>
/// <item>No vendor lock-in: Not tied to NVIDIA's ecosystem</item>
/// <item>Mature ecosystem: CLBlast, clBLAS provide cuBLAS-like performance</item>
/// </list>
/// </remarks>
public static class OpenClNative
{
    // OpenCL library - standard name across platforms
    private const string OpenClLibrary = "OpenCL";

    #region Enums and Constants

    /// <summary>OpenCL error codes.</summary>
    public enum ClError : int
    {
        Success = 0,
        DeviceNotFound = -1,
        DeviceNotAvailable = -2,
        CompilerNotAvailable = -3,
        MemObjectAllocationFailure = -4,
        OutOfResources = -5,
        OutOfHostMemory = -6,
        ProfilingInfoNotAvailable = -7,
        MemCopyOverlap = -8,
        ImageFormatMismatch = -9,
        ImageFormatNotSupported = -10,
        BuildProgramFailure = -11,
        MapFailure = -12,
        MisalignedSubBufferOffset = -13,
        ExecStatusErrorForEventsInWaitList = -14,
        CompileProgramFailure = -15,
        LinkerNotAvailable = -16,
        LinkProgramFailure = -17,
        DevicePartitionFailed = -18,
        KernelArgInfoNotAvailable = -19,

        InvalidValue = -30,
        InvalidDeviceType = -31,
        InvalidPlatform = -32,
        InvalidDevice = -33,
        InvalidContext = -34,
        InvalidQueueProperties = -35,
        InvalidCommandQueue = -36,
        InvalidHostPtr = -37,
        InvalidMemObject = -38,
        InvalidImageFormatDescriptor = -39,
        InvalidImageSize = -40,
        InvalidSampler = -41,
        InvalidBinary = -42,
        InvalidBuildOptions = -43,
        InvalidProgram = -44,
        InvalidProgramExecutable = -45,
        InvalidKernelName = -46,
        InvalidKernelDefinition = -47,
        InvalidKernel = -48,
        InvalidArgIndex = -49,
        InvalidArgValue = -50,
        InvalidArgSize = -51,
        InvalidKernelArgs = -52,
        InvalidWorkDimension = -53,
        InvalidWorkGroupSize = -54,
        InvalidWorkItemSize = -55,
        InvalidGlobalOffset = -56,
        InvalidEventWaitList = -57,
        InvalidEvent = -58,
        InvalidOperation = -59,
        InvalidGlObject = -60,
        InvalidBufferSize = -61,
        InvalidMipLevel = -62,
        InvalidGlobalWorkSize = -63,
        InvalidProperty = -64,
        InvalidImageDescriptor = -65,
        InvalidCompilerOptions = -66,
        InvalidLinkerOptions = -67,
        InvalidDevicePartitionCount = -68,
        InvalidPipeSize = -69,
        InvalidDeviceQueue = -70,
        InvalidSpecId = -71,
        MaxSizeRestrictionExceeded = -72
    }

    /// <summary>OpenCL device types.</summary>
    [Flags]
    public enum ClDeviceType : ulong
    {
        Default = 1 << 0,
        Cpu = 1 << 1,
        Gpu = 1 << 2,
        Accelerator = 1 << 3,
        Custom = 1 << 4,
        All = 0xFFFFFFFF
    }

    /// <summary>OpenCL memory flags.</summary>
    [Flags]
    public enum ClMemFlags : ulong
    {
        ReadWrite = 1 << 0,
        WriteOnly = 1 << 1,
        ReadOnly = 1 << 2,
        UseHostPtr = 1 << 3,
        AllocHostPtr = 1 << 4,
        CopyHostPtr = 1 << 5,
        HostWriteOnly = 1 << 7,
        HostReadOnly = 1 << 8,
        HostNoAccess = 1 << 9,
        SvmFineGrainBuffer = 1 << 10,
        SvmAtomics = 1 << 11,
        KernelReadAndWrite = 1 << 12
    }

    /// <summary>OpenCL command queue properties.</summary>
    [Flags]
    public enum ClCommandQueueProperties : ulong
    {
        None = 0,
        OutOfOrderExecModeEnable = 1 << 0,
        ProfilingEnable = 1 << 1,
        OnDevice = 1 << 2,
        OnDeviceDefault = 1 << 3
    }

    /// <summary>OpenCL platform info.</summary>
    public enum ClPlatformInfo : uint
    {
        Profile = 0x0900,
        Version = 0x0901,
        Name = 0x0902,
        Vendor = 0x0903,
        Extensions = 0x0904,
        HostTimerResolution = 0x0905,
        NumericVersion = 0x0906,
        ExtensionsWithVersion = 0x0907
    }

    /// <summary>OpenCL device info.</summary>
    public enum ClDeviceInfo : uint
    {
        Type = 0x1000,
        VendorId = 0x1001,
        MaxComputeUnits = 0x1002,
        MaxWorkItemDimensions = 0x1003,
        MaxWorkGroupSize = 0x1004,
        MaxWorkItemSizes = 0x1005,
        PreferredVectorWidthChar = 0x1006,
        PreferredVectorWidthShort = 0x1007,
        PreferredVectorWidthInt = 0x1008,
        PreferredVectorWidthLong = 0x1009,
        PreferredVectorWidthFloat = 0x100A,
        PreferredVectorWidthDouble = 0x100B,
        MaxClockFrequency = 0x100C,
        AddressBits = 0x100D,
        MaxReadImageArgs = 0x100E,
        MaxWriteImageArgs = 0x100F,
        MaxMemAllocSize = 0x1010,
        Image2DMaxWidth = 0x1011,
        Image2DMaxHeight = 0x1012,
        Image3DMaxWidth = 0x1013,
        Image3DMaxHeight = 0x1014,
        Image3DMaxDepth = 0x1015,
        ImageSupport = 0x1016,
        MaxParameterSize = 0x1017,
        MaxSamplers = 0x1018,
        MemBaseAddrAlign = 0x1019,
        MinDataTypeAlignSize = 0x101A,
        SingleFpConfig = 0x101B,
        GlobalMemCacheType = 0x101C,
        GlobalMemCachelineSize = 0x101D,
        GlobalMemCacheSize = 0x101E,
        GlobalMemSize = 0x101F,
        MaxConstantBufferSize = 0x1020,
        MaxConstantArgs = 0x1021,
        LocalMemType = 0x1022,
        LocalMemSize = 0x1023,
        ErrorCorrectionSupport = 0x1024,
        ProfilingTimerResolution = 0x1025,
        EndianLittle = 0x1026,
        Available = 0x1027,
        CompilerAvailable = 0x1028,
        ExecutionCapabilities = 0x1029,
        QueueOnHostProperties = 0x102A,  // Deprecated in 2.0
        Name = 0x102B,
        Vendor = 0x102C,
        DriverVersion = 0x102D,
        Profile = 0x102E,
        Version = 0x102F,
        Extensions = 0x1030,
        Platform = 0x1031,
        DoubleFpConfig = 0x1032,
        HalfFpConfig = 0x1033,
        PreferredVectorWidthHalf = 0x1034,
        HostUnifiedMemory = 0x1035,  // Deprecated in 2.0
        NativeVectorWidthChar = 0x1036,
        NativeVectorWidthShort = 0x1037,
        NativeVectorWidthInt = 0x1038,
        NativeVectorWidthLong = 0x1039,
        NativeVectorWidthFloat = 0x103A,
        NativeVectorWidthDouble = 0x103B,
        NativeVectorWidthHalf = 0x103C,
        OpenClCVersion = 0x103D,
        LinkerAvailable = 0x103E,
        BuiltInKernels = 0x103F,
        ImageMaxBufferSize = 0x1040,
        ImageMaxArraySize = 0x1041,
        ParentDevice = 0x1042,
        PartitionMaxSubDevices = 0x1043,
        PartitionProperties = 0x1044,
        PartitionAffinityDomain = 0x1045,
        PartitionType = 0x1046,
        ReferenceCount = 0x1047,
        PreferredInteropUserSync = 0x1048,
        PrintfBufferSize = 0x1049,
        ImagePitchAlignment = 0x104A,
        ImageBaseAddressAlignment = 0x104B,
        MaxReadWriteImageArgs = 0x104C,
        MaxGlobalVariableSize = 0x104D,
        QueueOnDeviceProperties = 0x104E,
        QueueOnDevicePreferredSize = 0x104F,
        QueueOnDeviceMaxSize = 0x1050,
        MaxOnDeviceQueues = 0x1051,
        MaxOnDeviceEvents = 0x1052,
        SvmCapabilities = 0x1053,
        GlobalVariablePreferredTotalSize = 0x1054,
        MaxPipeArgs = 0x1055,
        PipeMaxActiveReservations = 0x1056,
        PipeMaxPacketSize = 0x1057,
        PreferredPlatformAtomicAlignment = 0x1058,
        PreferredGlobalAtomicAlignment = 0x1059,
        PreferredLocalAtomicAlignment = 0x105A,
        IlVersion = 0x105B,
        MaxNumSubGroups = 0x105C,
        SubGroupIndependentForwardProgress = 0x105D,
        NumericVersion = 0x105E,
        ExtensionsWithVersion = 0x105F,
        IlsWithVersion = 0x1060,
        BuiltInKernelsWithVersion = 0x1061,
        AtomicMemoryCapabilities = 0x1062,
        AtomicFenceCapabilities = 0x1063,
        NonUniformWorkGroupSupport = 0x1064,
        OpenClCAllVersions = 0x1065,
        PreferredWorkGroupSizeMultiple = 0x1066,
        WorkGroupCollectiveFunctionsSupport = 0x1067,
        GenericAddressSpaceSupport = 0x1068,
        OpenClCFeatures = 0x106F,
        DeviceEnqueueCapabilities = 0x1070,
        PipeSupport = 0x1071,
        LatestConformanceVersionPassed = 0x1072
    }

    /// <summary>OpenCL program build info.</summary>
    public enum ClProgramBuildInfo : uint
    {
        Status = 0x1181,
        Options = 0x1182,
        Log = 0x1183,
        BinaryType = 0x1184,
        GlobalVariableTotalSize = 0x1185
    }

    /// <summary>OpenCL program build status.</summary>
    public enum ClBuildStatus : int
    {
        Success = 0,
        None = -1,
        Error = -2,
        InProgress = -3
    }

    /// <summary>OpenCL kernel work group info.</summary>
    public enum ClKernelWorkGroupInfo : uint
    {
        WorkGroupSize = 0x11B0,
        CompileWorkGroupSize = 0x11B1,
        LocalMemSize = 0x11B2,
        PreferredWorkGroupSizeMultiple = 0x11B3,
        PrivateMemSize = 0x11B4,
        GlobalWorkSize = 0x11B5
    }

    #endregion

    #region Platform and Device Functions

    /// <summary>Gets available OpenCL platforms.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clGetPlatformIDs")]
    public static extern ClError clGetPlatformIDs(
        uint numEntries,
        [Out] IntPtr[] platforms,
        out uint numPlatforms);

    /// <summary>Gets platform info.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clGetPlatformInfo")]
    public static extern ClError clGetPlatformInfo(
        IntPtr platform,
        ClPlatformInfo paramName,
        UIntPtr paramValueSize,
        byte[] paramValue,
        out UIntPtr paramValueSizeRet);

    /// <summary>Gets available devices for a platform.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clGetDeviceIDs")]
    public static extern ClError clGetDeviceIDs(
        IntPtr platform,
        ClDeviceType deviceType,
        uint numEntries,
        [Out] IntPtr[] devices,
        out uint numDevices);

    /// <summary>Gets device info.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clGetDeviceInfo")]
    public static extern ClError clGetDeviceInfo(
        IntPtr device,
        ClDeviceInfo paramName,
        UIntPtr paramValueSize,
        byte[] paramValue,
        out UIntPtr paramValueSizeRet);

    /// <summary>Gets device info (native types).</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clGetDeviceInfo")]
    public static extern ClError clGetDeviceInfo(
        IntPtr device,
        ClDeviceInfo paramName,
        UIntPtr paramValueSize,
        out ulong paramValue,
        out UIntPtr paramValueSizeRet);

    /// <summary>Gets device info (uint value).</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clGetDeviceInfo")]
    public static extern ClError clGetDeviceInfo(
        IntPtr device,
        ClDeviceInfo paramName,
        UIntPtr paramValueSize,
        out uint paramValue,
        out UIntPtr paramValueSizeRet);

    #endregion

    #region Context Functions

    /// <summary>Creates an OpenCL context.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clCreateContext")]
    public static extern IntPtr clCreateContext(
        IntPtr[] properties,
        uint numDevices,
        IntPtr[] devices,
        IntPtr pfnNotify,
        IntPtr userData,
        out ClError errcode);

    /// <summary>Creates an OpenCL context from a device type.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clCreateContextFromType")]
    public static extern IntPtr clCreateContextFromType(
        IntPtr[] properties,
        ClDeviceType deviceType,
        IntPtr pfnNotify,
        IntPtr userData,
        out ClError errcode);

    /// <summary>Releases an OpenCL context.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clReleaseContext")]
    public static extern ClError clReleaseContext(IntPtr context);

    /// <summary>Retains (increments reference count) an OpenCL context.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clRetainContext")]
    public static extern ClError clRetainContext(IntPtr context);

    #endregion

    #region Command Queue Functions

    /// <summary>Creates a command queue (OpenCL 2.0+).</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clCreateCommandQueueWithProperties")]
    public static extern IntPtr clCreateCommandQueueWithProperties(
        IntPtr context,
        IntPtr device,
        ulong[] properties,
        out ClError errcode);

    /// <summary>Creates a command queue (OpenCL 1.x, deprecated in 2.0).</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clCreateCommandQueue")]
    public static extern IntPtr clCreateCommandQueue(
        IntPtr context,
        IntPtr device,
        ClCommandQueueProperties properties,
        out ClError errcode);

    /// <summary>Releases a command queue.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clReleaseCommandQueue")]
    public static extern ClError clReleaseCommandQueue(IntPtr commandQueue);

    /// <summary>Blocks until all commands in the queue complete.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clFinish")]
    public static extern ClError clFinish(IntPtr commandQueue);

    /// <summary>Issues all queued commands to the device.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clFlush")]
    public static extern ClError clFlush(IntPtr commandQueue);

    #endregion

    #region Memory Object Functions

    /// <summary>Creates a buffer object.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clCreateBuffer")]
    public static extern IntPtr clCreateBuffer(
        IntPtr context,
        ClMemFlags flags,
        UIntPtr size,
        IntPtr hostPtr,
        out ClError errcode);

    /// <summary>Releases a memory object.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clReleaseMemObject")]
    public static extern ClError clReleaseMemObject(IntPtr memobj);

    /// <summary>Enqueues a command to read from a buffer to host memory.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clEnqueueReadBuffer")]
    public static extern ClError clEnqueueReadBuffer(
        IntPtr commandQueue,
        IntPtr buffer,
        uint blockingRead,  // CL_TRUE = 1, CL_FALSE = 0
        UIntPtr offset,
        UIntPtr size,
        IntPtr ptr,
        uint numEventsInWaitList,
        IntPtr[] eventWaitList,
        out IntPtr evt);

    /// <summary>Enqueues a command to write from host memory to a buffer.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clEnqueueWriteBuffer")]
    public static extern ClError clEnqueueWriteBuffer(
        IntPtr commandQueue,
        IntPtr buffer,
        uint blockingWrite,
        UIntPtr offset,
        UIntPtr size,
        IntPtr ptr,
        uint numEventsInWaitList,
        IntPtr[] eventWaitList,
        out IntPtr evt);

    /// <summary>Enqueues a command to fill a buffer with a pattern.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clEnqueueFillBuffer")]
    public static extern ClError clEnqueueFillBuffer(
        IntPtr commandQueue,
        IntPtr buffer,
        IntPtr pattern,
        UIntPtr patternSize,
        UIntPtr offset,
        UIntPtr size,
        uint numEventsInWaitList,
        IntPtr[] eventWaitList,
        out IntPtr evt);

    /// <summary>Enqueues a command to copy from one buffer to another.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clEnqueueCopyBuffer")]
    public static extern ClError clEnqueueCopyBuffer(
        IntPtr commandQueue,
        IntPtr srcBuffer,
        IntPtr dstBuffer,
        UIntPtr srcOffset,
        UIntPtr dstOffset,
        UIntPtr size,
        uint numEventsInWaitList,
        IntPtr[] eventWaitList,
        out IntPtr evt);

    #endregion

    #region Program Functions

    /// <summary>Creates a program object from source code.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clCreateProgramWithSource")]
    public static extern IntPtr clCreateProgramWithSource(
        IntPtr context,
        uint count,
        string[] strings,
        UIntPtr[] lengths,
        out ClError errcode);

    /// <summary>Creates a program from IL (SPIR-V).</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clCreateProgramWithIL")]
    public static extern IntPtr clCreateProgramWithIL(
        IntPtr context,
        byte[] il,
        UIntPtr length,
        out ClError errcode);

    /// <summary>Creates a program from binary.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clCreateProgramWithBinary")]
    public static extern IntPtr clCreateProgramWithBinary(
        IntPtr context,
        uint numDevices,
        IntPtr[] deviceList,
        UIntPtr[] lengths,
        IntPtr[] binaries,
        int[] binaryStatus,
        out ClError errcode);

    /// <summary>Builds (compiles and links) a program.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clBuildProgram")]
    public static extern ClError clBuildProgram(
        IntPtr program,
        uint numDevices,
        IntPtr[] deviceList,
        string options,
        IntPtr pfnNotify,
        IntPtr userData);

    /// <summary>Gets program build info.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clGetProgramBuildInfo")]
    public static extern ClError clGetProgramBuildInfo(
        IntPtr program,
        IntPtr device,
        ClProgramBuildInfo paramName,
        UIntPtr paramValueSize,
        byte[] paramValue,
        out UIntPtr paramValueSizeRet);

    /// <summary>Releases a program object.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clReleaseProgram")]
    public static extern ClError clReleaseProgram(IntPtr program);

    #endregion

    #region Kernel Functions

    /// <summary>Creates a kernel object.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clCreateKernel")]
    public static extern IntPtr clCreateKernel(
        IntPtr program,
        string kernelName,
        out ClError errcode);

    /// <summary>Creates kernel objects for all kernels in a program.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clCreateKernelsInProgram")]
    public static extern ClError clCreateKernelsInProgram(
        IntPtr program,
        uint numKernels,
        IntPtr[] kernels,
        out uint numKernelsRet);

    /// <summary>Sets a kernel argument.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clSetKernelArg")]
    public static extern ClError clSetKernelArg(
        IntPtr kernel,
        uint argIndex,
        UIntPtr argSize,
        IntPtr argValue);

    /// <summary>Sets a kernel argument (memory object).</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clSetKernelArg")]
    public static extern ClError clSetKernelArg(
        IntPtr kernel,
        uint argIndex,
        UIntPtr argSize,
        ref IntPtr argValue);

    /// <summary>Sets a kernel argument (int value).</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clSetKernelArg")]
    public static extern ClError clSetKernelArgInt(
        IntPtr kernel,
        uint argIndex,
        UIntPtr argSize,
        ref int argValue);

    /// <summary>Sets a kernel argument (float value).</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clSetKernelArg")]
    public static extern ClError clSetKernelArgFloat(
        IntPtr kernel,
        uint argIndex,
        UIntPtr argSize,
        ref float argValue);

    /// <summary>Gets kernel work group info.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clGetKernelWorkGroupInfo")]
    public static extern ClError clGetKernelWorkGroupInfo(
        IntPtr kernel,
        IntPtr device,
        ClKernelWorkGroupInfo paramName,
        UIntPtr paramValueSize,
        out UIntPtr paramValue,
        out UIntPtr paramValueSizeRet);

    /// <summary>Releases a kernel object.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clReleaseKernel")]
    public static extern ClError clReleaseKernel(IntPtr kernel);

    /// <summary>Enqueues a command to execute a kernel on a device.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clEnqueueNDRangeKernel")]
    public static extern ClError clEnqueueNDRangeKernel(
        IntPtr commandQueue,
        IntPtr kernel,
        uint workDim,
        UIntPtr[] globalWorkOffset,
        UIntPtr[] globalWorkSize,
        UIntPtr[] localWorkSize,
        uint numEventsInWaitList,
        IntPtr[] eventWaitList,
        out IntPtr evt);

    #endregion

    #region Event Functions

    /// <summary>Waits for a list of events to complete.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clWaitForEvents")]
    public static extern ClError clWaitForEvents(
        uint numEvents,
        IntPtr[] eventList);

    /// <summary>Releases an event object.</summary>
    [DllImport(OpenClLibrary, EntryPoint = "clReleaseEvent")]
    public static extern ClError clReleaseEvent(IntPtr evt);

    #endregion
}

/// <summary>
/// High-level OpenCL context wrapper with automatic resource management.
/// Provides simplified access to OpenCL operations for neural network layers.
/// </summary>
public sealed class OpenClContext : IDisposable
{
    private IntPtr _context;
    private IntPtr _device;
    private IntPtr _platform;
    private IntPtr _commandQueue;
    private bool _disposed;
    private static bool _isAvailable;
    private static bool _checkedAvailability;
    private static readonly object _lock = new object();

    private string _deviceName = string.Empty;
    private string _deviceVendor = string.Empty;
    private uint _maxComputeUnits;
    private ulong _globalMemSize;
    private ulong _localMemSize;

    /// <summary>Gets whether OpenCL is available on this system.</summary>
    public static bool IsAvailable
    {
        get
        {
            if (!_checkedAvailability)
            {
                lock (_lock)
                {
                    if (!_checkedAvailability)
                    {
                        _isAvailable = CheckAvailability();
                        _checkedAvailability = true;
                    }
                }
            }
            return _isAvailable;
        }
    }

    /// <summary>Gets the OpenCL context handle.</summary>
    public IntPtr Handle => _context;

    /// <summary>Gets the OpenCL device handle.</summary>
    public IntPtr Device => _device;

    /// <summary>Gets the OpenCL command queue handle.</summary>
    public IntPtr CommandQueue => _commandQueue;

    /// <summary>Gets the device name.</summary>
    public string DeviceName => _deviceName;

    /// <summary>Gets the device vendor.</summary>
    public string DeviceVendor => _deviceVendor;

    /// <summary>Gets the number of compute units.</summary>
    public uint MaxComputeUnits => _maxComputeUnits;

    /// <summary>Gets the global memory size in bytes.</summary>
    public ulong GlobalMemSize => _globalMemSize;

    /// <summary>Gets the local memory size in bytes.</summary>
    public ulong LocalMemSize => _localMemSize;

    /// <summary>Creates a new OpenCL context with the first available GPU.</summary>
    public OpenClContext() : this(OpenClNative.ClDeviceType.Gpu)
    {
    }

    /// <summary>Creates a new OpenCL context with specified device type.</summary>
    public OpenClContext(OpenClNative.ClDeviceType deviceType)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("OpenCL is not available on this system.");

        // Get platforms
        uint numPlatforms;
        var error = OpenClNative.clGetPlatformIDs(0, null!, out numPlatforms);
        if (error != OpenClNative.ClError.Success || numPlatforms == 0)
            throw new InvalidOperationException($"No OpenCL platforms found: {error}");

        var platforms = new IntPtr[numPlatforms];
        error = OpenClNative.clGetPlatformIDs(numPlatforms, platforms, out numPlatforms);
        if (error != OpenClNative.ClError.Success)
            throw new InvalidOperationException($"Failed to get OpenCL platforms: {error}");

        // Find a GPU device
        bool foundDevice = false;
        foreach (var platform in platforms)
        {
            uint numDevices;
            error = OpenClNative.clGetDeviceIDs(platform, deviceType, 0, null!, out numDevices);
            if (error != OpenClNative.ClError.Success || numDevices == 0)
                continue;

            var devices = new IntPtr[numDevices];
            error = OpenClNative.clGetDeviceIDs(platform, deviceType, numDevices, devices, out numDevices);
            if (error != OpenClNative.ClError.Success)
                continue;

            _platform = platform;
            _device = devices[0];  // Use first device
            foundDevice = true;
            break;
        }

        if (!foundDevice)
            throw new InvalidOperationException($"No OpenCL {deviceType} device found.");

        // Query device info
        QueryDeviceInfo();

        // Create context
        var contextProperties = new IntPtr[]
        {
            new IntPtr(0x1084),  // CL_CONTEXT_PLATFORM
            _platform,
            IntPtr.Zero
        };

        _context = OpenClNative.clCreateContext(
            contextProperties,
            1,
            new[] { _device },
            IntPtr.Zero,
            IntPtr.Zero,
            out error);

        if (error != OpenClNative.ClError.Success)
            throw new InvalidOperationException($"Failed to create OpenCL context: {error}");

        // Create command queue (try 2.0 API first, fall back to 1.x)
        _commandQueue = OpenClNative.clCreateCommandQueueWithProperties(
            _context, _device, null!, out error);

        if (error != OpenClNative.ClError.Success)
        {
            // Fall back to OpenCL 1.x API
            _commandQueue = OpenClNative.clCreateCommandQueue(
                _context, _device,
                OpenClNative.ClCommandQueueProperties.None,
                out error);

            if (error != OpenClNative.ClError.Success)
                throw new InvalidOperationException($"Failed to create OpenCL command queue: {error}");
        }
    }

    private void QueryDeviceInfo()
    {
        UIntPtr returnSize;

        // Get device name
        var nameBuffer = new byte[256];
        var error = OpenClNative.clGetDeviceInfo(_device, OpenClNative.ClDeviceInfo.Name,
            (UIntPtr)nameBuffer.Length, nameBuffer, out returnSize);
        if (error == OpenClNative.ClError.Success)
            _deviceName = Encoding.UTF8.GetString(nameBuffer, 0, (int)returnSize - 1).TrimEnd('\0');

        // Get device vendor
        var vendorBuffer = new byte[256];
        error = OpenClNative.clGetDeviceInfo(_device, OpenClNative.ClDeviceInfo.Vendor,
            (UIntPtr)vendorBuffer.Length, vendorBuffer, out returnSize);
        if (error == OpenClNative.ClError.Success)
            _deviceVendor = Encoding.UTF8.GetString(vendorBuffer, 0, (int)returnSize - 1).TrimEnd('\0');

        // Get max compute units
        error = OpenClNative.clGetDeviceInfo(_device, OpenClNative.ClDeviceInfo.MaxComputeUnits,
            (UIntPtr)sizeof(uint), out _maxComputeUnits, out returnSize);

        // Get global memory size
        error = OpenClNative.clGetDeviceInfo(_device, OpenClNative.ClDeviceInfo.GlobalMemSize,
            (UIntPtr)sizeof(ulong), out _globalMemSize, out returnSize);

        // Get local memory size
        error = OpenClNative.clGetDeviceInfo(_device, OpenClNative.ClDeviceInfo.LocalMemSize,
            (UIntPtr)sizeof(ulong), out _localMemSize, out returnSize);
    }

    private static bool CheckAvailability()
    {
        try
        {
            uint numPlatforms;
            var error = OpenClNative.clGetPlatformIDs(0, null!, out numPlatforms);
            return error == OpenClNative.ClError.Success && numPlatforms > 0;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Blocks until all commands in the queue have completed.</summary>
    public void Finish()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClContext));
        OpenClNative.clFinish(_commandQueue);
    }

    /// <summary>Gets error message for OpenCL error code.</summary>
    public static string GetErrorString(OpenClNative.ClError error)
    {
        return error.ToString();
    }

    /// <summary>Throws if error indicates a failure.</summary>
    public static void CheckError(OpenClNative.ClError error, string operation)
    {
        if (error != OpenClNative.ClError.Success)
        {
            throw new InvalidOperationException($"OpenCL {operation} failed: {GetErrorString(error)}");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_commandQueue != IntPtr.Zero)
        {
            OpenClNative.clReleaseCommandQueue(_commandQueue);
            _commandQueue = IntPtr.Zero;
        }

        if (_context != IntPtr.Zero)
        {
            OpenClNative.clReleaseContext(_context);
            _context = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~OpenClContext()
    {
        Dispose();
    }
}

/// <summary>
/// OpenCL memory buffer wrapper with automatic resource management.
/// </summary>
/// <typeparam name="T">The unmanaged element type.</typeparam>
public sealed class OpenClBuffer<T> : IDisposable where T : unmanaged
{
    private IntPtr _buffer;
    private readonly OpenClContext _context;
    private readonly int _elementCount;
    private bool _disposed;

    /// <summary>Gets the buffer handle.</summary>
    public IntPtr Handle => _buffer;

    /// <summary>Gets the element count.</summary>
    public int Length => _elementCount;

    /// <summary>Gets the size in bytes.</summary>
    public ulong SizeInBytes => (ulong)(_elementCount * Marshal.SizeOf<T>());

    /// <summary>Creates a new OpenCL buffer.</summary>
    public OpenClBuffer(OpenClContext context, int elementCount, OpenClNative.ClMemFlags flags = OpenClNative.ClMemFlags.ReadWrite)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _elementCount = elementCount;

        var size = (UIntPtr)((ulong)elementCount * (ulong)Marshal.SizeOf<T>());
        OpenClNative.ClError error;
        _buffer = OpenClNative.clCreateBuffer(
            context.Handle,
            flags,
            size,
            IntPtr.Zero,
            out error);

        OpenClContext.CheckError(error, "CreateBuffer");
    }

    /// <summary>Creates a new OpenCL buffer from host data.</summary>
    public OpenClBuffer(OpenClContext context, T[] data, OpenClNative.ClMemFlags flags = OpenClNative.ClMemFlags.ReadWrite)
        : this(context, data.Length, flags)
    {
        CopyFromHost(data);
    }

    /// <summary>Copies data from host to device.</summary>
    public void CopyFromHost(T[] data)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClBuffer<T>));
        if (data.Length > _elementCount)
            throw new ArgumentException($"Data too large: {data.Length} > {_elementCount}");

        var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            IntPtr evt;
            var error = OpenClNative.clEnqueueWriteBuffer(
                _context.CommandQueue,
                _buffer,
                1,  // Blocking
                UIntPtr.Zero,
                (UIntPtr)((ulong)data.Length * (ulong)Marshal.SizeOf<T>()),
                handle.AddrOfPinnedObject(),
                0,
                null!,
                out evt);

            OpenClContext.CheckError(error, "EnqueueWriteBuffer");

            if (evt != IntPtr.Zero)
                OpenClNative.clReleaseEvent(evt);
        }
        finally
        {
            handle.Free();
        }
    }

    /// <summary>Copies data from device to host.</summary>
    public void CopyToHost(T[] data)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClBuffer<T>));
        if (data.Length < _elementCount)
            throw new ArgumentException($"Data array too small: {data.Length} < {_elementCount}");

        var handle = GCHandle.Alloc(data, GCHandleType.Pinned);
        try
        {
            IntPtr evt;
            var error = OpenClNative.clEnqueueReadBuffer(
                _context.CommandQueue,
                _buffer,
                1,  // Blocking
                UIntPtr.Zero,
                (UIntPtr)((ulong)_elementCount * (ulong)Marshal.SizeOf<T>()),
                handle.AddrOfPinnedObject(),
                0,
                null!,
                out evt);

            OpenClContext.CheckError(error, "EnqueueReadBuffer");

            if (evt != IntPtr.Zero)
                OpenClNative.clReleaseEvent(evt);
        }
        finally
        {
            handle.Free();
        }
    }

    /// <summary>Copies data from device to a new host array.</summary>
    public T[] ToArray()
    {
        var result = new T[_elementCount];
        CopyToHost(result);
        return result;
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_buffer != IntPtr.Zero)
        {
            OpenClNative.clReleaseMemObject(_buffer);
            _buffer = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~OpenClBuffer()
    {
        Dispose();
    }
}

/// <summary>
/// OpenCL program wrapper with automatic resource management.
/// </summary>
public sealed class OpenClProgram : IDisposable
{
    private IntPtr _program;
    private readonly OpenClContext _context;
    private bool _disposed;

    /// <summary>Gets the program handle.</summary>
    public IntPtr Handle => _program;

    /// <summary>Creates a program from source code.</summary>
    public OpenClProgram(OpenClContext context, string source)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));

        var sources = new[] { source };
        var lengths = new UIntPtr[] { (UIntPtr)source.Length };

        OpenClNative.ClError error;
        _program = OpenClNative.clCreateProgramWithSource(
            context.Handle,
            1,
            sources,
            lengths,
            out error);

        OpenClContext.CheckError(error, "CreateProgramWithSource");
    }

    /// <summary>Builds the program.</summary>
    public void Build(string options = "")
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClProgram));

        var error = OpenClNative.clBuildProgram(
            _program,
            1,
            new[] { _context.Device },
            options,
            IntPtr.Zero,
            IntPtr.Zero);

        if (error != OpenClNative.ClError.Success)
        {
            // Get build log
            UIntPtr logSize;
            OpenClNative.clGetProgramBuildInfo(
                _program, _context.Device,
                OpenClNative.ClProgramBuildInfo.Log,
                UIntPtr.Zero, null!, out logSize);

            var log = new byte[(int)logSize];
            OpenClNative.clGetProgramBuildInfo(
                _program, _context.Device,
                OpenClNative.ClProgramBuildInfo.Log,
                logSize, log, out logSize);

            var buildLog = Encoding.UTF8.GetString(log).TrimEnd('\0');
            throw new InvalidOperationException($"OpenCL program build failed: {error}\n{buildLog}");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_program != IntPtr.Zero)
        {
            OpenClNative.clReleaseProgram(_program);
            _program = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~OpenClProgram()
    {
        Dispose();
    }
}

/// <summary>
/// OpenCL kernel wrapper with automatic resource management.
/// </summary>
public sealed class OpenClKernel : IDisposable
{
    private IntPtr _kernel;
    private readonly OpenClContext _context;
    private bool _disposed;

    /// <summary>Gets the kernel handle.</summary>
    public IntPtr Handle => _kernel;

    /// <summary>Creates a kernel from a program.</summary>
    public OpenClKernel(OpenClContext context, OpenClProgram program, string kernelName)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));

        OpenClNative.ClError error;
        _kernel = OpenClNative.clCreateKernel(program.Handle, kernelName, out error);
        OpenClContext.CheckError(error, "CreateKernel");
    }

    /// <summary>Sets a buffer argument.</summary>
    public void SetArg(uint index, IntPtr buffer)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClKernel));

        var error = OpenClNative.clSetKernelArg(_kernel, index, (UIntPtr)IntPtr.Size, ref buffer);
        OpenClContext.CheckError(error, "SetKernelArg");
    }

    /// <summary>Sets an int argument.</summary>
    public void SetArg(uint index, int value)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClKernel));

        var error = OpenClNative.clSetKernelArgInt(_kernel, index, (UIntPtr)sizeof(int), ref value);
        OpenClContext.CheckError(error, "SetKernelArg");
    }

    /// <summary>Sets a float argument.</summary>
    public void SetArg(uint index, float value)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClKernel));

        var error = OpenClNative.clSetKernelArgFloat(_kernel, index, (UIntPtr)sizeof(float), ref value);
        OpenClContext.CheckError(error, "SetKernelArg");
    }

    /// <summary>Sets a local memory argument.</summary>
    public void SetLocalMemArg(uint index, ulong sizeInBytes)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClKernel));

        var error = OpenClNative.clSetKernelArg(_kernel, index, (UIntPtr)sizeInBytes, IntPtr.Zero);
        OpenClContext.CheckError(error, "SetKernelArg");
    }

    /// <summary>Gets the preferred work group size multiple.</summary>
    public ulong GetPreferredWorkGroupSizeMultiple()
    {
        UIntPtr value, returnSize;
        var error = OpenClNative.clGetKernelWorkGroupInfo(
            _kernel, _context.Device,
            OpenClNative.ClKernelWorkGroupInfo.PreferredWorkGroupSizeMultiple,
            (UIntPtr)sizeof(ulong), out value, out returnSize);

        return error == OpenClNative.ClError.Success ? (ulong)value : 64;
    }

    /// <summary>Gets the max work group size.</summary>
    public ulong GetMaxWorkGroupSize()
    {
        UIntPtr value, returnSize;
        var error = OpenClNative.clGetKernelWorkGroupInfo(
            _kernel, _context.Device,
            OpenClNative.ClKernelWorkGroupInfo.WorkGroupSize,
            (UIntPtr)sizeof(ulong), out value, out returnSize);

        return error == OpenClNative.ClError.Success ? (ulong)value : 256;
    }

    /// <summary>Enqueues the kernel for execution.</summary>
    public void Enqueue(ulong[] globalWorkSize, ulong[]? localWorkSize = null)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClKernel));

        var globalSize = Array.ConvertAll(globalWorkSize, g => (UIntPtr)g);
        var localSize = localWorkSize is not null
            ? Array.ConvertAll(localWorkSize, l => (UIntPtr)l)
            : null;

        IntPtr evt;
        var error = OpenClNative.clEnqueueNDRangeKernel(
            _context.CommandQueue,
            _kernel,
            (uint)globalWorkSize.Length,
            null!,
            globalSize,
            localSize!,
            0,
            null!,
            out evt);

        OpenClContext.CheckError(error, "EnqueueNDRangeKernel");

        if (evt != IntPtr.Zero)
            OpenClNative.clReleaseEvent(evt);
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_kernel != IntPtr.Zero)
        {
            OpenClNative.clReleaseKernel(_kernel);
            _kernel = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~OpenClKernel()
    {
        Dispose();
    }
}

/// <summary>
/// High-level OpenCL matrix multiplication helper.
/// Provides GEMM operations for AMD/Intel GPUs.
/// </summary>
public sealed class OpenClMatMul : IDisposable
{
    private readonly OpenClContext _context;
    private readonly bool _ownsContext;
    private OpenClProgram? _gemmProgram;
    private OpenClKernel? _gemmKernel;
    private bool _disposed;

    // Optimized GEMM kernel source
    private const string GemmKernelSource = @"
// Tiled SGEMM kernel for OpenCL
// C = alpha * A * B + beta * C
// A: M x K, B: K x N, C: M x N (row-major)

#define TILE_SIZE 32

__kernel void sgemm(
    const int M, const int N, const int K,
    const float alpha, const float beta,
    __global const float* A,
    __global const float* B,
    __global float* C)
{
    // Block indices
    const int bx = get_group_id(0);
    const int by = get_group_id(1);

    // Thread indices
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    // Global indices
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    // Shared memory for tiles
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    // Loop over tiles
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tiles into shared memory
        const int aCol = t * TILE_SIZE + tx;
        const int bRow = t * TILE_SIZE + ty;

        if (row < M && aCol < K)
            As[ty][tx] = A[row * K + aCol];
        else
            As[ty][tx] = 0.0f;

        if (bRow < K && col < N)
            Bs[ty][tx] = B[bRow * N + col];
        else
            Bs[ty][tx] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (row < M && col < N) {
        if (beta == 0.0f)
            C[row * N + col] = alpha * acc;
        else
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}
";

    /// <summary>Gets whether OpenCL GEMM is available.</summary>
    public static bool IsAvailable => OpenClContext.IsAvailable;

    /// <summary>Creates a new OpenCL GEMM helper with shared context.</summary>
    public OpenClMatMul(OpenClContext context)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _ownsContext = false;
        InitializeKernel();
    }

    /// <summary>Creates a new OpenCL GEMM helper with its own context.</summary>
    public OpenClMatMul()
    {
        _context = new OpenClContext();
        _ownsContext = true;
        InitializeKernel();
    }

    private void InitializeKernel()
    {
        _gemmProgram = new OpenClProgram(_context, GemmKernelSource);
        _gemmProgram.Build("-cl-fast-relaxed-math -cl-mad-enable");
        _gemmKernel = new OpenClKernel(_context, _gemmProgram, "sgemm");
    }

    /// <summary>
    /// Performs matrix multiplication C = alpha * A * B + beta * C.
    /// </summary>
    /// <param name="A">Input matrix A (row-major, M x K)</param>
    /// <param name="B">Input matrix B (row-major, K x N)</param>
    /// <param name="M">Number of rows in A</param>
    /// <param name="K">Number of columns in A / rows in B</param>
    /// <param name="N">Number of columns in B</param>
    /// <param name="alpha">Scalar multiplier for A*B</param>
    /// <param name="beta">Scalar multiplier for C</param>
    /// <returns>Result matrix C (M x N)</returns>
    public float[]? MatMulFloat(float[] A, int M, int K, float[] B, int N, float alpha = 1.0f, float beta = 0.0f)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OpenClMatMul));
        if (_gemmKernel is null) return null;

        try
        {
            int outputSize = M * N;

            using var bufferA = new OpenClBuffer<float>(_context, A, OpenClNative.ClMemFlags.ReadOnly);
            using var bufferB = new OpenClBuffer<float>(_context, B, OpenClNative.ClMemFlags.ReadOnly);
            using var bufferC = new OpenClBuffer<float>(_context, outputSize, OpenClNative.ClMemFlags.WriteOnly);

            // Set kernel arguments
            _gemmKernel.SetArg(0, M);
            _gemmKernel.SetArg(1, N);
            _gemmKernel.SetArg(2, K);
            _gemmKernel.SetArg(3, alpha);
            _gemmKernel.SetArg(4, beta);
            _gemmKernel.SetArg(5, bufferA.Handle);
            _gemmKernel.SetArg(6, bufferB.Handle);
            _gemmKernel.SetArg(7, bufferC.Handle);

            // Calculate work sizes
            const int tileSize = 32;
            ulong globalM = (ulong)((M + tileSize - 1) / tileSize * tileSize);
            ulong globalN = (ulong)((N + tileSize - 1) / tileSize * tileSize);

            var globalWorkSize = new ulong[] { globalN, globalM };
            var localWorkSize = new ulong[] { tileSize, tileSize };

            // Execute kernel
            _gemmKernel.Enqueue(globalWorkSize, localWorkSize);

            // Wait for completion
            _context.Finish();

            // Read result
            return bufferC.ToArray();
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// Performs matrix multiplication for DenseLayer (input * weightsT).
    /// </summary>
    /// <param name="input">Input tensor [batch, inputSize]</param>
    /// <param name="weights">Weights tensor [inputSize, outputSize] (already transposed)</param>
    /// <param name="batch">Batch size</param>
    /// <param name="inputSize">Input feature size</param>
    /// <param name="outputSize">Output feature size</param>
    /// <returns>Output tensor [batch, outputSize]</returns>
    public float[]? DenseForward(float[] input, float[] weights, int batch, int inputSize, int outputSize)
    {
        return MatMulFloat(input, batch, inputSize, weights, outputSize);
    }

    public void Dispose()
    {
        if (_disposed) return;

        _gemmKernel?.Dispose();
        _gemmProgram?.Dispose();

        if (_ownsContext)
        {
            _context.Dispose();
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~OpenClMatMul()
    {
        Dispose();
    }
}

/// <summary>
/// Lists available OpenCL platforms and devices.
/// </summary>
public static class OpenClInfo
{
    /// <summary>Gets information about all available OpenCL platforms and devices.</summary>
    public static string GetSystemInfo()
    {
        if (!OpenClContext.IsAvailable)
            return "OpenCL is not available on this system.";

        var sb = new StringBuilder();
        sb.AppendLine("=== OpenCL System Information ===");
        sb.AppendLine();

        try
        {
            // Get platforms
            uint numPlatforms;
            var error = OpenClNative.clGetPlatformIDs(0, null!, out numPlatforms);
            if (error != OpenClNative.ClError.Success)
            {
                sb.AppendLine($"Error getting platforms: {error}");
                return sb.ToString();
            }

            sb.AppendLine($"Found {numPlatforms} OpenCL platform(s):");
            sb.AppendLine();

            var platforms = new IntPtr[numPlatforms];
            OpenClNative.clGetPlatformIDs(numPlatforms, platforms, out numPlatforms);

            for (int p = 0; p < numPlatforms; p++)
            {
                var platform = platforms[p];
                sb.AppendLine($"Platform {p}:");

                // Platform name
                var nameBuffer = new byte[256];
                UIntPtr returnSize;
                OpenClNative.clGetPlatformInfo(platform, OpenClNative.ClPlatformInfo.Name,
                    (UIntPtr)nameBuffer.Length, nameBuffer, out returnSize);
                sb.AppendLine($"  Name: {Encoding.UTF8.GetString(nameBuffer, 0, (int)returnSize - 1).TrimEnd('\0')}");

                // Platform vendor
                var vendorBuffer = new byte[256];
                OpenClNative.clGetPlatformInfo(platform, OpenClNative.ClPlatformInfo.Vendor,
                    (UIntPtr)vendorBuffer.Length, vendorBuffer, out returnSize);
                sb.AppendLine($"  Vendor: {Encoding.UTF8.GetString(vendorBuffer, 0, (int)returnSize - 1).TrimEnd('\0')}");

                // Platform version
                var versionBuffer = new byte[256];
                OpenClNative.clGetPlatformInfo(platform, OpenClNative.ClPlatformInfo.Version,
                    (UIntPtr)versionBuffer.Length, versionBuffer, out returnSize);
                sb.AppendLine($"  Version: {Encoding.UTF8.GetString(versionBuffer, 0, (int)returnSize - 1).TrimEnd('\0')}");

                // Get devices
                uint numDevices;
                error = OpenClNative.clGetDeviceIDs(platform, OpenClNative.ClDeviceType.All, 0, null!, out numDevices);
                if (error != OpenClNative.ClError.Success || numDevices == 0)
                {
                    sb.AppendLine("  No devices found");
                    sb.AppendLine();
                    continue;
                }

                var devices = new IntPtr[numDevices];
                OpenClNative.clGetDeviceIDs(platform, OpenClNative.ClDeviceType.All, numDevices, devices, out numDevices);

                sb.AppendLine($"  Devices ({numDevices}):");

                for (int d = 0; d < numDevices; d++)
                {
                    var device = devices[d];
                    sb.AppendLine($"    Device {d}:");

                    // Device name
                    var deviceNameBuffer = new byte[256];
                    OpenClNative.clGetDeviceInfo(device, OpenClNative.ClDeviceInfo.Name,
                        (UIntPtr)deviceNameBuffer.Length, deviceNameBuffer, out returnSize);
                    sb.AppendLine($"      Name: {Encoding.UTF8.GetString(deviceNameBuffer, 0, (int)returnSize - 1).TrimEnd('\0')}");

                    // Device type
                    ulong deviceType;
                    OpenClNative.clGetDeviceInfo(device, OpenClNative.ClDeviceInfo.Type,
                        (UIntPtr)sizeof(ulong), out deviceType, out returnSize);
                    var typeStr = ((OpenClNative.ClDeviceType)deviceType).ToString();
                    sb.AppendLine($"      Type: {typeStr}");

                    // Max compute units
                    uint maxComputeUnits;
                    OpenClNative.clGetDeviceInfo(device, OpenClNative.ClDeviceInfo.MaxComputeUnits,
                        (UIntPtr)sizeof(uint), out maxComputeUnits, out returnSize);
                    sb.AppendLine($"      Compute Units: {maxComputeUnits}");

                    // Global memory
                    ulong globalMem;
                    OpenClNative.clGetDeviceInfo(device, OpenClNative.ClDeviceInfo.GlobalMemSize,
                        (UIntPtr)sizeof(ulong), out globalMem, out returnSize);
                    sb.AppendLine($"      Global Memory: {globalMem / (1024 * 1024 * 1024.0):F2} GB");

                    // Max clock frequency
                    uint maxClock;
                    OpenClNative.clGetDeviceInfo(device, OpenClNative.ClDeviceInfo.MaxClockFrequency,
                        (UIntPtr)sizeof(uint), out maxClock, out returnSize);
                    sb.AppendLine($"      Max Clock: {maxClock} MHz");
                }

                sb.AppendLine();
            }
        }
        catch (Exception ex)
        {
            sb.AppendLine($"Error: {ex.Message}");
        }

        return sb.ToString();
    }
}
#endif
