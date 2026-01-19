using System;
using System.Collections.Concurrent;
using System.IO;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Intel oneDNN (Deep Neural Network Library) provider for optimized convolution operations.
/// Uses P/Invoke to call native oneDNN library for highly optimized CPU convolution.
///
/// Performance optimizations:
/// 1. Primitive caching - caches primitives by dimension key to avoid recreation overhead
/// 2. Pinned memory - uses GCHandle.Pinned to avoid buffer copies
/// 3. Memory object pooling - reuses memory objects and updates data handles
/// </summary>
internal static class OneDnnProvider
{
    private static readonly object InitLock = new object();
    private static bool _initialized;
    private static bool _available;
    private static IntPtr _engine;
    private static IntPtr _stream;
    private static readonly bool TraceEnabled = ReadEnvBool("AIDOTNET_ONEDNN_TRACE");
    private static bool _resolverRegistered;
    private static bool _firstConv2DLogged;

    // Primitive cache for avoiding recreation overhead
    private static readonly ConcurrentDictionary<Conv2DKey, CachedConv2D> _conv2DCache = new();
    private const int MaxCacheSize = 32; // Limit cache size to avoid memory bloat

    // Windows API for DLL search path manipulation
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    private static extern bool SetDllDirectory(string lpPathName);

    // Static constructor to register the native library resolver
    static OneDnnProvider()
    {
        RegisterDllResolver();
    }

    private static void RegisterDllResolver()
    {
        if (_resolverRegistered) return;
        _resolverRegistered = true;

        NativeLibrary.SetDllImportResolver(typeof(OneDnnProvider).Assembly, (libraryName, assembly, searchPath) =>
        {
            if (libraryName != "dnnl")
            {
                return IntPtr.Zero;
            }

            // Try to load from the application directory first
            string? assemblyDir = Path.GetDirectoryName(assembly.Location);
            if (assemblyDir != null)
            {
                return TryLoadFromDirectory(assemblyDir);
            }

            // Try AppContext.BaseDirectory
            string baseDir = AppContext.BaseDirectory;
            var handle = TryLoadFromDirectory(baseDir);
            if (handle != IntPtr.Zero) return handle;

            // Try current directory
            handle = TryLoadFromDirectory(Environment.CurrentDirectory);
            if (handle != IntPtr.Zero) return handle;

            return IntPtr.Zero;
        });
    }

    private static IntPtr TryLoadFromDirectory(string directory)
    {
        string dnnlPath = Path.Combine(directory, "dnnl.dll");

        if (!File.Exists(dnnlPath))
        {
            if (TraceEnabled) Console.WriteLine($"[oneDNN] dnnl.dll not found in: {directory}");
            return IntPtr.Zero;
        }

        if (TraceEnabled) Console.WriteLine($"[oneDNN] Found dnnl.dll in: {directory}");

        SetDllDirectory(directory);
        if (TraceEnabled) Console.WriteLine($"[oneDNN] Set DLL directory to: {directory}");

        if (NativeLibrary.TryLoad(dnnlPath, out IntPtr handle))
        {
            if (TraceEnabled) Console.WriteLine($"[oneDNN] Successfully loaded dnnl.dll");
            return handle;
        }

        if (TraceEnabled) Console.WriteLine($"[oneDNN] Failed to load dnnl.dll from: {dnnlPath}");
        SetDllDirectory(null!);
        return IntPtr.Zero;
    }

    #region oneDNN Constants

    private const int DnnlSuccess = 0;
    private const int DnnlCpu = 1;
    private const int DnnlForwardInference = 96;
    private const int DnnlConvolutionAuto = 3;
    private const int DnnlConvolutionDirect = 1;
    private const int DnnlConvolutionWinograd = 2;
    private const int DnnlF32 = 3;

    // Format tags (from dnnl_types.h)
    private const int DnnlFormatTagAny = 1;  // Let oneDNN choose optimal format (format_tag_any = 1, not 0)
    private const int DnnlFormatTagNCHW = 11; // abcd format for 4D tensors
    private const int DnnlFormatTagOIHW = 11; // Same as NCHW for weights

    // Argument indices (from dnnl_types.h)
    private const int DnnlArgSrc = 1;
    private const int DnnlArgDst = 17;
    private const int DnnlArgWeights = 33;
    private const int DnnlArgScratchpad = 80;

    // Query types
    private const int DnnlQuerySrcMd = 129;
    private const int DnnlQueryWeightsMd = 131;
    private const int DnnlQueryDstMd = 133;
    private const int DnnlQueryScratchpadMd = 132;

    #endregion

    #region Structs

    /// <summary>
    /// Execution argument for dnnl_primitive_execute.
    /// </summary>
    [StructLayout(LayoutKind.Explicit, Size = 16)]
    private struct DnnlExecArg
    {
        [FieldOffset(0)]
        public int Arg;
        [FieldOffset(8)]
        public IntPtr Memory;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct DnnlVersion
    {
        public int Major;
        public int Minor;
        public int Patch;
        public IntPtr Hash;
        public uint CpuRuntime;
        public uint GpuRuntime;
    }

    /// <summary>
    /// Cache key for Conv2D primitives.
    /// </summary>
    private readonly struct Conv2DKey : IEquatable<Conv2DKey>
    {
        public readonly int Batch;
        public readonly int InChannels;
        public readonly int Height;
        public readonly int Width;
        public readonly int OutChannels;
        public readonly int KernelH;
        public readonly int KernelW;
        public readonly int StrideH;
        public readonly int StrideW;
        public readonly int PadH;
        public readonly int PadW;
        public readonly int DilationH;
        public readonly int DilationW;

        public Conv2DKey(int batch, int inC, int h, int w, int outC, int kH, int kW,
            int strideH, int strideW, int padH, int padW, int dilH, int dilW)
        {
            Batch = batch;
            InChannels = inC;
            Height = h;
            Width = w;
            OutChannels = outC;
            KernelH = kH;
            KernelW = kW;
            StrideH = strideH;
            StrideW = strideW;
            PadH = padH;
            PadW = padW;
            DilationH = dilH;
            DilationW = dilW;
        }

        public bool Equals(Conv2DKey other) =>
            Batch == other.Batch && InChannels == other.InChannels &&
            Height == other.Height && Width == other.Width &&
            OutChannels == other.OutChannels && KernelH == other.KernelH && KernelW == other.KernelW &&
            StrideH == other.StrideH && StrideW == other.StrideW &&
            PadH == other.PadH && PadW == other.PadW &&
            DilationH == other.DilationH && DilationW == other.DilationW;

        public override bool Equals(object? obj) => obj is Conv2DKey other && Equals(other);

        public override int GetHashCode()
        {
            var hash = new HashCode();
            hash.Add(Batch);
            hash.Add(InChannels);
            hash.Add(Height);
            hash.Add(Width);
            hash.Add(OutChannels);
            hash.Add(KernelH);
            hash.Add(KernelW);
            hash.Add(StrideH);
            hash.Add(StrideW);
            hash.Add(PadH);
            hash.Add(PadW);
            hash.Add(DilationH);
            hash.Add(DilationW);
            return hash.ToHashCode();
        }
    }

    /// <summary>
    /// Cached Conv2D primitive with memory objects for reuse.
    /// Includes reorder primitives for optimal data formats when oneDNN chooses
    /// a different internal format than the user's NCHW format.
    /// </summary>
    private sealed class CachedConv2D : IDisposable
    {
        // Main convolution primitive
        public IntPtr Primitive;
        public IntPtr PrimDesc;

        // User format memory objects (for setting data handles)
        public IntPtr UserSrcMem;
        public IntPtr UserWeightsMem;
        public IntPtr UserDstMem;

        // Optimal format memory objects (for convolution execution)
        // These may point to the same memory as UserXxxMem if no reorder is needed
        public IntPtr ConvSrcMem;
        public IntPtr ConvWeightsMem;
        public IntPtr ConvDstMem;
        public IntPtr ScratchpadMem;

        // Reorder primitives (IntPtr.Zero if no reorder needed)
        public IntPtr SrcReorderPrim;
        public IntPtr WeightsReorderPrim;
        public IntPtr DstReorderPrim;

        // Memory descriptors (user format)
        public IntPtr UserSrcDesc;
        public IntPtr UserWeightsDesc;
        public IntPtr UserDstDesc;

        // Allocated buffers for reordered data (IntPtr.Zero if no reorder)
        public IntPtr ReorderedSrcBuffer;
        public IntPtr ReorderedWeightsBuffer;
        public IntPtr ReorderedDstBuffer;

        // Flags
        public bool NeedsSrcReorder;
        public bool NeedsWeightsReorder;
        public bool NeedsDstReorder;
        public bool WeightsReordered; // True after first weights reorder

        public int OutHeight;
        public int OutWidth;
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            // Free reorder primitives
            if (DstReorderPrim != IntPtr.Zero) dnnl_primitive_destroy(DstReorderPrim);
            if (WeightsReorderPrim != IntPtr.Zero) dnnl_primitive_destroy(WeightsReorderPrim);
            if (SrcReorderPrim != IntPtr.Zero) dnnl_primitive_destroy(SrcReorderPrim);

            // Free memory objects (only free Conv* if they are different from User*)
            if (ScratchpadMem != IntPtr.Zero) dnnl_memory_destroy(ScratchpadMem);
            if (ConvDstMem != IntPtr.Zero && ConvDstMem != UserDstMem) dnnl_memory_destroy(ConvDstMem);
            if (ConvWeightsMem != IntPtr.Zero && ConvWeightsMem != UserWeightsMem) dnnl_memory_destroy(ConvWeightsMem);
            if (ConvSrcMem != IntPtr.Zero && ConvSrcMem != UserSrcMem) dnnl_memory_destroy(ConvSrcMem);
            if (UserDstMem != IntPtr.Zero) dnnl_memory_destroy(UserDstMem);
            if (UserWeightsMem != IntPtr.Zero) dnnl_memory_destroy(UserWeightsMem);
            if (UserSrcMem != IntPtr.Zero) dnnl_memory_destroy(UserSrcMem);

            // Free primitives
            if (Primitive != IntPtr.Zero) dnnl_primitive_destroy(Primitive);
            if (PrimDesc != IntPtr.Zero) dnnl_primitive_desc_destroy(PrimDesc);

            // Free allocated buffers
            if (ReorderedDstBuffer != IntPtr.Zero) Marshal.FreeHGlobal(ReorderedDstBuffer);
            if (ReorderedWeightsBuffer != IntPtr.Zero) Marshal.FreeHGlobal(ReorderedWeightsBuffer);
            if (ReorderedSrcBuffer != IntPtr.Zero) Marshal.FreeHGlobal(ReorderedSrcBuffer);

            // Free memory descriptors
            if (UserDstDesc != IntPtr.Zero) dnnl_memory_desc_destroy(UserDstDesc);
            if (UserWeightsDesc != IntPtr.Zero) dnnl_memory_desc_destroy(UserWeightsDesc);
            if (UserSrcDesc != IntPtr.Zero) dnnl_memory_desc_destroy(UserSrcDesc);
        }
    }

    #endregion

    #region P/Invoke Declarations

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_engine_create(out IntPtr engine, int kind, int index);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr dnnl_version();

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_engine_destroy(IntPtr engine);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_stream_create(out IntPtr stream, IntPtr engine, uint flags);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_stream_destroy(IntPtr stream);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_stream_wait(IntPtr stream);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe int dnnl_memory_desc_create_with_strides(
        out IntPtr memoryDesc,
        int ndims,
        long* dims,
        int dataType,
        long* strides);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_memory_desc_destroy(IntPtr memoryDesc);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe int dnnl_memory_desc_create_with_tag(
        out IntPtr memoryDesc,
        int ndims,
        long* dims,
        int dataType,
        int formatTag);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_reorder_primitive_desc_create(
        out IntPtr reorderPrimDesc,
        IntPtr srcMemDesc,
        IntPtr srcEngine,
        IntPtr dstMemDesc,
        IntPtr dstEngine,
        IntPtr attr);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_memory_desc_equal(IntPtr lhs, IntPtr rhs);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern long dnnl_memory_desc_get_size(IntPtr memoryDesc);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_memory_get_data_handle(IntPtr memory, out IntPtr handle);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_memory_create(
        out IntPtr memory,
        IntPtr memoryDesc,
        IntPtr engine,
        IntPtr handle);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_memory_destroy(IntPtr memory);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_memory_set_data_handle(IntPtr memory, IntPtr handle);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe int dnnl_convolution_forward_primitive_desc_create(
        out IntPtr primitiveDesc,
        IntPtr engine,
        int propKind,
        int algKind,
        IntPtr srcDesc,
        IntPtr weightsDesc,
        IntPtr biasDesc,
        IntPtr dstDesc,
        long* strides,
        long* dilates,
        long* paddingL,
        long* paddingR,
        IntPtr attr);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_primitive_desc_destroy(IntPtr primitiveDesc);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr dnnl_primitive_desc_query_md(IntPtr primitiveDesc, int what, int index);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_primitive_create(out IntPtr primitive, IntPtr primitiveDesc);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_primitive_destroy(IntPtr primitive);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe int dnnl_primitive_execute(
        IntPtr primitive,
        IntPtr stream,
        int nargs,
        DnnlExecArg* args);

    #endregion

    /// <summary>
    /// Returns true if oneDNN is available.
    /// </summary>
    internal static bool IsAvailable
    {
        get
        {
            EnsureInitialized();
            return _available;
        }
    }

    /// <summary>
    /// Performs Conv2D using oneDNN with cached primitives and pinned memory for maximum performance.
    /// </summary>
    internal static unsafe bool TryConv2D(
        float* input, int batch, int inChannels, int height, int width,
        float* kernel, int outChannels, int kernelH, int kernelW,
        float* output, int outHeight, int outWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        bool logThis = !_firstConv2DLogged || TraceEnabled;

        if (!EnsureInitialized())
        {
            if (logThis) Console.WriteLine("[oneDNN] TryConv2D: Not initialized");
            return false;
        }

        if (logThis)
        {
            Console.WriteLine($"[oneDNN] TryConv2D: batch={batch}, inC={inChannels}, H={height}, W={width}, outC={outChannels}, kH={kernelH}, kW={kernelW}");
        }

        // Create cache key
        var key = new Conv2DKey(batch, inChannels, height, width, outChannels, kernelH, kernelW,
            strideH, strideW, padH, padW, dilationH, dilationW);

        // Try to get cached primitive or create new one
        if (!_conv2DCache.TryGetValue(key, out var cached))
        {
            cached = CreateConv2DPrimitive(key, outHeight, outWidth, logThis);
            if (cached == null)
            {
                return false;
            }

            // Add to cache (limit cache size)
            if (_conv2DCache.Count >= MaxCacheSize)
            {
                // Remove a random entry to make room
                foreach (var k in _conv2DCache.Keys)
                {
                    if (_conv2DCache.TryRemove(k, out var removed))
                    {
                        removed.Dispose();
                        break;
                    }
                }
            }
            _conv2DCache[key] = cached;
            if (logThis) Console.WriteLine("[oneDNN] Created and cached new primitive");
        }
        else
        {
            if (logThis) Console.WriteLine("[oneDNN] Using cached primitive");
        }

        try
        {
            int status;

            // Step 1: Set user memory data handles to point to user data
            status = dnnl_memory_set_data_handle(cached.UserSrcMem, (IntPtr)input);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to set src handle: {status}"); return false; }

            status = dnnl_memory_set_data_handle(cached.UserWeightsMem, (IntPtr)kernel);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to set weights handle: {status}"); return false; }

            status = dnnl_memory_set_data_handle(cached.UserDstMem, (IntPtr)output);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to set dst handle: {status}"); return false; }

            // Step 2: Execute source reorder if needed (NCHW -> blocked format)
            if (cached.NeedsSrcReorder)
            {
                DnnlExecArg* srcReorderArgs = stackalloc DnnlExecArg[2];
                srcReorderArgs[0].Arg = DnnlArgSrc;
                srcReorderArgs[0].Memory = cached.UserSrcMem;
                srcReorderArgs[1].Arg = DnnlArgDst;
                srcReorderArgs[1].Memory = cached.ConvSrcMem;

                status = dnnl_primitive_execute(cached.SrcReorderPrim, _stream, 2, srcReorderArgs);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Src reorder failed: {status}"); return false; }
            }

            // Step 3: Execute weights reorder if needed (only first time - weights are cached)
            if (cached.NeedsWeightsReorder && !cached.WeightsReordered)
            {
                DnnlExecArg* weightsReorderArgs = stackalloc DnnlExecArg[2];
                weightsReorderArgs[0].Arg = DnnlArgSrc;
                weightsReorderArgs[0].Memory = cached.UserWeightsMem;
                weightsReorderArgs[1].Arg = DnnlArgDst;
                weightsReorderArgs[1].Memory = cached.ConvWeightsMem;

                status = dnnl_primitive_execute(cached.WeightsReorderPrim, _stream, 2, weightsReorderArgs);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Weights reorder failed: {status}"); return false; }

                cached.WeightsReordered = true;
                if (logThis) Console.WriteLine("[oneDNN] Weights reordered to blocked format (cached for future use)");
            }

            // Step 4: Execute convolution with optimal format memory objects
            DnnlExecArg* convArgs = stackalloc DnnlExecArg[4];
            convArgs[0].Arg = DnnlArgSrc;
            convArgs[0].Memory = cached.ConvSrcMem;
            convArgs[1].Arg = DnnlArgWeights;
            convArgs[1].Memory = cached.ConvWeightsMem;
            convArgs[2].Arg = DnnlArgDst;
            convArgs[2].Memory = cached.ConvDstMem;
            convArgs[3].Arg = DnnlArgScratchpad;
            convArgs[3].Memory = cached.ScratchpadMem;

            status = dnnl_primitive_execute(cached.Primitive, _stream, 4, convArgs);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Conv execute failed: {status}"); return false; }

            // Step 5: Execute destination reorder if needed (blocked format -> NCHW)
            if (cached.NeedsDstReorder)
            {
                DnnlExecArg* dstReorderArgs = stackalloc DnnlExecArg[2];
                dstReorderArgs[0].Arg = DnnlArgSrc;
                dstReorderArgs[0].Memory = cached.ConvDstMem;
                dstReorderArgs[1].Arg = DnnlArgDst;
                dstReorderArgs[1].Memory = cached.UserDstMem;

                status = dnnl_primitive_execute(cached.DstReorderPrim, _stream, 2, dstReorderArgs);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Dst reorder failed: {status}"); return false; }
            }

            dnnl_stream_wait(_stream);

            if (logThis)
            {
                Console.WriteLine("[oneDNN] Conv2D completed successfully");
                _firstConv2DLogged = true;
            }
            return true;
        }
        catch (Exception ex)
        {
            if (logThis)
            {
                Console.WriteLine($"[oneDNN] Conv2D exception: {ex.Message}");
                _firstConv2DLogged = true;
            }
            return false;
        }
    }

    /// <summary>
    /// Creates a new Conv2D primitive with all associated memory objects.
    /// Uses "any" format descriptors to let oneDNN choose optimal data layouts,
    /// with reorder primitives for format conversion.
    /// </summary>
    private static unsafe CachedConv2D? CreateConv2DPrimitive(Conv2DKey key, int outHeight, int outWidth, bool logThis)
    {
        var cached = new CachedConv2D { OutHeight = outHeight, OutWidth = outWidth };

        IntPtr anySrcDesc = IntPtr.Zero;
        IntPtr anyWeightsDesc = IntPtr.Zero;
        IntPtr anyDstDesc = IntPtr.Zero;

        try
        {
            // Create dimension arrays
            long* srcDims = stackalloc long[4];
            srcDims[0] = key.Batch;
            srcDims[1] = key.InChannels;
            srcDims[2] = key.Height;
            srcDims[3] = key.Width;

            long* weightsDims = stackalloc long[4];
            weightsDims[0] = key.OutChannels;
            weightsDims[1] = key.InChannels;
            weightsDims[2] = key.KernelH;
            weightsDims[3] = key.KernelW;

            long* dstDims = stackalloc long[4];
            dstDims[0] = key.Batch;
            dstDims[1] = key.OutChannels;
            dstDims[2] = outHeight;
            dstDims[3] = outWidth;

            long* stridesArr = stackalloc long[2];
            stridesArr[0] = key.StrideH;
            stridesArr[1] = key.StrideW;

            long* dilatesArr = stackalloc long[2];
            dilatesArr[0] = key.DilationH - 1;
            dilatesArr[1] = key.DilationW - 1;

            long* paddingLArr = stackalloc long[2];
            paddingLArr[0] = key.PadH;
            paddingLArr[1] = key.PadW;

            long* paddingRArr = stackalloc long[2];
            paddingRArr[0] = key.PadH;
            paddingRArr[1] = key.PadW;

            // Create NCHW strides for user format
            long* srcStrides = stackalloc long[4];
            srcStrides[0] = (long)key.InChannels * key.Height * key.Width;
            srcStrides[1] = (long)key.Height * key.Width;
            srcStrides[2] = key.Width;
            srcStrides[3] = 1;

            long* weightsStrides = stackalloc long[4];
            weightsStrides[0] = (long)key.InChannels * key.KernelH * key.KernelW;
            weightsStrides[1] = (long)key.KernelH * key.KernelW;
            weightsStrides[2] = key.KernelW;
            weightsStrides[3] = 1;

            long* dstStrides = stackalloc long[4];
            dstStrides[0] = (long)key.OutChannels * outHeight * outWidth;
            dstStrides[1] = (long)outHeight * outWidth;
            dstStrides[2] = outWidth;
            dstStrides[3] = 1;

            // Create user format memory descriptors (plain NCHW using strides)
            int status = dnnl_memory_desc_create_with_strides(out cached.UserSrcDesc, 4, srcDims, DnnlF32, srcStrides);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create user src desc: {status}"); cached.Dispose(); return null; }

            status = dnnl_memory_desc_create_with_strides(out cached.UserWeightsDesc, 4, weightsDims, DnnlF32, weightsStrides);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create user weights desc: {status}"); cached.Dispose(); return null; }

            status = dnnl_memory_desc_create_with_strides(out cached.UserDstDesc, 4, dstDims, DnnlF32, dstStrides);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create user dst desc: {status}"); cached.Dispose(); return null; }

            // Determine whether to use format_tag_any (blocked formats) or user format (NCHW)
            // Blocked formats enable brgconv algorithm which is ~6x faster than GEMM, even with reorder overhead
            // Only skip for very small tensors where primitive overhead dominates
            long outputElements = (long)key.Batch * key.OutChannels * outHeight * outWidth;
            bool useBlockedFormat = outputElements >= 8 * 1024; // 8K elements (~32KB output) - use blocked for most cases

            if (logThis)
            {
                Console.WriteLine($"[oneDNN] Output elements: {outputElements}, using blocked format: {useBlockedFormat}");
            }

            if (useBlockedFormat)
            {
                // Create "any" format descriptors to let oneDNN choose optimal layout
                // This enables blocked formats (nChw16c) which allow Winograd for 3x3 kernels
                status = dnnl_memory_desc_create_with_tag(out anySrcDesc, 4, srcDims, DnnlF32, DnnlFormatTagAny);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create any src desc: {status}"); cached.Dispose(); return null; }

                status = dnnl_memory_desc_create_with_tag(out anyWeightsDesc, 4, weightsDims, DnnlF32, DnnlFormatTagAny);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create any weights desc: {status}"); cached.Dispose(); return null; }

                status = dnnl_memory_desc_create_with_tag(out anyDstDesc, 4, dstDims, DnnlF32, DnnlFormatTagAny);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create any dst desc: {status}"); cached.Dispose(); return null; }

                // Create convolution primitive descriptor with "any" format descriptors
                status = dnnl_convolution_forward_primitive_desc_create(
                    out cached.PrimDesc, _engine, DnnlForwardInference, DnnlConvolutionAuto,
                    anySrcDesc, anyWeightsDesc, IntPtr.Zero, anyDstDesc,
                    stridesArr, dilatesArr, paddingLArr, paddingRArr, IntPtr.Zero);
            }
            else
            {
                // Use user format (NCHW) directly to avoid reorder overhead for small tensors
                status = dnnl_convolution_forward_primitive_desc_create(
                    out cached.PrimDesc, _engine, DnnlForwardInference, DnnlConvolutionAuto,
                    cached.UserSrcDesc, cached.UserWeightsDesc, IntPtr.Zero, cached.UserDstDesc,
                    stridesArr, dilatesArr, paddingLArr, paddingRArr, IntPtr.Zero);
            }
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create conv prim desc: {status}"); cached.Dispose(); return null; }

            // Query the memory descriptors that oneDNN chose
            IntPtr convSrcDesc = dnnl_primitive_desc_query_md(cached.PrimDesc, DnnlQuerySrcMd, 0);
            IntPtr convWeightsDesc = dnnl_primitive_desc_query_md(cached.PrimDesc, DnnlQueryWeightsMd, 0);
            IntPtr convDstDesc = dnnl_primitive_desc_query_md(cached.PrimDesc, DnnlQueryDstMd, 0);
            IntPtr scratchpadDesc = dnnl_primitive_desc_query_md(cached.PrimDesc, DnnlQueryScratchpadMd, 0);

            if (convSrcDesc == IntPtr.Zero || convWeightsDesc == IntPtr.Zero || convDstDesc == IntPtr.Zero)
            {
                if (logThis) Console.WriteLine("[oneDNN] Queried memory descriptors are null");
                cached.Dispose();
                return null;
            }

            // Check if reorders are needed by comparing user format with oneDNN's chosen format
            cached.NeedsSrcReorder = dnnl_memory_desc_equal(cached.UserSrcDesc, convSrcDesc) == 0;
            cached.NeedsWeightsReorder = dnnl_memory_desc_equal(cached.UserWeightsDesc, convWeightsDesc) == 0;
            cached.NeedsDstReorder = dnnl_memory_desc_equal(cached.UserDstDesc, convDstDesc) == 0;

            if (logThis)
            {
                Console.WriteLine($"[oneDNN] Reorder needed: src={cached.NeedsSrcReorder}, weights={cached.NeedsWeightsReorder}, dst={cached.NeedsDstReorder}");
            }

            // Create primitive
            status = dnnl_primitive_create(out cached.Primitive, cached.PrimDesc);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create primitive: {status}"); cached.Dispose(); return null; }

            // Create user format memory objects
            status = dnnl_memory_create(out cached.UserSrcMem, cached.UserSrcDesc, _engine, IntPtr.Zero);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create user src mem: {status}"); cached.Dispose(); return null; }

            status = dnnl_memory_create(out cached.UserWeightsMem, cached.UserWeightsDesc, _engine, IntPtr.Zero);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create user weights mem: {status}"); cached.Dispose(); return null; }

            status = dnnl_memory_create(out cached.UserDstMem, cached.UserDstDesc, _engine, IntPtr.Zero);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create user dst mem: {status}"); cached.Dispose(); return null; }

            // Create reorder primitives and allocate buffers if needed
            if (cached.NeedsSrcReorder)
            {
                long srcSize = dnnl_memory_desc_get_size(convSrcDesc);
                cached.ReorderedSrcBuffer = Marshal.AllocHGlobal((IntPtr)srcSize);

                status = dnnl_memory_create(out cached.ConvSrcMem, convSrcDesc, _engine, cached.ReorderedSrcBuffer);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create conv src mem: {status}"); cached.Dispose(); return null; }

                // Create reorder primitive: user src -> conv src
                status = dnnl_reorder_primitive_desc_create(out IntPtr srcReorderPrimDesc,
                    cached.UserSrcDesc, _engine, convSrcDesc, _engine, IntPtr.Zero);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create src reorder prim desc: {status}"); cached.Dispose(); return null; }

                status = dnnl_primitive_create(out cached.SrcReorderPrim, srcReorderPrimDesc);
                dnnl_primitive_desc_destroy(srcReorderPrimDesc);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create src reorder prim: {status}"); cached.Dispose(); return null; }
            }
            else
            {
                // No reorder needed - conv uses user memory directly
                cached.ConvSrcMem = cached.UserSrcMem;
            }

            if (cached.NeedsWeightsReorder)
            {
                long weightsSize = dnnl_memory_desc_get_size(convWeightsDesc);
                cached.ReorderedWeightsBuffer = Marshal.AllocHGlobal((IntPtr)weightsSize);

                status = dnnl_memory_create(out cached.ConvWeightsMem, convWeightsDesc, _engine, cached.ReorderedWeightsBuffer);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create conv weights mem: {status}"); cached.Dispose(); return null; }

                // Create reorder primitive: user weights -> conv weights
                status = dnnl_reorder_primitive_desc_create(out IntPtr weightsReorderPrimDesc,
                    cached.UserWeightsDesc, _engine, convWeightsDesc, _engine, IntPtr.Zero);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create weights reorder prim desc: {status}"); cached.Dispose(); return null; }

                status = dnnl_primitive_create(out cached.WeightsReorderPrim, weightsReorderPrimDesc);
                dnnl_primitive_desc_destroy(weightsReorderPrimDesc);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create weights reorder prim: {status}"); cached.Dispose(); return null; }
            }
            else
            {
                cached.ConvWeightsMem = cached.UserWeightsMem;
            }

            if (cached.NeedsDstReorder)
            {
                long dstSize = dnnl_memory_desc_get_size(convDstDesc);
                cached.ReorderedDstBuffer = Marshal.AllocHGlobal((IntPtr)dstSize);

                status = dnnl_memory_create(out cached.ConvDstMem, convDstDesc, _engine, cached.ReorderedDstBuffer);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create conv dst mem: {status}"); cached.Dispose(); return null; }

                // Create reorder primitive: conv dst -> user dst
                status = dnnl_reorder_primitive_desc_create(out IntPtr dstReorderPrimDesc,
                    convDstDesc, _engine, cached.UserDstDesc, _engine, IntPtr.Zero);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create dst reorder prim desc: {status}"); cached.Dispose(); return null; }

                status = dnnl_primitive_create(out cached.DstReorderPrim, dstReorderPrimDesc);
                dnnl_primitive_desc_destroy(dstReorderPrimDesc);
                if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create dst reorder prim: {status}"); cached.Dispose(); return null; }
            }
            else
            {
                cached.ConvDstMem = cached.UserDstMem;
            }

            // Create scratchpad memory
            IntPtr scratchpadDescToUse = scratchpadDesc != IntPtr.Zero ? scratchpadDesc : cached.UserDstDesc;
            status = dnnl_memory_create(out cached.ScratchpadMem, scratchpadDescToUse, _engine, IntPtr.Zero);
            if (status != DnnlSuccess) { if (logThis) Console.WriteLine($"[oneDNN] Failed to create scratchpad mem: {status}"); cached.Dispose(); return null; }

            return cached;
        }
        catch (Exception ex)
        {
            if (logThis) Console.WriteLine($"[oneDNN] CreateConv2DPrimitive exception: {ex.Message}");
            cached.Dispose();
            return null;
        }
        finally
        {
            // Clean up "any" format descriptors (they're no longer needed after primitive creation)
            if (anySrcDesc != IntPtr.Zero) dnnl_memory_desc_destroy(anySrcDesc);
            if (anyWeightsDesc != IntPtr.Zero) dnnl_memory_desc_destroy(anyWeightsDesc);
            if (anyDstDesc != IntPtr.Zero) dnnl_memory_desc_destroy(anyDstDesc);
        }
    }

    private static bool EnsureInitialized()
    {
        if (_initialized)
        {
            return _available;
        }

        lock (InitLock)
        {
            if (_initialized)
            {
                return _available;
            }

            _available = TryInitialize();
            _initialized = true;
            return _available;
        }
    }

    private static bool TryInitialize()
    {
        try
        {
            try
            {
                IntPtr versionPtr = dnnl_version();
                if (versionPtr != IntPtr.Zero)
                {
                    var version = Marshal.PtrToStructure<DnnlVersion>(versionPtr);
                    Console.WriteLine($"[oneDNN] Version: {version.Major}.{version.Minor}.{version.Patch}");
                }
            }
            catch
            {
                // Version check failed, continue anyway
            }

            Trace("[oneDNN] Initializing...");

            int status = dnnl_engine_create(out _engine, DnnlCpu, 0);
            if (status != DnnlSuccess)
            {
                Trace($"[oneDNN] Failed to create engine: status={status}");
                return false;
            }

            status = dnnl_stream_create(out _stream, _engine, 0);
            if (status != DnnlSuccess)
            {
                Trace($"[oneDNN] Failed to create stream: status={status}");
                dnnl_engine_destroy(_engine);
                _engine = IntPtr.Zero;
                return false;
            }

            Trace("[oneDNN] Initialized successfully");
            return true;
        }
        catch (DllNotFoundException ex)
        {
            Trace($"[oneDNN] DLL not found: {ex.Message}");
            return false;
        }
        catch (Exception ex)
        {
            Trace($"[oneDNN] Initialization failed: {ex.Message}");
            return false;
        }
    }

    private static void Trace(string message)
    {
        if (TraceEnabled)
        {
            Console.WriteLine(message);
        }
    }

    private static bool ReadEnvBool(string name)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return false;
        }

        return string.Equals(raw, "1", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase);
    }
}
