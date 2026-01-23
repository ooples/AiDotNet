#if !NET471
using System;
using System.IO;
using System.Runtime.InteropServices;

namespace AiDotNetBenchmarkTests;

public static class OneDnnDiagnostic
{
    public static void RunDiagnostic()
    {
        Console.WriteLine("=== oneDNN Loading Diagnostic ===\n");

        var baseDir = AppContext.BaseDirectory;
        Console.WriteLine($"Base directory: {baseDir}");
        Console.WriteLine($"Current directory: {Environment.CurrentDirectory}");

        // Check for DLLs
        var dnnlPath = Path.Combine(baseDir, "dnnl.dll");

        Console.WriteLine($"\ndnnl.dll exists: {File.Exists(dnnlPath)} at {dnnlPath}");

        // Set DLL search directory FIRST
        Console.WriteLine($"\nSetting DLL directory to: {baseDir}");
        SetDllDirectory(baseDir);

        // Try loading dnnl.dll (vcomp variant uses Visual C++ OpenMP)
        if (File.Exists(dnnlPath))
        {
            Console.WriteLine("\nAttempting to load dnnl.dll...");
            if (NativeLibrary.TryLoad(dnnlPath, out IntPtr dnnlHandle))
            {
                Console.WriteLine("  SUCCESS: dnnl.dll loaded");

                // Try to get a function pointer
                if (NativeLibrary.TryGetExport(dnnlHandle, "dnnl_engine_create", out IntPtr funcPtr))
                {
                    Console.WriteLine("  SUCCESS: Found dnnl_engine_create export");
                }
                else
                {
                    Console.WriteLine("  FAILED: Could not find dnnl_engine_create");
                }
            }
            else
            {
                Console.WriteLine("  FAILED: Could not load dnnl.dll");

                // Try with LoadLibraryEx flags
                Console.WriteLine("\nTrying with LOAD_WITH_ALTERED_SEARCH_PATH...");
                IntPtr handle = LoadLibraryEx(dnnlPath, IntPtr.Zero, LOAD_WITH_ALTERED_SEARCH_PATH);
                if (handle != IntPtr.Zero)
                {
                    Console.WriteLine("  SUCCESS with LOAD_WITH_ALTERED_SEARCH_PATH");
                    FreeLibrary(handle);
                }
                else
                {
                    int error = Marshal.GetLastWin32Error();
                    Console.WriteLine($"  FAILED with error code: {error}");
                    Console.WriteLine($"  Error message: {GetErrorMessage(error)}");
                }

                // Try using "dnnl" just as a name, let Windows search
                Console.WriteLine("\nTrying NativeLibrary.TryLoad with just 'dnnl'...");
                if (NativeLibrary.TryLoad("dnnl", typeof(OneDnnDiagnostic).Assembly, null, out handle))
                {
                    Console.WriteLine("  SUCCESS: Loaded dnnl via assembly search");
                    NativeLibrary.Free(handle);
                }
                else
                {
                    Console.WriteLine("  FAILED: Could not load dnnl via assembly search");
                }
            }
        }

        // Check for VC++ Runtime
        Console.WriteLine("\n=== Checking common dependencies ===");
        var vcRuntimePaths = new[]
        {
            Path.Combine(Environment.SystemDirectory, "vcruntime140.dll"),
            Path.Combine(Environment.SystemDirectory, "msvcp140.dll"),
            Path.Combine(Environment.SystemDirectory, "vcruntime140_1.dll")
        };

        foreach (var path in vcRuntimePaths)
        {
            Console.WriteLine($"{Path.GetFileName(path)}: {(File.Exists(path) ? "EXISTS" : "MISSING")}");
        }

        // Check AiDotNet oneDNN availability
        Console.WriteLine("\n=== Checking AiDotNet oneDNN availability ===");
        // Note: OneDnnProvider is internal to AiDotNet.Tensors - check via dnnl.dll loading above
        Console.WriteLine("OneDnnProvider availability check: See dnnl.dll loading results above");

        // Check for API version by checking function exports
        Console.WriteLine("\n=== Checking oneDNN API exports ===");
        if (NativeLibrary.TryLoad(dnnlPath, out IntPtr dnnlCheckHandle))
        {
            // v2+ API
            bool hasV2Api = NativeLibrary.TryGetExport(dnnlCheckHandle, "dnnl_memory_desc_create_with_tag", out _);
            // v1 API
            bool hasV1Api = NativeLibrary.TryGetExport(dnnlCheckHandle, "dnnl_memory_desc_init_by_tag", out _);
            // Version function
            bool hasVersionFunc = NativeLibrary.TryGetExport(dnnlCheckHandle, "dnnl_version", out _);

            Console.WriteLine($"dnnl_memory_desc_create_with_tag (v2+ API): {hasV2Api}");
            Console.WriteLine($"dnnl_memory_desc_init_by_tag (v1 API): {hasV1Api}");
            Console.WriteLine($"dnnl_version: {hasVersionFunc}");
            NativeLibrary.Free(dnnlCheckHandle);
        }

        // Direct API test with various format tags
        Console.WriteLine("\n=== Direct API test for memory descriptors ===");
        TestDirectOneDnnApi();

        // Set tracing to see oneDNN diagnostics
        Console.WriteLine("\n=== Testing Conv2D with oneDNN (AIDOTNET_ONEDNN_TRACE=1) ===");
        Environment.SetEnvironmentVariable("AIDOTNET_ONEDNN_TRACE", "1");

        try
        {
            var engine = new AiDotNet.Tensors.Engines.CpuEngine();

            // Create test data: batch=1, inChannels=16, height=64, width=64
            var inputData = new float[1 * 16 * 64 * 64];
            for (int i = 0; i < inputData.Length; i++) inputData[i] = 0.1f;
            var input = new Tensor<float>(inputData, [1, 16, 64, 64]);

            // Kernel: outChannels=32, inChannels=16, kernelH=3, kernelW=3
            var kernelData = new float[32 * 16 * 3 * 3];
            for (int i = 0; i < kernelData.Length; i++) kernelData[i] = 0.1f;
            var kernel = new Tensor<float>(kernelData, [32, 16, 3, 3]);

            // Time the Conv2D operation
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = engine.Conv2D(input, kernel, stride: 1, padding: 1, dilation: 1);
            sw.Stop();

            Console.WriteLine($"Conv2D result shape: [{string.Join(", ", result.Shape)}]");
            Console.WriteLine($"Conv2D time: {sw.Elapsed.TotalMilliseconds * 1000:F1} us");

            // Do a few warmup iterations and measure average
            Console.WriteLine("\nRunning 10 iterations for more accurate timing...");
            sw.Restart();
            for (int i = 0; i < 10; i++)
            {
                result = engine.Conv2D(input, kernel, stride: 1, padding: 1, dilation: 1);
            }
            sw.Stop();
            Console.WriteLine($"Average Conv2D time: {sw.Elapsed.TotalMilliseconds * 100:F1} us");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Conv2D test failed: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }

        Console.WriteLine("\n=== Diagnostic complete ===");
    }

    private const uint LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008;

    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    private static extern bool SetDllDirectory(string lpPathName);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    private static extern IntPtr LoadLibraryEx(string lpFileName, IntPtr hFile, uint dwFlags);

    [DllImport("kernel32.dll")]
    private static extern bool FreeLibrary(IntPtr hModule);

    private static string GetErrorMessage(int errorCode)
    {
        return errorCode switch
        {
            2 => "ERROR_FILE_NOT_FOUND - The system cannot find the file specified",
            126 => "ERROR_MOD_NOT_FOUND - The specified module could not be found (missing dependency)",
            127 => "ERROR_PROC_NOT_FOUND - The specified procedure could not be found",
            193 => "ERROR_BAD_EXE_FORMAT - Not a valid Win32 application (32/64-bit mismatch)",
            _ => $"Unknown error code: {errorCode}"
        };
    }

    // Direct test of oneDNN API
    private static unsafe void TestDirectOneDnnApi()
    {
        // Create engine first
        int status = dnnl_engine_create(out IntPtr engine, 1, 0); // 1 = CPU
        if (status != 0)
        {
            Console.WriteLine($"Failed to create engine: status={status}");
            return;
        }
        Console.WriteLine($"Engine created: {engine}");

        // Test memory descriptor creation with various format tags
        long[] dims = { 1, 16, 64, 64, 0, 0, 0, 0, 0, 0, 0, 0 }; // 4D with 12 elements

        fixed (long* dimsPtr = dims)
        {
            // Test format_tag_any (1)
            status = dnnl_memory_desc_create_with_tag(out IntPtr descAny, 4, dimsPtr, 3, 1);
            Console.WriteLine($"format_tag_any (1): status={status}, desc={descAny}");
            if (descAny != IntPtr.Zero) dnnl_memory_desc_destroy(descAny);

            // Test format_tag a (2) - 1D
            status = dnnl_memory_desc_create_with_tag(out IntPtr descA, 1, dimsPtr, 3, 2);
            Console.WriteLine($"format_tag a (2) 1D: status={status}, desc={descA}");
            if (descA != IntPtr.Zero) dnnl_memory_desc_destroy(descA);

            // Test format_tag ab (3) - 2D
            status = dnnl_memory_desc_create_with_tag(out IntPtr descAb, 2, dimsPtr, 3, 3);
            Console.WriteLine($"format_tag ab (3) 2D: status={status}, desc={descAb}");
            if (descAb != IntPtr.Zero) dnnl_memory_desc_destroy(descAb);

            // Test format_tag abcd (11) - 4D
            status = dnnl_memory_desc_create_with_tag(out IntPtr descAbcd, 4, dimsPtr, 3, 11);
            Console.WriteLine($"format_tag abcd (11) 4D: status={status}, desc={descAbcd}");
            if (descAbcd != IntPtr.Zero) dnnl_memory_desc_destroy(descAbcd);

            // Test with explicit 4 dims
            long[] dims4 = { 1, 16, 64, 64 };
            fixed (long* dims4Ptr = dims4)
            {
                status = dnnl_memory_desc_create_with_tag(out IntPtr descAbcd4, 4, dims4Ptr, 3, 11);
                Console.WriteLine($"format_tag abcd (11) 4D (short array): status={status}, desc={descAbcd4}");
                if (descAbcd4 != IntPtr.Zero) dnnl_memory_desc_destroy(descAbcd4);
            }

            // Test with different data type values to see which one is f32
            for (int dt = 0; dt <= 7; dt++)
            {
                status = dnnl_memory_desc_create_with_tag(out IntPtr descDt, 4, dimsPtr, dt, 1);
                Console.WriteLine($"data_type={dt}, format_any: status={status}");
                if (descDt != IntPtr.Zero) dnnl_memory_desc_destroy(descDt);
            }
            Console.Out.Flush();
        }

        // Test strides-based API (separate fixed block for safety)
        Console.WriteLine("\n--- Testing dnnl_memory_desc_create_with_strides ---");
        Console.Out.Flush();
        try
        {
            // NCHW strides: for dims [1, 16, 64, 64], strides are [16*64*64, 64*64, 64, 1]
            long[] stridesNchw = { 16L * 64 * 64, 64L * 64, 64, 1 };
            long[] dims4D = { 1, 16, 64, 64 };
            fixed (long* stridesPtr = stridesNchw)
            fixed (long* dimsPtr2 = dims4D)
            {
                status = dnnl_memory_desc_create_with_strides(out IntPtr descStrides, 4, dimsPtr2, 3, stridesPtr);
                Console.WriteLine($"Strides API result: status={status}, desc={descStrides}");
                Console.Out.Flush();
                if (descStrides != IntPtr.Zero)
                {
                    Console.WriteLine("SUCCESS: Strides-based memory descriptor creation works!");
                    dnnl_memory_desc_destroy(descStrides);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Strides test exception: {ex.GetType().Name}: {ex.Message}");
        }
        Console.Out.Flush();

        dnnl_engine_destroy(engine);

        // Test complete convolution
        Console.WriteLine("\n=== Complete Convolution Test ===");
        TestCompleteConvolution();
    }

    /// <summary>
    /// Test complete convolution to diagnose primitive_execute failures.
    /// Uses DNNL_MEMORY_ALLOCATE to let oneDNN allocate memory buffers.
    /// </summary>
    private static unsafe void TestCompleteConvolution()
    {
        // Struct size verification
        Console.WriteLine($"DnnlExecArg size: {Marshal.SizeOf<DnnlExecArg>()} (expected: 16)");

        // Create engine and stream
        int status = dnnl_engine_create(out IntPtr engine, 1, 0);
        if (status != 0)
        {
            Console.WriteLine($"Failed to create engine: {status}");
            return;
        }
        Console.WriteLine($"Engine: {engine}");

        status = dnnl_stream_create(out IntPtr stream, engine, 0);
        if (status != 0)
        {
            Console.WriteLine($"Failed to create stream: {status}");
            dnnl_engine_destroy(engine);
            return;
        }
        Console.WriteLine($"Stream: {stream}");

        // Small test dimensions: batch=1, inC=3, H=4, W=4, outC=2, kH=3, kW=3
        // Output: (H - kH + 2*pad) / stride + 1 = (4 - 3 + 0) / 1 + 1 = 2
        int batch = 1, inC = 3, H = 4, W = 4;
        int outC = 2, kH = 3, kW = 3;
        int outH = 2, outW = 2;

        long[] srcDims = { batch, inC, H, W };
        long[] weightsDims = { outC, inC, kH, kW };
        long[] dstDims = { batch, outC, outH, outW };

        long[] srcStrides = { inC * H * W, H * W, W, 1 };
        long[] weightsStrides = { inC * kH * kW, kH * kW, kW, 1 };
        long[] dstStrides = { outC * outH * outW, outH * outW, outW, 1 };

        long[] convStrides = { 1, 1 };
        long[] dilates = { 0, 0 };  // dilation-1
        long[] padL = { 0, 0 };
        long[] padR = { 0, 0 };

        IntPtr srcDesc = IntPtr.Zero, weightsDesc = IntPtr.Zero, dstDesc = IntPtr.Zero;
        IntPtr primDesc = IntPtr.Zero, primitive = IntPtr.Zero;
        IntPtr srcMem = IntPtr.Zero, weightsMem = IntPtr.Zero, dstMem = IntPtr.Zero;

        try
        {
            // Create memory descriptors using strides (NCHW format)
            fixed (long* srcDimsPtr = srcDims)
            fixed (long* srcStridesPtr = srcStrides)
            {
                status = dnnl_memory_desc_create_with_strides(out srcDesc, 4, srcDimsPtr, 3, srcStridesPtr);
                Console.WriteLine($"srcDesc create: status={status}, desc={srcDesc}");
            }
            if (status != 0) return;

            fixed (long* weightsDimsPtr = weightsDims)
            fixed (long* weightsStridesPtr = weightsStrides)
            {
                status = dnnl_memory_desc_create_with_strides(out weightsDesc, 4, weightsDimsPtr, 3, weightsStridesPtr);
                Console.WriteLine($"weightsDesc create: status={status}, desc={weightsDesc}");
            }
            if (status != 0) return;

            fixed (long* dstDimsPtr = dstDims)
            fixed (long* dstStridesPtr = dstStrides)
            {
                status = dnnl_memory_desc_create_with_strides(out dstDesc, 4, dstDimsPtr, 3, dstStridesPtr);
                Console.WriteLine($"dstDesc create: status={status}, desc={dstDesc}");
            }
            if (status != 0) return;

            // Create convolution primitive descriptor
            fixed (long* convStridesPtr = convStrides)
            fixed (long* dilatesPtr = dilates)
            fixed (long* padLPtr = padL)
            fixed (long* padRPtr = padR)
            {
                status = dnnl_convolution_forward_primitive_desc_create(
                    out primDesc, engine,
                    96,  // forward_inference
                    1,   // convolution_direct
                    srcDesc, weightsDesc, IntPtr.Zero, dstDesc,
                    convStridesPtr, dilatesPtr, padLPtr, padRPtr, IntPtr.Zero);
                Console.WriteLine($"primDesc create: status={status}, desc={primDesc}");
            }
            if (status != 0) return;

            // Create primitive
            status = dnnl_primitive_create(out primitive, primDesc);
            Console.WriteLine($"primitive create: status={status}, prim={primitive}");
            if (status != 0) return;

            // Query memory descriptors from primitive descriptor
            // The primitive may use different internal formats than what we specified
            IntPtr queriedSrcMd = dnnl_primitive_desc_query_md(primDesc, 129, 0);  // DNNL_QUERY_SRC_MD
            IntPtr queriedWeightsMd = dnnl_primitive_desc_query_md(primDesc, 131, 0);  // DNNL_QUERY_WEIGHTS_MD
            IntPtr queriedDstMd = dnnl_primitive_desc_query_md(primDesc, 133, 0);  // DNNL_QUERY_DST_MD
            IntPtr scratchpadMd = dnnl_primitive_desc_query_md(primDesc, 135, 0);  // DNNL_QUERY_SCRATCHPAD_MD

            Console.WriteLine($"Queried MDs: src={queriedSrcMd}, weights={queriedWeightsMd}, dst={queriedDstMd}, scratchpad={scratchpadMd}");

            if (queriedSrcMd == IntPtr.Zero || queriedWeightsMd == IntPtr.Zero || queriedDstMd == IntPtr.Zero)
            {
                Console.WriteLine("Query failed - got null memory descriptors");
                return;
            }

            // Get buffer sizes for queried descriptors
            UIntPtr srcBufSize = dnnl_memory_desc_get_size(queriedSrcMd);
            UIntPtr weightsBufSize = dnnl_memory_desc_get_size(queriedWeightsMd);
            UIntPtr dstBufSize = dnnl_memory_desc_get_size(queriedDstMd);
            Console.WriteLine($"Queried sizes: src={srcBufSize}, weights={weightsBufSize}, dst={dstBufSize}");

            // Allocate buffers using Marshal.AllocHGlobal (native heap allocation)
            IntPtr srcBuffer = Marshal.AllocHGlobal((int)(ulong)srcBufSize);
            IntPtr weightsBuffer = Marshal.AllocHGlobal((int)(ulong)weightsBufSize);
            IntPtr dstBuffer = Marshal.AllocHGlobal((int)(ulong)dstBufSize);

            Console.WriteLine($"Allocated (AllocHGlobal): src={srcBuffer}, weights={weightsBuffer}, dst={dstBuffer}");

            // Initialize data (using NCHW layout since that's what we specified)
            int srcSize = batch * inC * H * W;
            int weightsSize = outC * inC * kH * kW;
            float* srcPtr = (float*)srcBuffer;
            float* weightsPtr = (float*)weightsBuffer;
            for (int i = 0; i < srcSize; i++) srcPtr[i] = 1.0f;
            for (int i = 0; i < weightsSize; i++) weightsPtr[i] = 1.0f;
            Console.WriteLine("Data initialized");

            // Use QUERIED descriptors from primitive descriptor for memory creation
            // Create with DNNL_MEMORY_NONE first, then set_data_handle
            Console.WriteLine("\n--- Using QUERIED descriptors with set_data_handle ---");

            // Create memory with null handle first
            status = dnnl_memory_create(out srcMem, queriedSrcMd, engine, IntPtr.Zero);
            Console.WriteLine($"srcMem create (null handle): status={status}, mem={srcMem}");
            if (status != 0) { Marshal.FreeHGlobal(srcBuffer); Marshal.FreeHGlobal(weightsBuffer); Marshal.FreeHGlobal(dstBuffer); return; }

            status = dnnl_memory_create(out weightsMem, queriedWeightsMd, engine, IntPtr.Zero);
            Console.WriteLine($"weightsMem create (null handle): status={status}, mem={weightsMem}");
            if (status != 0) { Marshal.FreeHGlobal(srcBuffer); Marshal.FreeHGlobal(weightsBuffer); Marshal.FreeHGlobal(dstBuffer); return; }

            status = dnnl_memory_create(out dstMem, queriedDstMd, engine, IntPtr.Zero);
            Console.WriteLine($"dstMem create (null handle): status={status}, mem={dstMem}");
            if (status != 0) { Marshal.FreeHGlobal(srcBuffer); Marshal.FreeHGlobal(weightsBuffer); Marshal.FreeHGlobal(dstBuffer); return; }

            // Now set the data handles
            status = dnnl_memory_set_data_handle(srcMem, srcBuffer);
            Console.WriteLine($"srcMem set_data_handle: status={status}");
            if (status != 0) { Marshal.FreeHGlobal(srcBuffer); Marshal.FreeHGlobal(weightsBuffer); Marshal.FreeHGlobal(dstBuffer); return; }

            status = dnnl_memory_set_data_handle(weightsMem, weightsBuffer);
            Console.WriteLine($"weightsMem set_data_handle: status={status}");
            if (status != 0) { Marshal.FreeHGlobal(srcBuffer); Marshal.FreeHGlobal(weightsBuffer); Marshal.FreeHGlobal(dstBuffer); return; }

            status = dnnl_memory_set_data_handle(dstMem, dstBuffer);
            Console.WriteLine($"dstMem set_data_handle: status={status}");
            if (status != 0) { Marshal.FreeHGlobal(srcBuffer); Marshal.FreeHGlobal(weightsBuffer); Marshal.FreeHGlobal(dstBuffer); return; }

            // Check if scratchpad is needed
            IntPtr scratchpadMem = IntPtr.Zero;
            IntPtr scratchpadBuffer = IntPtr.Zero;
            int nargs = 3;
            if (scratchpadMd != IntPtr.Zero)
            {
                UIntPtr scratchpadSize = dnnl_memory_desc_get_size(scratchpadMd);
                Console.WriteLine($"Scratchpad required, size={scratchpadSize}");
                if ((ulong)scratchpadSize > 0)
                {
                    scratchpadBuffer = Marshal.AllocHGlobal((int)(ulong)scratchpadSize);
                    if (scratchpadBuffer != IntPtr.Zero)
                    {
                        status = dnnl_memory_create(out scratchpadMem, scratchpadMd, engine, scratchpadBuffer);
                        if (status == 0)
                        {
                            Console.WriteLine($"Scratchpad memory created: {scratchpadMem}");
                            nargs = 4;
                        }
                        else
                        {
                            Console.WriteLine($"Failed to create scratchpad memory: {status}");
                            Marshal.FreeHGlobal(scratchpadBuffer);
                            scratchpadBuffer = IntPtr.Zero;
                        }
                    }
                }
            }
            else
            {
                Console.WriteLine("No scratchpad required");
            }

            // Execute convolution with CORRECT argument indices!
            // DNNL_ARG_SRC = 1, DNNL_ARG_DST = 17 (NOT 2!), DNNL_ARG_WEIGHTS = 33, DNNL_ARG_SCRATCHPAD = 80
            Console.WriteLine("\n--- Executing convolution with CORRECT arg indices ---");
            Console.WriteLine("DNNL_ARG_SRC=1, DNNL_ARG_DST=17, DNNL_ARG_WEIGHTS=33, DNNL_ARG_SCRATCHPAD=80");

            // Create scratchpad memory (even with size 0, it must be provided)
            // Use queriedDstMd as dummy descriptor and null handle for zero-size scratchpad
            if (scratchpadMem == IntPtr.Zero)
            {
                // Create a dummy scratchpad with null handle - oneDNN accepts this for zero-size scratchpads
                status = dnnl_memory_create(out scratchpadMem, dstDesc, engine, IntPtr.Zero);
                Console.WriteLine($"Dummy scratchpad created: status={status}, mem={scratchpadMem}");
            }

            // Execute with struct
            DnnlExecArg* args = stackalloc DnnlExecArg[4];
            args[0].Arg = 1;   // DNNL_ARG_SRC
            args[0].Memory = srcMem;
            args[1].Arg = 33;  // DNNL_ARG_WEIGHTS
            args[1].Memory = weightsMem;
            args[2].Arg = 17;  // DNNL_ARG_DST (CORRECT: 17, not 2!)
            args[2].Memory = dstMem;
            args[3].Arg = 80;  // DNNL_ARG_SCRATCHPAD
            args[3].Memory = scratchpadMem;

            Console.WriteLine($"Arguments:");
            Console.WriteLine($"  SRC (1): mem={srcMem}");
            Console.WriteLine($"  WEIGHTS (33): mem={weightsMem}");
            Console.WriteLine($"  DST (17): mem={dstMem}");
            Console.WriteLine($"  SCRATCHPAD (80): mem={scratchpadMem}");

            status = dnnl_primitive_execute(primitive, stream, 4, args);
            Console.WriteLine($"primitive_execute: status={status}");

            if (status == 0)
            {
                // Wait for completion
                status = dnnl_stream_wait(stream);
                Console.WriteLine($"stream_wait: status={status}");

                // Read result
                float* dstPtr = (float*)dstBuffer;
                int dstSize = batch * outC * outH * outW;
                Console.WriteLine("Output values:");
                for (int i = 0; i < dstSize; i++)
                {
                    Console.WriteLine($"  dst[{i}] = {dstPtr[i]}");
                }

                Console.WriteLine("\n*** CONVOLUTION SUCCESS! ***");
            }
            else
            {
                Console.WriteLine($"\n*** CONVOLUTION FAILED with status={status} ***");
                Console.WriteLine("Status 2 = invalid_arguments");
            }

            // Free user buffers
            Marshal.FreeHGlobal(srcBuffer);
            Marshal.FreeHGlobal(weightsBuffer);
            Marshal.FreeHGlobal(dstBuffer);
            if (scratchpadBuffer != IntPtr.Zero) Marshal.FreeHGlobal(scratchpadBuffer);
        }
        finally
        {
            if (dstMem != IntPtr.Zero) dnnl_memory_destroy(dstMem);
            if (weightsMem != IntPtr.Zero) dnnl_memory_destroy(weightsMem);
            if (srcMem != IntPtr.Zero) dnnl_memory_destroy(srcMem);
            if (primitive != IntPtr.Zero) dnnl_primitive_destroy(primitive);
            if (primDesc != IntPtr.Zero) dnnl_primitive_desc_destroy(primDesc);
            if (dstDesc != IntPtr.Zero) dnnl_memory_desc_destroy(dstDesc);
            if (weightsDesc != IntPtr.Zero) dnnl_memory_desc_destroy(weightsDesc);
            if (srcDesc != IntPtr.Zero) dnnl_memory_desc_destroy(srcDesc);
            dnnl_stream_destroy(stream);
            dnnl_engine_destroy(engine);
        }
    }

    // Execution argument struct - must match native dnnl_exec_arg_t
    // Using Sequential layout with natural padding (CLR adds 4 bytes after int for pointer alignment)
    [StructLayout(LayoutKind.Sequential)]
    private struct DnnlExecArg
    {
        public int Arg;
        public IntPtr Memory;
    }

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_engine_create(out IntPtr engine, int kind, int index);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_engine_destroy(IntPtr engine);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe int dnnl_memory_desc_create_with_tag(
        out IntPtr memoryDesc,
        int ndims,
        long* dims,
        int dataType,
        int formatTag);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_memory_desc_destroy(IntPtr memoryDesc);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe int dnnl_memory_desc_create_with_strides(
        out IntPtr memoryDesc,
        int ndims,
        long* dims,
        int dataType,
        long* strides);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_stream_create(out IntPtr stream, IntPtr engine, uint flags);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_stream_destroy(IntPtr stream);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_stream_wait(IntPtr stream);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_memory_create(
        out IntPtr memory,
        IntPtr memoryDesc,
        IntPtr engine,
        IntPtr handle);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_memory_destroy(IntPtr memory);

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl)]
    private static extern int dnnl_memory_get_data_handle(IntPtr memory, out IntPtr handle);

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
    private static extern UIntPtr dnnl_memory_desc_get_size(IntPtr memoryDesc);

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

    [DllImport("dnnl", CallingConvention = CallingConvention.Cdecl, EntryPoint = "dnnl_primitive_execute")]
    private static extern int dnnl_primitive_execute_raw(
        IntPtr primitive,
        IntPtr stream,
        int nargs,
        IntPtr args);

    // Windows memory allocation for aligned buffers
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr VirtualAlloc(IntPtr lpAddress, UIntPtr dwSize, uint flAllocationType, uint flProtect);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool VirtualFree(IntPtr lpAddress, UIntPtr dwSize, uint dwFreeType);

    private const uint MEM_COMMIT = 0x1000;
    private const uint MEM_RESERVE = 0x2000;
    private const uint MEM_RELEASE = 0x8000;
    private const uint PAGE_READWRITE = 0x04;
}
#endif
