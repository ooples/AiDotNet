using System;
using AiDotNet.InferenceOptimization;
using AiDotNet.InferenceOptimization.Kernels;
using AiDotNet.InferenceOptimization.CpuOptimization;
using AiDotNet.InferenceOptimization.Profiling;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.InferenceOptimization.Examples
{
    /// <summary>
    /// Basic usage examples for the inference optimization module
    /// </summary>
    public class BasicUsageExample
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("=== AiDotNet Inference Optimization Examples ===\n");

            // Example 1: Platform detection
            PlatformDetectionExample();

            // Example 2: Using optimized GEMM
            OptimizedGemmExample();

            // Example 3: Using fused attention
            FusedAttentionExample();

            // Example 4: Custom operator registration
            CustomOperatorExample();

            // Example 5: Performance profiling
            ProfilingExample();

            // Example 6: CPU optimization utilities
            CpuOptimizationExample();

            Console.WriteLine("\n=== Examples Complete ===");
        }

        static void PlatformDetectionExample()
        {
            Console.WriteLine("### Example 1: Platform Detection ###\n");

            // Get platform capabilities
            var caps = PlatformDetector.Capabilities;

            Console.WriteLine($"Architecture: {caps.Architecture}");
            Console.WriteLine($"Processor Count: {caps.ProcessorCount}");
            Console.WriteLine($"Best SIMD: {caps.GetBestSimdSet()}");
            Console.WriteLine($"Has AVX2: {caps.HasAVX2}");
            Console.WriteLine($"Has NEON: {caps.HasNeon}");
            Console.WriteLine($"Has CUDA: {caps.HasCudaSupport}");

            // Print detailed capabilities
            Console.WriteLine("\n" + PlatformDetector.GetCapabilitiesDescription());
        }

        static void OptimizedGemmExample()
        {
            Console.WriteLine("### Example 2: Optimized GEMM (Matrix Multiplication) ###\n");

            // Initialize optimization system
            OptimizationInitializer.Initialize(enableProfiling: false);

            // Create matrices
            int size = 500;
            var matrixA = new Tensor<float>(new[] { size, size });
            var matrixB = new Tensor<float>(new[] { size, size });

            var random = new Random(42);
            for (int i = 0; i < matrixA.Data.Length; i++)
            {
                matrixA.Data[i] = (float)random.NextDouble();
                matrixB.Data[i] = (float)random.NextDouble();
            }

            // Use optimized GEMM kernel
            var gemmKernel = new GemmKernel();

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var result = gemmKernel.Execute(matrixA, matrixB);
            stopwatch.Stop();

            Console.WriteLine($"Matrix multiplication ({size}x{size}) completed in {stopwatch.ElapsedMilliseconds} ms");
            Console.WriteLine($"Expected speedup: {gemmKernel.EstimatedSpeedup():F1}x over naive implementation");
            Console.WriteLine($"Result dimensions: [{result.Dimensions[0]}, {result.Dimensions[1]}]");
            Console.WriteLine();
        }

        static void FusedAttentionExample()
        {
            Console.WriteLine("### Example 3: Fused Attention Kernel ###\n");

            // Initialize
            OptimizationInitializer.Initialize(enableProfiling: false);

            // Create Q, K, V tensors for attention
            int batchSize = 2;
            int seqLen = 128;
            int dModel = 64;

            var q = new Tensor<float>(new[] { batchSize, seqLen, dModel });
            var k = new Tensor<float>(new[] { batchSize, seqLen, dModel });
            var v = new Tensor<float>(new[] { batchSize, seqLen, dModel });

            var random = new Random(42);
            for (int i = 0; i < q.Data.Length; i++)
            {
                q.Data[i] = (float)random.NextDouble();
                k.Data[i] = (float)random.NextDouble();
                v.Data[i] = (float)random.NextDouble();
            }

            // Use fused attention kernel
            var attentionKernel = new AttentionKernel();

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var attended = attentionKernel.Execute(q, k, v);
            stopwatch.Stop();

            Console.WriteLine($"Fused attention (batch={batchSize}, seq_len={seqLen}, d_model={dModel})");
            Console.WriteLine($"Completed in {stopwatch.ElapsedMilliseconds} ms");
            Console.WriteLine($"Expected speedup: {attentionKernel.EstimatedSpeedup():F1}x");
            Console.WriteLine($"Result shape: [{attended.Dimensions[0]}, {attended.Dimensions[1]}, {attended.Dimensions[2]}]");

            // Multi-head attention
            stopwatch.Restart();
            var multiHead = attentionKernel.MultiHeadAttention(q, k, v, numHeads: 8);
            stopwatch.Stop();

            Console.WriteLine($"\nMulti-head attention (8 heads) completed in {stopwatch.ElapsedMilliseconds} ms");
            Console.WriteLine();
        }

        static void CustomOperatorExample()
        {
            Console.WriteLine("### Example 4: Custom Operator Registration ###\n");

            // Initialize
            OptimizationInitializer.Initialize(enableProfiling: false);

            // Register custom operators
            var registry = CustomOperatorRegistry.Instance;

            // Check what operators are available
            Console.WriteLine("Registered operators:");
            foreach (var name in registry.GetRegisteredOperatorNames())
            {
                var op = registry.GetOperator(name);
                Console.WriteLine($"  - {name}: {(op.IsSupported() ? "✓ Supported" : "✗ Not supported")}");
                Console.WriteLine($"    Version: {op.Version}, Priority: {op.Priority}, Speedup: {op.EstimatedSpeedup():F1}x");
            }

            // Get detailed operator info
            Console.WriteLine("\nDetailed operator information:");
            var operatorInfo = registry.GetOperatorInfo();
            foreach (var kvp in operatorInfo)
            {
                Console.WriteLine($"\n{kvp.Key}:");
                foreach (var info in kvp.Value)
                {
                    Console.WriteLine($"  Type: {info.Type}");
                    Console.WriteLine($"  Supported: {info.IsSupported}");
                    Console.WriteLine($"  Priority: {info.Priority}");
                    Console.WriteLine($"  Estimated Speedup: {info.EstimatedSpeedup:F1}x");
                }
            }
            Console.WriteLine();
        }

        static void ProfilingExample()
        {
            Console.WriteLine("### Example 5: Performance Profiling ###\n");

            // Initialize with profiling enabled
            OptimizationInitializer.Initialize(enableProfiling: true);

            var profiler = PerformanceProfiler.Instance;
            profiler.Enabled = true;

            // Perform some operations
            var random = new Random(42);

            for (int i = 0; i < 5; i++)
            {
                using (profiler.Profile("MatrixMultiplication"))
                {
                    var gemmKernel = new GemmKernel();
                    var a = new Tensor<float>(new[] { 256, 256 });
                    var b = new Tensor<float>(new[] { 256, 256 });

                    for (int j = 0; j < a.Data.Length; j++)
                    {
                        a.Data[j] = (float)random.NextDouble();
                        b.Data[j] = (float)random.NextDouble();
                    }

                    var result = gemmKernel.Execute(a, b);
                }

                using (profiler.Profile("VectorOperations"))
                {
                    var arr = new float[100000];
                    for (int j = 0; j < arr.Length; j++)
                    {
                        arr[j] = (float)random.NextDouble();
                    }

                    unsafe
                    {
                        fixed (float* pArr = arr)
                        {
                            float sum = SimdKernels.Sum(pArr, arr.Length);
                        }
                    }
                }
            }

            // Generate performance report
            Console.WriteLine(profiler.GenerateReport());

            // Reset statistics
            profiler.Clear();
            Console.WriteLine();
        }

        static void CpuOptimizationExample()
        {
            Console.WriteLine("### Example 6: CPU Optimization Utilities ###\n");

            // Cache optimization
            Console.WriteLine("Cache-aware tile sizes:");
            Console.WriteLine($"  L1 Block Size: {CacheOptimizer.L1BlockSize} elements");
            Console.WriteLine($"  L2 Block Size: {CacheOptimizer.L2BlockSize} elements");
            Console.WriteLine($"  L3 Block Size: {CacheOptimizer.L3BlockSize} elements");

            // Optimal tiling for matrix operations
            int m = 1000, n = 1000, k = 1000;
            var (tileM, tileN, tileK) = CacheOptimizer.ComputeOptimalTiling(m, n, k);
            Console.WriteLine($"\nOptimal tiling for {m}x{n}x{k} operation:");
            Console.WriteLine($"  Tile M: {tileM}");
            Console.WriteLine($"  Tile N: {tileN}");
            Console.WriteLine($"  Tile K: {tileK}");

            // Loop optimization
            Console.WriteLine("\nLoop optimization example:");
            int matrixSize = 512;
            int tileSize = LoopOptimizer.DetermineOptimalTileSize(matrixSize);
            Console.WriteLine($"  Optimal tile size for {matrixSize}x{matrixSize} matrix: {tileSize}");

            // Demonstrate tiled loop
            var data = new float[matrixSize, matrixSize];
            int tilesProcessed = 0;

            LoopOptimizer.Tile2D(matrixSize, matrixSize, tileSize,
                (iStart, iEnd, jStart, jEnd) =>
                {
                    // Process tile
                    for (int i = iStart; i < iEnd; i++)
                    {
                        for (int j = jStart; j < jEnd; j++)
                        {
                            data[i, j] = i + j;
                        }
                    }
                    tilesProcessed++;
                });

            Console.WriteLine($"  Processed {tilesProcessed} tiles");

            // Cache miss estimation
            int dataSize = 1000000;
            int cacheSize = PlatformDetector.Capabilities.L1CacheSize;
            double missRate = CacheOptimizer.EstimateCacheMisses(dataSize, 1, cacheSize, 64);
            Console.WriteLine($"\nCache miss estimation:");
            Console.WriteLine($"  Sequential access miss rate: ~{missRate / (dataSize / 64) * 100:F1}%");

            double stridedMissRate = CacheOptimizer.EstimateCacheMisses(dataSize, 128, cacheSize, 64);
            Console.WriteLine($"  Strided access (stride=128) miss rate: ~{stridedMissRate / (dataSize / 64) * 100:F1}%");

            Console.WriteLine();
        }
    }
}
