using BenchmarkDotNet.Running;
using AiDotNet.Tensors.Benchmarks;

namespace AiDotNet.Tensors.Benchmarks;

class Program
{
    static void Main(string[] args)
    {
        // Run quick performance test first for immediate feedback
        if (args.Length == 0 || args[0] == "--quick")
        {
            QuickPerformanceTest.Run();
            return;
        }

        // Run full BenchmarkDotNet suite if requested
        if (args[0] == "--full")
        {
            var summary = BenchmarkRunner.Run<TrigonometricOperatorBenchmarks>();
            return;
        }

        // Run linear algebra benchmarks
        if (args[0] == "--linalg")
        {
            BenchmarkRunner.Run<LinearAlgebraBenchmarks>();
            BenchmarkRunner.Run<SmallMatrixBenchmarks>();
            return;
        }

#if !NET462
        // Run cuBLAS vs ILGPU GEMM benchmark
        if (args[0] == "--cublas")
        {
            CuBlasGemmBenchmark.Run();
            return;
        }

        // Run OpenCL GEMM benchmark (AMD/Intel GPUs)
        if (args[0] == "--opencl")
        {
            OpenClGemmBenchmark.Run();
            return;
        }

        // Run CLBlast vs AiDotNet OpenCL comparison benchmark
        if (args[0] == "--clblast")
        {
            ClBlastBenchmark.Run();
            return;
        }

        // Run DirectGpu comprehensive benchmark (all 10 optimizations)
        if (args[0] == "--directgpu")
        {
            DirectGpuGemmBenchmark.RunComprehensive();
            return;
        }
#endif

        Console.WriteLine("Usage:");
        Console.WriteLine("  --quick    : Run quick performance validation (default)");
        Console.WriteLine("  --full     : Run full BenchmarkDotNet suite (trigonometric)");
        Console.WriteLine("  --linalg   : Run linear algebra benchmarks vs MathNet.Numerics");
#if !NET462
        Console.WriteLine("  --cublas   : Run cuBLAS vs ILGPU GEMM benchmark");
        Console.WriteLine("  --opencl   : Run OpenCL GEMM benchmark (AMD/Intel GPUs)");
        Console.WriteLine("  --clblast  : Run CLBlast vs AiDotNet OpenCL comparison (AMD/Intel)");
        Console.WriteLine("  --directgpu: Run DirectGpu comprehensive benchmark (all 10 optimizations)");
#endif
    }
}
