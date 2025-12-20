using System;
using AiDotNet.InferenceOptimization.Kernels;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;

namespace AiDotNet.InferenceOptimization
{
    /// <summary>
    /// Initializes and registers all optimized kernels and operators
    /// </summary>
    public static class OptimizationInitializer
    {
        private static bool _initialized = false;
        private static readonly object _lock = new object();

        /// <summary>
        /// Initializes the inference optimization system
        /// </summary>
        public static void Initialize(bool enableProfiling = false)
        {
            lock (_lock)
            {
                if (_initialized)
                    return;

                // Enable profiling if requested
                PerformanceProfiler.Instance.Enabled = enableProfiling;

                // Register optimized kernels
                RegisterKernels();

                // Print platform capabilities
                LogPlatformInfo();

                _initialized = true;
            }
        }

        private static void RegisterKernels()
        {
            var registry = CustomOperatorRegistry.Instance;

            // Register GEMM kernel
            registry.Register(new GemmKernel());

            // Register Attention kernel
            registry.Register(new AttentionKernel());

            // Register Convolution kernel
            registry.Register(new ConvolutionKernel());

            // Future: Register GPU kernels when available
            // if (PlatformDetector.Capabilities.HasCudaSupport)
            // {
            //     registry.Register(new CudaGemmKernel());
            //     registry.Register(new CudaConvolutionKernel());
            // }
        }

        private static void LogPlatformInfo()
        {
            Console.WriteLine("=== AiDotNet Inference Optimization ===");
            Console.WriteLine(PlatformDetector.GetCapabilitiesDescription());
            Console.WriteLine();
            Console.WriteLine("Registered Operators:");

            var operatorInfo = CustomOperatorRegistry.Instance.GetOperatorInfo();
            foreach (var kvp in operatorInfo)
            {
                Console.WriteLine($"  {kvp.Key}:");
                foreach (var info in kvp.Value)
                {
                    var status = info.IsSupported ? "✓" : "✗";
                    Console.WriteLine($"    {status} {info.Version} - Priority: {info.Priority}, Speedup: {info.EstimatedSpeedup:F1}x");
                }
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Gets a performance summary
        /// </summary>
        public static string GetPerformanceSummary()
        {
            if (!_initialized)
                return "Optimization system not initialized.";

            var report = PerformanceProfiler.Instance.GenerateReport();
            return report;
        }

        /// <summary>
        /// Resets all profiling statistics
        /// </summary>
        public static void ResetStatistics()
        {
            PerformanceProfiler.Instance.Clear();
        }

        /// <summary>
        /// Enables or disables profiling at runtime
        /// </summary>
        public static void SetProfilingEnabled(bool enabled)
        {
            PerformanceProfiler.Instance.Enabled = enabled;
        }
    }
}
