// Copyright (c) AiDotNet. All rights reserved.
// Shared OpenCL build options for consistent kernel compilation.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    internal static class OpenClBuildOptions
    {
        // Aggressive optimization flags for maximum performance.
        public const string OptimizationFlags =
            "-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-no-signed-zeros";
    }
}
