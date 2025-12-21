using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Platform characteristics for hardware cost estimation.
    /// Contains performance metrics like GFLOPS, memory bandwidth, and energy efficiency.
    /// </summary>
    public class PlatformCharacteristics
    {
        /// <summary>
        /// Peak computational throughput in GFLOPS (billions of floating-point operations per second).
        /// </summary>
        public double GFlops { get; set; }

        /// <summary>
        /// Memory bandwidth in GB/s.
        /// </summary>
        public double MemoryBandwidthGBps { get; set; }

        /// <summary>
        /// Energy consumption per GFLOP in millijoules.
        /// </summary>
        public double EnergyPerGFlop { get; set; }

        /// <summary>
        /// Memory access energy cost in millijoules per GB.
        /// </summary>
        public double EnergyPerGBMemory { get; set; }
    }

    /// <summary>
    /// Models hardware costs for neural architecture search operations using FLOP-based estimation.
    /// Supports latency, energy, and memory cost estimation for different hardware platforms.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class HardwareCostModel<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly HardwarePlatform _platform;
        private readonly PlatformCharacteristics _characteristics;
        private readonly Dictionary<string, double> _calibrationFactors;
        private readonly HashSet<string> _unknownOperations;

        /// <summary>
        /// Gets the target hardware platform.
        /// </summary>
        public HardwarePlatform Platform => _platform;

        /// <summary>
        /// Gets the platform characteristics used for cost estimation.
        /// </summary>
        public PlatformCharacteristics Characteristics => _characteristics;

        public HardwareCostModel(HardwarePlatform platform = HardwarePlatform.Mobile)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _platform = platform;
            _characteristics = GetPlatformCharacteristics(platform);
            _calibrationFactors = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
            _unknownOperations = new HashSet<string>(StringComparer.Ordinal);
        }

        /// <summary>
        /// Sets a calibration factor for a specific operation based on actual measurements.
        /// </summary>
        /// <param name="operation">The operation name</param>
        /// <param name="factor">The calibration factor (1.0 = no adjustment)</param>
        public void SetCalibrationFactor(string operation, double factor)
        {
            if (string.IsNullOrEmpty(operation))
                throw new ArgumentNullException(nameof(operation));
            if (factor <= 0)
                throw new ArgumentOutOfRangeException(nameof(factor), "Calibration factor must be positive.");

            _calibrationFactors[operation] = factor;
        }

        /// <summary>
        /// Gets the calibration factor for an operation, or 1.0 if not set.
        /// </summary>
        public double GetCalibrationFactor(string operation)
        {
            return _calibrationFactors.TryGetValue(operation, out var factor) ? factor : 1.0;
        }

        /// <summary>
        /// Clears all calibration factors.
        /// </summary>
        public void ClearCalibration()
        {
            _calibrationFactors.Clear();
        }

        /// <summary>
        /// Estimates the hardware cost for a given operation using FLOP-based calculation.
        /// </summary>
        public HardwareCost<T> EstimateOperationCost(string operation, int inputChannels, int outputChannels, int spatialSize)
        {
            if (inputChannels <= 0) inputChannels = 1;
            if (outputChannels <= 0) outputChannels = 1;
            if (spatialSize <= 0) spatialSize = 1;

            var flops = CalculateOperationFlops(operation, inputChannels, outputChannels, spatialSize);
            var memoryBytes = CalculateOperationMemory(operation, inputChannels, outputChannels, spatialSize);

            // Apply calibration factor
            var calibration = GetCalibrationFactor(operation);
            flops *= calibration;

            // Calculate latency: time = FLOPs / GFLOPS (convert to ms)
            var latencyMs = (flops / 1e9) / _characteristics.GFlops * 1000.0;

            // Calculate memory transfer time contribution
            var memoryGb = memoryBytes / (1024.0 * 1024.0 * 1024.0);
            var memoryLatencyMs = (memoryGb / _characteristics.MemoryBandwidthGBps) * 1000.0;

            // Total latency is max of compute and memory bound (simplified roofline model)
            var totalLatencyMs = Math.Max(latencyMs, memoryLatencyMs);

            // Calculate energy: FLOPs * energy per GFLOP + memory access energy
            var computeEnergy = (flops / 1e9) * _characteristics.EnergyPerGFlop;
            var memoryEnergy = memoryGb * _characteristics.EnergyPerGBMemory;
            var totalEnergy = computeEnergy + memoryEnergy;

            // Memory in MB (for weight storage, not activations)
            var memoryMb = memoryBytes / (1024.0 * 1024.0);

            return new HardwareCost<T>
            {
                Latency = _ops.FromDouble(totalLatencyMs),
                Energy = _ops.FromDouble(totalEnergy),
                Memory = _ops.FromDouble(memoryMb)
            };
        }

        /// <summary>
        /// Calculates the number of floating-point operations for a given operation.
        /// </summary>
        public static double CalculateOperationFlops(string operation, int inputChannels, int outputChannels, int spatialSize)
        {
            if (string.IsNullOrEmpty(operation))
                return 0;

            var spatialElements = (double)spatialSize * spatialSize;

            return operation.ToLowerInvariant() switch
            {
                // Identity/skip connection: minimal computation
                "identity" or "skip" or "none" => spatialElements * inputChannels * 2,

                // Standard convolutions: 2 * K^2 * Cin * Cout * H * W
                "conv1x1" => 2.0 * 1 * 1 * inputChannels * outputChannels * spatialElements,
                "conv3x3" => 2.0 * 9 * inputChannels * outputChannels * spatialElements,
                "conv5x5" => 2.0 * 25 * inputChannels * outputChannels * spatialElements,
                "conv7x7" => 2.0 * 49 * inputChannels * outputChannels * spatialElements,

                // Depthwise convolutions: 2 * K^2 * C * H * W (no channel mixing)
                "depthwise_conv3x3" or "dw_conv3x3" => 2.0 * 9 * inputChannels * spatialElements,
                "depthwise_conv5x5" or "dw_conv5x5" => 2.0 * 25 * inputChannels * spatialElements,
                "depthwise_conv7x7" or "dw_conv7x7" => 2.0 * 49 * inputChannels * spatialElements,

                // Separable convolutions: depthwise + pointwise
                "separable_conv3x3" or "sep_conv3x3" =>
                    2.0 * 9 * inputChannels * spatialElements + 2.0 * inputChannels * outputChannels * spatialElements,
                "separable_conv5x5" or "sep_conv5x5" =>
                    2.0 * 25 * inputChannels * spatialElements + 2.0 * inputChannels * outputChannels * spatialElements,

                // Dilated convolutions (same FLOPs as regular, different memory pattern)
                "dilated_conv3x3" or "dil_conv3x3" => 2.0 * 9 * inputChannels * outputChannels * spatialElements,
                "dilated_conv5x5" or "dil_conv5x5" => 2.0 * 25 * inputChannels * outputChannels * spatialElements,

                // Pooling operations: comparisons per spatial location
                "maxpool3x3" or "max_pool_3x3" => 9.0 * inputChannels * spatialElements,
                "avgpool3x3" or "avg_pool_3x3" => 9.0 * inputChannels * spatialElements,
                "maxpool2x2" => 4.0 * inputChannels * spatialElements,
                "avgpool2x2" => 4.0 * inputChannels * spatialElements,
                "global_avgpool" or "gap" => inputChannels * spatialElements,

                // Batch normalization: 4 ops per element (subtract mean, divide std, scale, shift)
                "batch_norm" or "bn" => 4.0 * inputChannels * spatialElements,

                // Activation functions
                "relu" => inputChannels * spatialElements, // One comparison
                "swish" or "silu" => 4.0 * inputChannels * spatialElements, // sigmoid + multiply
                "gelu" => 8.0 * inputChannels * spatialElements, // More complex approximation
                "sigmoid" => 4.0 * inputChannels * spatialElements,

                // Squeeze-and-Excitation block: global pool + FC + FC + scale
                "se_block" or "squeeze_excite" =>
                    inputChannels * spatialElements + // Global pool
                    2.0 * inputChannels * (inputChannels / 16) + // FC reduce
                    2.0 * (inputChannels / 16) * inputChannels + // FC expand
                    inputChannels * spatialElements, // Scale

                // MBConv blocks (Mobile Inverted Bottleneck)
                "mbconv_k3_e1" => CalculateMBConvFlops(inputChannels, outputChannels, spatialElements, 3, 1),
                "mbconv_k3_e3" => CalculateMBConvFlops(inputChannels, outputChannels, spatialElements, 3, 3),
                "mbconv_k3_e6" => CalculateMBConvFlops(inputChannels, outputChannels, spatialElements, 3, 6),
                "mbconv_k5_e3" => CalculateMBConvFlops(inputChannels, outputChannels, spatialElements, 5, 3),
                "mbconv_k5_e6" => CalculateMBConvFlops(inputChannels, outputChannels, spatialElements, 5, 6),
                "mbconv_k7_e6" => CalculateMBConvFlops(inputChannels, outputChannels, spatialElements, 7, 6),

                // Unknown operation - log warning and return conservative estimate
                _ => HandleUnknownOperation(operation, inputChannels, outputChannels, spatialElements)
            };
        }

        private static double CalculateMBConvFlops(int inputChannels, int outputChannels, double spatialElements, int kernelSize, int expansion)
        {
            var expandedChannels = inputChannels * expansion;
            var k2 = kernelSize * kernelSize;

            // Expansion 1x1 conv (skip if expansion=1)
            var expansionFlops = expansion > 1 ? 2.0 * inputChannels * expandedChannels * spatialElements : 0;
            // Depthwise conv
            var dwFlops = 2.0 * k2 * expandedChannels * spatialElements;
            // Projection 1x1 conv
            var projectionFlops = 2.0 * expandedChannels * outputChannels * spatialElements;

            return expansionFlops + dwFlops + projectionFlops;
        }

        private static double HandleUnknownOperation(string operation, int inputChannels, int outputChannels, double spatialElements)
        {
            // Return a conservative estimate based on a 3x3 conv
            System.Diagnostics.Trace.TraceWarning(
                $"AiDotNet.AutoML.NAS: Unknown operation '{operation}' in FLOP calculation. Using conv3x3 estimate.");
            return 2.0 * 9 * inputChannels * outputChannels * spatialElements;
        }

        /// <summary>
        /// Calculates the memory footprint (weights) for a given operation in bytes.
        /// </summary>
        public static double CalculateOperationMemory(string operation, int inputChannels, int outputChannels, int spatialSize)
        {
            if (string.IsNullOrEmpty(operation))
                return 0;

            const double bytesPerParam = 4.0; // float32

            return operation.ToLowerInvariant() switch
            {
                "identity" or "skip" or "none" => 0,

                "conv1x1" => bytesPerParam * 1 * 1 * inputChannels * outputChannels,
                "conv3x3" => bytesPerParam * 9 * inputChannels * outputChannels,
                "conv5x5" => bytesPerParam * 25 * inputChannels * outputChannels,
                "conv7x7" => bytesPerParam * 49 * inputChannels * outputChannels,

                "depthwise_conv3x3" or "dw_conv3x3" => bytesPerParam * 9 * inputChannels,
                "depthwise_conv5x5" or "dw_conv5x5" => bytesPerParam * 25 * inputChannels,
                "depthwise_conv7x7" or "dw_conv7x7" => bytesPerParam * 49 * inputChannels,

                "separable_conv3x3" or "sep_conv3x3" =>
                    bytesPerParam * (9 * inputChannels + inputChannels * outputChannels),
                "separable_conv5x5" or "sep_conv5x5" =>
                    bytesPerParam * (25 * inputChannels + inputChannels * outputChannels),

                "dilated_conv3x3" or "dil_conv3x3" => bytesPerParam * 9 * inputChannels * outputChannels,
                "dilated_conv5x5" or "dil_conv5x5" => bytesPerParam * 25 * inputChannels * outputChannels,

                // Pooling has no learnable parameters
                "maxpool3x3" or "max_pool_3x3" or "avgpool3x3" or "avg_pool_3x3" => 0,
                "maxpool2x2" or "avgpool2x2" or "global_avgpool" or "gap" => 0,

                // Batch norm: gamma, beta, running_mean, running_var
                "batch_norm" or "bn" => bytesPerParam * 4 * inputChannels,

                // Activations have no parameters
                "relu" or "swish" or "silu" or "gelu" or "sigmoid" => 0,

                // SE block: two FC layers
                "se_block" or "squeeze_excite" =>
                    bytesPerParam * (inputChannels * (inputChannels / 16) + (inputChannels / 16) * inputChannels),

                // MBConv memory
                "mbconv_k3_e1" => CalculateMBConvMemory(inputChannels, outputChannels, 3, 1),
                "mbconv_k3_e3" => CalculateMBConvMemory(inputChannels, outputChannels, 3, 3),
                "mbconv_k3_e6" => CalculateMBConvMemory(inputChannels, outputChannels, 3, 6),
                "mbconv_k5_e3" => CalculateMBConvMemory(inputChannels, outputChannels, 5, 3),
                "mbconv_k5_e6" => CalculateMBConvMemory(inputChannels, outputChannels, 5, 6),
                "mbconv_k7_e6" => CalculateMBConvMemory(inputChannels, outputChannels, 7, 6),

                _ => bytesPerParam * 9 * inputChannels * outputChannels // Default to conv3x3 estimate
            };
        }

        private static double CalculateMBConvMemory(int inputChannels, int outputChannels, int kernelSize, int expansion)
        {
            const double bytesPerParam = 4.0;
            var expandedChannels = inputChannels * expansion;
            var k2 = kernelSize * kernelSize;

            var expansionMemory = expansion > 1 ? bytesPerParam * inputChannels * expandedChannels : 0;
            var dwMemory = bytesPerParam * k2 * expandedChannels;
            var projectionMemory = bytesPerParam * expandedChannels * outputChannels;

            return expansionMemory + dwMemory + projectionMemory;
        }

        /// <summary>
        /// Estimates the total cost for an entire architecture.
        /// </summary>
        public HardwareCost<T> EstimateArchitectureCost(Architecture<T> architecture, int inputChannels, int spatialSize)
        {
            var totalCost = new HardwareCost<T>
            {
                Latency = _ops.Zero,
                Energy = _ops.Zero,
                Memory = _ops.Zero
            };

            if (architecture == null || architecture.Operations.Count == 0)
                return totalCost;

            foreach (var (toNode, fromNode, operation) in architecture.Operations)
            {
                var fromChannels = architecture.NodeChannels.TryGetValue(fromNode, out var fromNodeChannels)
                    ? fromNodeChannels
                    : inputChannels;

                var toChannels = architecture.NodeChannels.TryGetValue(toNode, out var toNodeChannels)
                    ? toNodeChannels
                    : fromChannels;

                var opCost = EstimateOperationCost(operation, fromChannels, toChannels, spatialSize);
                totalCost.Latency = _ops.Add(totalCost.Latency, opCost.Latency);
                totalCost.Energy = _ops.Add(totalCost.Energy, opCost.Energy);
                totalCost.Memory = _ops.Add(totalCost.Memory, opCost.Memory);
            }

            return totalCost;
        }

        /// <summary>
        /// Gets a breakdown of costs per operation in the architecture.
        /// </summary>
        public Dictionary<string, HardwareCost<T>> GetCostBreakdown(Architecture<T> architecture, int inputChannels, int spatialSize)
        {
            var breakdown = new Dictionary<string, HardwareCost<T>>();

            if (architecture == null)
                return breakdown;

            foreach (var (toNode, fromNode, operation) in architecture.Operations)
            {
                var fromChannels = architecture.NodeChannels.TryGetValue(fromNode, out var fromNodeChannels)
                    ? fromNodeChannels
                    : inputChannels;

                var toChannels = architecture.NodeChannels.TryGetValue(toNode, out var toNodeChannels)
                    ? toNodeChannels
                    : fromChannels;

                var opCost = EstimateOperationCost(operation, fromChannels, toChannels, spatialSize);
                var key = $"{operation}_{fromNode}_{toNode}";

                breakdown[key] = opCost;
            }

            return breakdown;
        }

        /// <summary>
        /// Calculates total FLOPs for an architecture.
        /// </summary>
        public double GetTotalFlops(Architecture<T> architecture, int inputChannels, int spatialSize)
        {
            if (architecture == null)
                return 0;

            double totalFlops = 0;

            foreach (var (toNode, fromNode, operation) in architecture.Operations)
            {
                var fromChannels = architecture.NodeChannels.TryGetValue(fromNode, out var fromNodeChannels)
                    ? fromNodeChannels
                    : inputChannels;

                var toChannels = architecture.NodeChannels.TryGetValue(toNode, out var toNodeChannels)
                    ? toNodeChannels
                    : fromChannels;

                totalFlops += CalculateOperationFlops(operation, fromChannels, toChannels, spatialSize);
            }

            return totalFlops;
        }

        /// <summary>
        /// Calculates total parameters (weights) for an architecture.
        /// </summary>
        public double GetTotalParameters(Architecture<T> architecture, int inputChannels)
        {
            if (architecture == null)
                return 0;

            double totalParams = 0;

            foreach (var (toNode, fromNode, operation) in architecture.Operations)
            {
                var fromChannels = architecture.NodeChannels.TryGetValue(fromNode, out var fromNodeChannels)
                    ? fromNodeChannels
                    : inputChannels;

                var toChannels = architecture.NodeChannels.TryGetValue(toNode, out var toNodeChannels)
                    ? toNodeChannels
                    : fromChannels;

                // Convert bytes to parameter count (divide by 4 for float32)
                totalParams += CalculateOperationMemory(operation, fromChannels, toChannels, 1) / 4.0;
            }

            return totalParams;
        }

        /// <summary>
        /// Checks if an architecture meets the hardware constraints.
        /// </summary>
        public bool MeetsConstraints(Architecture<T> architecture, HardwareConstraints<T> constraints,
            int inputChannels, int spatialSize)
        {
            var cost = EstimateArchitectureCost(architecture, inputChannels, spatialSize);

            if (constraints.MaxLatency.HasValue && _ops.ToDouble(cost.Latency) > constraints.MaxLatency.Value)
                return false;

            if (constraints.MaxEnergy.HasValue && _ops.ToDouble(cost.Energy) > constraints.MaxEnergy.Value)
                return false;

            if (constraints.MaxMemory.HasValue && _ops.ToDouble(cost.Memory) > constraints.MaxMemory.Value)
                return false;

            return true;
        }

        private static PlatformCharacteristics GetPlatformCharacteristics(HardwarePlatform platform)
        {
            return platform switch
            {
                HardwarePlatform.Mobile => new PlatformCharacteristics
                {
                    GFlops = 0.5,              // ~500 MFLOPS typical mobile CPU
                    MemoryBandwidthGBps = 10,  // LPDDR4
                    EnergyPerGFlop = 2.0,      // Higher energy per op
                    EnergyPerGBMemory = 10.0
                },
                HardwarePlatform.GPU => new PlatformCharacteristics
                {
                    GFlops = 10.0,             // Entry-level GPU
                    MemoryBandwidthGBps = 200, // GDDR6
                    EnergyPerGFlop = 0.1,      // More efficient
                    EnergyPerGBMemory = 1.0
                },
                HardwarePlatform.EdgeTPU => new PlatformCharacteristics
                {
                    GFlops = 4.0,              // Edge TPU optimized for inference
                    MemoryBandwidthGBps = 8,   // Limited memory
                    EnergyPerGFlop = 0.5,      // Very efficient for int8
                    EnergyPerGBMemory = 2.0
                },
                HardwarePlatform.CPU => new PlatformCharacteristics
                {
                    GFlops = 0.2,              // Single-thread CPU
                    MemoryBandwidthGBps = 25,  // DDR4
                    EnergyPerGFlop = 5.0,      // Less efficient
                    EnergyPerGBMemory = 5.0
                },
                _ => new PlatformCharacteristics
                {
                    GFlops = 1.0,
                    MemoryBandwidthGBps = 20,
                    EnergyPerGFlop = 1.0,
                    EnergyPerGBMemory = 5.0
                }
            };
        }
    }
}
