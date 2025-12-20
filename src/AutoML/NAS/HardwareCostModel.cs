using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Models hardware costs for neural architecture search operations.
    /// Supports latency, energy, and memory cost estimation for different hardware platforms.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class HardwareCostModel<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly HardwarePlatform _platform;
        private readonly Dictionary<string, HardwareCost<T>> _operationCosts;
        private readonly HashSet<string> _unknownOperations;

        public HardwareCostModel(HardwarePlatform platform = HardwarePlatform.Mobile)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _platform = platform;
            _operationCosts = new Dictionary<string, HardwareCost<T>>();
            _unknownOperations = new HashSet<string>(StringComparer.Ordinal);
            InitializeOperationCosts();
        }

        /// <summary>
        /// Estimates the hardware cost for a given operation
        /// </summary>
        public HardwareCost<T> EstimateOperationCost(string operation, int inputChannels, int outputChannels, int spatialSize)
        {
            if (_operationCosts.TryGetValue(operation, out var baseCost))
            {
                // Scale cost based on actual parameters
                var scaleFactor = _ops.FromDouble(((double)inputChannels * outputChannels * spatialSize * spatialSize) / 1000.0);

                return new HardwareCost<T>
                {
                    Latency = _ops.Multiply(baseCost.Latency, scaleFactor),
                    Energy = _ops.Multiply(baseCost.Energy, scaleFactor),
                    Memory = _ops.Multiply(baseCost.Memory, scaleFactor)
                };
            }

            // Default cost for unknown operations
            if (_unknownOperations.Add(operation))
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"AiDotNet.AutoML.NAS: Unknown operation '{operation}' encountered in hardware cost estimation. Using default unit cost.");
            }

            return new HardwareCost<T>
            {
                Latency = _ops.FromDouble(1.0),
                Energy = _ops.FromDouble(1.0),
                Memory = _ops.FromDouble(1.0)
            };
        }

        /// <summary>
        /// Estimates the total cost for an entire architecture
        /// </summary>
        public HardwareCost<T> EstimateArchitectureCost(Architecture<T> architecture, int inputChannels, int spatialSize)
        {
            var totalCost = new HardwareCost<T>
            {
                Latency = _ops.Zero,
                Energy = _ops.Zero,
                Memory = _ops.Zero
            };

            foreach (var (toNode, fromNode, operation) in architecture.Operations)
            {
                var fromChannels = architecture.NodeChannels.TryGetValue(fromNode, out var fromNodeChannels)
                    ? fromNodeChannels
                    : inputChannels;

                var toChannels = architecture.NodeChannels.TryGetValue(toNode, out var toNodeChannels)
                    ? toNodeChannels
                    : fromChannels;

                // If the architecture does not provide per-node channel counts, this falls back to a uniform channel
                // assumption (fromChannels == toChannels == inputChannels), which is a safe default for early NAS runs.
                var opCost = EstimateOperationCost(operation, fromChannels, toChannels, spatialSize);
                totalCost.Latency = _ops.Add(totalCost.Latency, opCost.Latency);
                totalCost.Energy = _ops.Add(totalCost.Energy, opCost.Energy);
                totalCost.Memory = _ops.Add(totalCost.Memory, opCost.Memory);
            }

            return totalCost;
        }

        /// <summary>
        /// Checks if an architecture meets the hardware constraints.
        /// </summary>
        /// <param name="architecture">The architecture to evaluate</param>
        /// <param name="constraints">The hardware constraints to check against</param>
        /// <param name="inputChannels">Number of input channels</param>
        /// <param name="spatialSize">Spatial size of the input (height/width)</param>
        /// <returns>True if the architecture meets all constraints, false otherwise</returns>
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

        private void InitializeOperationCosts()
        {
            // Platform-specific operation costs (latency in ms, energy in mJ, memory in MB)
            var platformMultiplier = _platform switch
            {
                HardwarePlatform.Mobile => 1.0,
                HardwarePlatform.GPU => 0.3,
                HardwarePlatform.EdgeTPU => 0.2,
                HardwarePlatform.CPU => 2.0,
                _ => 1.0
            };

            // Base costs (normalized per 1000 operations)
            _operationCosts["identity"] = new HardwareCost<T>
            {
                Latency = _ops.FromDouble(0.01 * platformMultiplier),
                Energy = _ops.FromDouble(0.001 * platformMultiplier),
                Memory = _ops.FromDouble(0.0)
            };

            _operationCosts["conv3x3"] = new HardwareCost<T>
            {
                Latency = _ops.FromDouble(1.0 * platformMultiplier),
                Energy = _ops.FromDouble(0.5 * platformMultiplier),
                Memory = _ops.FromDouble(0.1 * platformMultiplier)
            };

            _operationCosts["conv5x5"] = new HardwareCost<T>
            {
                Latency = _ops.FromDouble(2.5 * platformMultiplier),
                Energy = _ops.FromDouble(1.2 * platformMultiplier),
                Memory = _ops.FromDouble(0.25 * platformMultiplier)
            };

            _operationCosts["conv1x1"] = new HardwareCost<T>
            {
                Latency = _ops.FromDouble(0.3 * platformMultiplier),
                Energy = _ops.FromDouble(0.15 * platformMultiplier),
                Memory = _ops.FromDouble(0.05 * platformMultiplier)
            };

            _operationCosts["depthwise_conv3x3"] = new HardwareCost<T>
            {
                Latency = _ops.FromDouble(0.4 * platformMultiplier),
                Energy = _ops.FromDouble(0.2 * platformMultiplier),
                Memory = _ops.FromDouble(0.05 * platformMultiplier)
            };

            _operationCosts["maxpool3x3"] = new HardwareCost<T>
            {
                Latency = _ops.FromDouble(0.2 * platformMultiplier),
                Energy = _ops.FromDouble(0.05 * platformMultiplier),
                Memory = _ops.FromDouble(0.02 * platformMultiplier)
            };

            _operationCosts["avgpool3x3"] = new HardwareCost<T>
            {
                Latency = _ops.FromDouble(0.2 * platformMultiplier),
                Energy = _ops.FromDouble(0.05 * platformMultiplier),
                Memory = _ops.FromDouble(0.02 * platformMultiplier)
            };

            _operationCosts["se_block"] = new HardwareCost<T>
            {
                Latency = _ops.FromDouble(0.6 * platformMultiplier),
                Energy = _ops.FromDouble(0.3 * platformMultiplier),
                Memory = _ops.FromDouble(0.08 * platformMultiplier)
            };
        }
    }
}
