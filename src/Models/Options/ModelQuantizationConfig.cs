using System;
using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Configuration for model quantization
    /// </summary>
    public class ModelQuantizationConfig
    {
        /// <summary>
        /// Quantization type (e.g., "int8", "int4", "dynamic")
        /// </summary>
        public QuantizationType Type { get; set; } = QuantizationType.None;

        /// <summary>
        /// Whether to quantize weights
        /// </summary>
        public bool QuantizeWeights { get; set; } = true;

        /// <summary>
        /// Whether to quantize activations
        /// </summary>
        public bool QuantizeActivations { get; set; } = false;

        /// <summary>
        /// Calibration dataset size for quantization
        /// </summary>
        public int CalibrationSamples { get; set; } = 100;

        /// <summary>
        /// Layers to exclude from quantization
        /// </summary>
        public List<string> ExcludeLayers { get; set; } = new List<string>();

        /// <summary>
        /// Validates the quantization configuration
        /// </summary>
        public void Validate()
        {
            if (CalibrationSamples <= 0)
            {
                throw new InvalidOperationException("CalibrationSamples must be greater than 0");
            }
        }
    }
}