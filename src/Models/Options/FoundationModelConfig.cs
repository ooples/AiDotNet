using System;
using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Configuration options for foundation models.
    /// Controls model behavior, resource usage, and optimization settings.
    /// </summary>
    public class FoundationModelConfig
    {
        /// <summary>
        /// Model identifier or path
        /// </summary>
        public string ModelId { get; set; } = string.Empty;

        /// <summary>
        /// Device to run the model on (e.g., "cpu", "cuda", "cuda:0")
        /// </summary>
        public string Device { get; set; } = "cpu";

        /// <summary>
        /// Data type for model weights (e.g., "float32", "float16", "int8")
        /// </summary>
        public string DataType { get; set; } = "float32";

        /// <summary>
        /// Maximum batch size for inference
        /// </summary>
        public int MaxBatchSize { get; set; } = 1;

        /// <summary>
        /// Maximum sequence length to support
        /// </summary>
        public int? MaxSequenceLength { get; set; }

        /// <summary>
        /// Whether to use mixed precision for faster inference
        /// </summary>
        public bool UseMixedPrecision { get; set; } = false;

        /// <summary>
        /// Quantization settings for model compression
        /// </summary>
        public QuantizationConfig? Quantization { get; set; }

        /// <summary>
        /// Cache configuration for model outputs
        /// </summary>
        public CacheConfig? CacheConfig { get; set; }

        /// <summary>
        /// Memory allocation settings
        /// </summary>
        public MemoryConfig MemoryConfig { get; set; } = new();

        /// <summary>
        /// Optimization level for inference
        /// </summary>
        public OptimizationLevel OptimizationLevel { get; set; } = OptimizationLevel.O1;

        /// <summary>
        /// Whether to enable gradient checkpointing (for fine-tuning)
        /// </summary>
        public bool GradientCheckpointing { get; set; } = false;

        /// <summary>
        /// Number of threads for CPU inference
        /// </summary>
        public int? NumThreads { get; set; }

        /// <summary>
        /// Custom model configuration parameters
        /// </summary>
        public Dictionary<string, object> CustomConfig { get; set; } = new();

        /// <summary>
        /// Path to custom tokenizer (if different from model)
        /// </summary>
        public string? CustomTokenizerPath { get; set; }

        /// <summary>
        /// Whether to trust remote code (for custom models)
        /// </summary>
        public bool TrustRemoteCode { get; set; } = false;

        /// <summary>
        /// Model-specific generation configuration
        /// </summary>
        public GenerationConfig GenerationConfig { get; set; } = new();

        /// <summary>
        /// Creates a default configuration for CPU inference
        /// </summary>
        public static FoundationModelConfig CreateDefault(string modelId)
        {
            return new FoundationModelConfig
            {
                ModelId = modelId,
                Device = "cpu",
                DataType = "float32",
                MaxBatchSize = 1,
                OptimizationLevel = OptimizationLevel.O1
            };
        }

        /// <summary>
        /// Creates an optimized configuration for GPU inference
        /// </summary>
        public static FoundationModelConfig CreateGPUOptimized(string modelId, int deviceId = 0)
        {
            return new FoundationModelConfig
            {
                ModelId = modelId,
                Device = $"cuda:{deviceId}",
                DataType = "float16",
                MaxBatchSize = 8,
                UseMixedPrecision = true,
                OptimizationLevel = OptimizationLevel.O2,
                MemoryConfig = new MemoryConfig
                {
                    EnableMemoryOptimization = true,
                    MaxMemoryMB = 8192
                }
            };
        }

        /// <summary>
        /// Validates the configuration
        /// </summary>
        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(ModelId))
            {
                throw new InvalidOperationException("ModelId cannot be null or empty");
            }

            if (MaxBatchSize <= 0)
            {
                throw new InvalidOperationException("MaxBatchSize must be greater than 0");
            }

            if (MaxSequenceLength.HasValue && MaxSequenceLength.Value <= 0)
            {
                throw new InvalidOperationException("MaxSequenceLength must be greater than 0");
            }

            if (NumThreads.HasValue && NumThreads.Value <= 0)
            {
                throw new InvalidOperationException("NumThreads must be greater than 0");
            }

            Quantization?.Validate();
            CacheConfig?.Validate();
            MemoryConfig?.Validate();
            GenerationConfig?.Validate();
        }
    }

    /// <summary>
    /// Configuration for model quantization
    /// </summary>
    public class QuantizationConfig
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
        public List<string> ExcludeLayers { get; set; } = new();

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
