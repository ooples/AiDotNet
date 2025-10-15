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

    /// <summary>
    /// Configuration for output caching
    /// </summary>
    public class CacheConfig
    {
        /// <summary>
        /// Whether to enable output caching
        /// </summary>
        public bool EnableCache { get; set; } = true;

        /// <summary>
        /// Maximum number of cached entries
        /// </summary>
        public int MaxCacheEntries { get; set; } = 1000;

        /// <summary>
        /// Cache expiration time in seconds
        /// </summary>
        public int CacheExpirationSeconds { get; set; } = 3600;

        /// <summary>
        /// Whether to cache attention weights
        /// </summary>
        public bool CacheAttentionWeights { get; set; } = false;

        /// <summary>
        /// Validates the cache configuration
        /// </summary>
        public void Validate()
        {
            if (MaxCacheEntries <= 0)
            {
                throw new InvalidOperationException("MaxCacheEntries must be greater than 0");
            }

            if (CacheExpirationSeconds <= 0)
            {
                throw new InvalidOperationException("CacheExpirationSeconds must be greater than 0");
            }
        }
    }

    /// <summary>
    /// Memory allocation configuration
    /// </summary>
    public class MemoryConfig
    {
        /// <summary>
        /// Whether to enable memory optimization
        /// </summary>
        public bool EnableMemoryOptimization { get; set; } = true;

        /// <summary>
        /// Maximum memory usage in MB
        /// </summary>
        public int? MaxMemoryMB { get; set; }

        /// <summary>
        /// Whether to offload to CPU when GPU memory is full
        /// </summary>
        public bool EnableCPUOffload { get; set; } = false;

        /// <summary>
        /// Memory growth strategy for GPU
        /// </summary>
        public bool AllowMemoryGrowth { get; set; } = true;

        /// <summary>
        /// Pre-allocate memory percentage (0-100)
        /// </summary>
        public int PreAllocatePercent { get; set; } = 80;

        /// <summary>
        /// Validates the memory configuration
        /// </summary>
        public void Validate()
        {
            if (MaxMemoryMB.HasValue && MaxMemoryMB.Value <= 0)
            {
                throw new InvalidOperationException("MaxMemoryMB must be greater than 0");
            }

            if (PreAllocatePercent < 0 || PreAllocatePercent > 100)
            {
                throw new InvalidOperationException("PreAllocatePercent must be between 0 and 100");
            }
        }
    }

    /// <summary>
    /// Generation-specific configuration
    /// </summary>
    public class GenerationConfig
    {
        /// <summary>
        /// Default temperature for generation
        /// </summary>
        public double DefaultTemperature { get; set; } = 1.0;

        /// <summary>
        /// Default top-p value
        /// </summary>
        public double DefaultTopP { get; set; } = 1.0;

        /// <summary>
        /// Default top-k value
        /// </summary>
        public int? DefaultTopK { get; set; }

        /// <summary>
        /// Repetition penalty
        /// </summary>
        public double RepetitionPenalty { get; set; } = 1.0;

        /// <summary>
        /// Length penalty for beam search
        /// </summary>
        public double LengthPenalty { get; set; } = 1.0;

        /// <summary>
        /// Number of beams for beam search
        /// </summary>
        public int NumBeams { get; set; } = 1;

        /// <summary>
        /// Early stopping for beam search
        /// </summary>
        public bool EarlyStopping { get; set; } = false;

        /// <summary>
        /// Bad words to avoid in generation
        /// </summary>
        public List<string> BadWords { get; set; } = new();

        /// <summary>
        /// Forced decoder IDs
        /// </summary>
        public Dictionary<int, int> ForcedDecoderIds { get; set; } = new();

        /// <summary>
        /// Validates the generation configuration
        /// </summary>
        public void Validate()
        {
            if (DefaultTemperature <= 0)
            {
                throw new InvalidOperationException("DefaultTemperature must be greater than 0");
            }

            if (DefaultTopP <= 0 || DefaultTopP > 1)
            {
                throw new InvalidOperationException("DefaultTopP must be between 0 and 1");
            }

            if (DefaultTopK.HasValue && DefaultTopK.Value <= 0)
            {
                throw new InvalidOperationException("DefaultTopK must be greater than 0");
            }

            if (RepetitionPenalty < 0)
            {
                throw new InvalidOperationException("RepetitionPenalty must be non-negative");
            }

            if (NumBeams <= 0)
            {
                throw new InvalidOperationException("NumBeams must be greater than 0");
            }
        }
    }
}