using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for foundation/large language models.
    /// Extends IFullModel to provide complete model functionality with foundation model-specific features.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public interface IFoundationModel<T> : IFullModel<T, string, string>
    {
        /// <summary>
        /// Gets the model architecture type
        /// </summary>
        string Architecture { get; }

        /// <summary>
        /// Gets the number of parameters in the model
        /// </summary>
        long ParameterCount { get; }

        /// <summary>
        /// Gets the vocabulary size
        /// </summary>
        int VocabularySize { get; }

        /// <summary>
        /// Gets the maximum context length
        /// </summary>
        int MaxContextLength { get; }

        /// <summary>
        /// Generates text completion
        /// </summary>
        /// <param name="prompt">Input prompt</param>
        /// <param name="maxTokens">Maximum tokens to generate</param>
        /// <param name="temperature">Sampling temperature</param>
        /// <param name="topP">Top-p sampling parameter</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Generated text</returns>
        Task<string> GenerateAsync(
            string prompt,
            int maxTokens = 100,
            double temperature = 1.0,
            double topP = 1.0,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Generates embeddings for input text
        /// </summary>
        /// <param name="text">Input text</param>
        /// <returns>Embedding vector</returns>
        Task<T[]> GetEmbeddingAsync(string text);

        /// <summary>
        /// Tokenizes input text
        /// </summary>
        /// <param name="text">Text to tokenize</param>
        /// <returns>Token IDs</returns>
        Task<int[]> TokenizeAsync(string text);

        /// <summary>
        /// Decodes token IDs to text
        /// </summary>
        /// <param name="tokenIds">Token IDs</param>
        /// <returns>Decoded text</returns>
        Task<string> DecodeAsync(int[] tokenIds);

        /// <summary>
        /// Fine-tunes the model on specific data
        /// </summary>
        /// <param name="trainingData">Training examples</param>
        /// <param name="validationData">Validation examples</param>
        /// <param name="config">Fine-tuning configuration</param>
        /// <param name="progressCallback">Progress callback</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Fine-tuned model</returns>
        Task<IFoundationModel<T, TInput, TOutput>> FineTuneAsync(
            List<TrainingExample> trainingData,
            List<TrainingExample> validationData,
            FineTuningConfig config,
            Action<FineTuningProgress>? progressCallback = null,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Performs few-shot learning
        /// </summary>
        /// <param name="examples">Few-shot examples</param>
        /// <param name="query">Query to process</param>
        /// <returns>Model response</returns>
        Task<string> FewShotAsync(List<FewShotExample> examples, string query);

        /// <summary>
        /// Applies prompt engineering template
        /// </summary>
        /// <param name="template">Prompt template</param>
        /// <param name="variables">Template variables</param>
        /// <returns>Formatted prompt</returns>
        string ApplyPromptTemplate(string template, Dictionary<string, string> variables);

        /// <summary>
        /// Gets attention weights for input
        /// </summary>
        /// <param name="text">Input text</param>
        /// <returns>Attention weights</returns>
        Task<AttentionWeights> GetAttentionWeightsAsync(string text);

        /// <summary>
        /// Performs chain-of-thought reasoning
        /// </summary>
        /// <param name="problem">Problem statement</param>
        /// <returns>Reasoning steps and final answer</returns>
        Task<ChainOfThoughtResult> ChainOfThoughtAsync(string problem);

        /// <summary>
        /// Evaluates the model on a benchmark
        /// </summary>
        /// <param name="benchmark">Benchmark dataset</param>
        /// <returns>Evaluation results</returns>
        Task<BenchmarkResults> EvaluateBenchmarkAsync(IBenchmarkDataset benchmark);

        /// <summary>
        /// Applies a model adapter (LoRA, etc.)
        /// </summary>
        /// <param name="adapter">Adapter to apply</param>
        void ApplyAdapter(IModelAdapter<T, TInput, TOutput> adapter);

        /// <summary>
        /// Gets available model checkpoints
        /// </summary>
        /// <returns>List of checkpoint names</returns>
        List<string> GetAvailableCheckpoints();

        /// <summary>
        /// Loads a specific checkpoint
        /// </summary>
        /// <param name="checkpointName">Name of checkpoint to load</param>
        Task LoadCheckpointAsync(string checkpointName);
    }

    /// <summary>
    /// Training example for fine-tuning
    /// </summary>
    public class TrainingExample
    {
        public string Input { get; set; } = string.Empty;
        public string Output { get; set; } = string.Empty;
        public Dictionary<string, object>? Metadata { get; set; }
    }

    /// <summary>
    /// Fine-tuning configuration
    /// </summary>
    public class FineTuningConfig
    {
        public int Epochs { get; set; } = 3;
        public double LearningRate { get; set; } = 1e-5;
        public int BatchSize { get; set; } = 8;
        public double WeightDecay { get; set; } = 0.01;
        public int WarmupSteps { get; set; } = 500;
        public string OptimizerType { get; set; } = "AdamW";
        public bool UseMixedPrecision { get; set; } = true;
        public int GradientAccumulationSteps { get; set; } = 1;
    }

    /// <summary>
    /// Fine-tuning progress information
    /// </summary>
    public class FineTuningProgress
    {
        public int CurrentEpoch { get; set; }
        public int TotalEpochs { get; set; }
        public int CurrentStep { get; set; }
        public int TotalSteps { get; set; }
        public double TrainingLoss { get; set; }
        public double ValidationLoss { get; set; }
        public TimeSpan ElapsedTime { get; set; }
        public TimeSpan EstimatedTimeRemaining { get; set; }
    }

    /// <summary>
    /// Few-shot learning example
    /// </summary>
    public class FewShotExample
    {
        public string Input { get; set; } = string.Empty;
        public string Output { get; set; } = string.Empty;
        public string? Explanation { get; set; }
    }

    /// <summary>
    /// Attention weights information
    /// </summary>
    public class AttentionWeights
    {
        public List<List<double[,]>> LayerWeights { get; set; } = new();
        public string[] Tokens { get; set; } = new string[0];
        public int NumLayers { get; set; }
        public int NumHeads { get; set; }
    }

    /// <summary>
    /// Chain-of-thought reasoning result
    /// </summary>
    public class ChainOfThoughtResult
    {
        public List<string> ReasoningSteps { get; set; } = new();
        public string FinalAnswer { get; set; } = string.Empty;
        public double Confidence { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    /// <summary>
    /// Benchmark dataset interface
    /// </summary>
    public interface IBenchmarkDataset
    {
        string Name { get; }
        List<BenchmarkExample> GetExamples();
        double CalculateScore(List<BenchmarkPrediction> predictions);
    }

    /// <summary>
    /// Benchmark example
    /// </summary>
    public class BenchmarkExample
    {
        public string Id { get; set; } = string.Empty;
        public string Input { get; set; } = string.Empty;
        public string ExpectedOutput { get; set; } = string.Empty;
        public Dictionary<string, object>? Metadata { get; set; }
    }

    /// <summary>
    /// Benchmark prediction
    /// </summary>
    public class BenchmarkPrediction
    {
        public string ExampleId { get; set; } = string.Empty;
        public string Prediction { get; set; } = string.Empty;
        public double Confidence { get; set; }
    }

    /// <summary>
    /// Benchmark evaluation results
    /// </summary>
    public class BenchmarkResults
    {
        public string BenchmarkName { get; set; } = string.Empty;
        public double Score { get; set; }
        public Dictionary<string, double> Metrics { get; set; } = new();
        public TimeSpan EvaluationTime { get; set; }
        public int TotalExamples { get; set; }
    }

    /// <summary>
    /// Model adapter interface (for LoRA, etc.)
    /// </summary>
    public interface IModelAdapter
    {
        string AdapterType { get; }
        long AdapterParameters { get; }
        void Apply(IFoundationModel model);
    }
}