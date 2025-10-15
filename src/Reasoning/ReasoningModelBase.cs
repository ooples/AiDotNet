using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Statistics;
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interpretability;

namespace AiDotNet.Reasoning
{
    /// <summary>
    /// Provides a base implementation for reasoning models that perform multi-step logical reasoning,
    /// chain-of-thought processing, and explanation generation.
    /// </summary>
    /// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
    /// <remarks>
    /// <para>
    /// This abstract class implements common functionality for reasoning models, including
    /// multi-step reasoning, self-consistency checking, and iterative refinement. Specific reasoning
    /// algorithms should inherit from this class and implement the core reasoning methods.
    /// </para>
    /// <para>
    /// The class supports various reasoning strategies like forward chaining, backward chaining,
    /// and beam search, allowing flexible approaches to problem-solving.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Reasoning models are AI systems that can "think through" problems step by step, similar to
    /// how humans approach complex tasks. This base class provides the foundation for different
    /// reasoning techniques, handling common operations like generating reasoning chains, validating
    /// logic, and refining answers. Think of it as a template that specific reasoning algorithms
    /// can customize while reusing the shared functionality.
    /// </para>
    /// </remarks>
    public abstract class ReasoningModelBase<T> : IReasoningModel<T>
    {
        /// <summary>
        /// Gets the numeric operations for the specified type T.
        /// </summary>
        protected INumericOperations<T> NumOps { get; private set; }

        /// <summary>
        /// Gets the reasoning options.
        /// </summary>
        protected ReasoningModelOptions<T> Options { get; private set; }

        /// <summary>
        /// Gets or sets the current reasoning strategy.
        /// </summary>
        public ReasoningStrategy CurrentStrategy { get; protected set; }

        /// <summary>
        /// Gets the maximum reasoning depth the model can handle effectively.
        /// </summary>
        public abstract int MaxReasoningDepth { get; }

        /// <summary>
        /// Gets whether the model supports iterative refinement of its reasoning.
        /// </summary>
        public abstract bool SupportsIterativeRefinement { get; }

        /// <summary>
        /// Stores the reasoning steps from the last prediction.
        /// </summary>
        protected List<Tensor<T>> LastReasoningSteps { get; set; }

        /// <summary>
        /// Stores the confidence scores from the last reasoning process.
        /// </summary>
        protected Vector<T> LastConfidenceScores { get; set; }

        /// <summary>
        /// Stores diagnostic information from the last reasoning process.
        /// </summary>
        protected Dictionary<string, object> LastDiagnostics { get; set; }

        /// <summary>
        /// Set of feature indices that have been explicitly marked as active.
        /// </summary>
        private HashSet<int>? _explicitlySetActiveFeatures;

        /// <summary>
        /// Random number generator for stochastic reasoning strategies.
        /// </summary>
        protected Random Random { get; private set; }

        /// <summary>
        /// Initializes a new instance of the ReasoningModelBase class with the specified options.
        /// </summary>
        /// <param name="options">Configuration options for the reasoning model.</param>
        protected ReasoningModelBase(ReasoningModelOptions<T> options)
        {
            Options = options ?? throw new ArgumentNullException(nameof(options));
            NumOps = MathHelper.GetNumericOperations<T>();
            CurrentStrategy = options.DefaultStrategy;
            LastReasoningSteps = new List<Tensor<T>>();
            LastConfidenceScores = new Vector<T>(0);
            LastDiagnostics = new Dictionary<string, object>();
            Random = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();
        }

        /// <summary>
        /// Trains the reasoning model on the provided data.
        /// </summary>
        public abstract void Train(Tensor<T> input, Tensor<T> expectedOutput);

        /// <summary>
        /// Makes a prediction using the reasoning model.
        /// </summary>
        public virtual Tensor<T> Predict(Tensor<T> input)
        {
            // Clear previous reasoning data
            LastReasoningSteps.Clear();
            LastDiagnostics.Clear();

            var startTime = DateTime.UtcNow;

            // Perform multi-step reasoning
            var reasoningSteps = ReasonStepByStep(input, Options.DefaultMaxSteps);
            LastReasoningSteps.AddRange(reasoningSteps);

            // Get the final prediction from the reasoning chain
            var prediction = reasoningSteps.LastOrDefault() ?? throw new InvalidOperationException("No reasoning steps generated");

            // Calculate confidence scores
            LastConfidenceScores = CalculateConfidenceScores(reasoningSteps);

            // Store diagnostics
            LastDiagnostics["ReasoningTime"] = (DateTime.UtcNow - startTime).TotalMilliseconds;
            LastDiagnostics["StepCount"] = reasoningSteps.Count;
            LastDiagnostics["Strategy"] = CurrentStrategy.ToString();

            return prediction;
        }

        /// <summary>
        /// Performs multi-step reasoning on the input.
        /// </summary>
        public abstract List<Tensor<T>> ReasonStepByStep(Tensor<T> input, int maxSteps = 10);

        /// <summary>
        /// Generates an explanation for the model's prediction.
        /// </summary>
        public abstract Tensor<T> GenerateExplanation(Tensor<T> input, Tensor<T> prediction);

        /// <summary>
        /// Gets the confidence scores for each reasoning step.
        /// </summary>
        public Vector<T> GetReasoningConfidence()
        {
            return LastConfidenceScores;
        }

        /// <summary>
        /// Performs self-consistency checking by generating multiple reasoning paths.
        /// </summary>
        public virtual Tensor<T> SelfConsistencyCheck(Tensor<T> input, int numPaths = 3)
        {
            var results = new List<Tensor<T>>();
            var originalStrategy = CurrentStrategy;

            try
            {
                for (int i = 0; i < numPaths; i++)
                {
                    // Optionally vary the strategy for diversity
                    if (Options.VaryStrategyInSelfConsistency && i > 0)
                    {
                        CurrentStrategy = GetAlternativeStrategy(originalStrategy);
                    }

                    var result = Predict(input);
                    results.Add(result);
                }

                // Aggregate results (can be overridden for specific aggregation methods)
                return AggregateResults(results);
            }
            finally
            {
                CurrentStrategy = originalStrategy;
            }
        }

        /// <summary>
        /// Refines the reasoning process by iteratively improving the solution.
        /// </summary>
        public virtual Tensor<T> RefineReasoning(Tensor<T> input, Tensor<T> initialReasoning, int iterations = 3)
        {
            if (!SupportsIterativeRefinement)
            {
                throw new NotSupportedException("This model does not support iterative refinement");
            }

            var currentReasoning = initialReasoning;

            for (int i = 0; i < iterations; i++)
            {
                currentReasoning = PerformRefinementStep(input, currentReasoning, i);
            }

            return currentReasoning;
        }

        /// <summary>
        /// Sets the reasoning strategy for the model.
        /// </summary>
        public void SetReasoningStrategy(ReasoningStrategy strategy)
        {
            CurrentStrategy = strategy;
        }

        /// <summary>
        /// Validates the logical consistency of a reasoning chain.
        /// </summary>
        public abstract bool ValidateReasoningChain(List<Tensor<T>> reasoningSteps);

        /// <summary>
        /// Gets diagnostic information about the last reasoning process.
        /// </summary>
        public Dictionary<string, object> GetReasoningDiagnostics()
        {
            return new Dictionary<string, object>(LastDiagnostics);
        }

        /// <summary>
        /// Calculates confidence scores for each reasoning step.
        /// </summary>
        protected abstract Vector<T> CalculateConfidenceScores(List<Tensor<T>> reasoningSteps);

        /// <summary>
        /// Performs a single refinement step.
        /// </summary>
        protected abstract Tensor<T> PerformRefinementStep(Tensor<T> input, Tensor<T> currentReasoning, int iteration);

        /// <summary>
        /// Aggregates results from multiple reasoning paths.
        /// </summary>
        protected virtual Tensor<T> AggregateResults(List<Tensor<T>> results)
        {
            // Default implementation: average the results
            if (results.Count == 0)
            {
                throw new ArgumentException("No results to aggregate");
            }

            var shape = results[0].Shape;
            var aggregated = new Tensor<T>(shape);

            foreach (var result in results)
            {
                aggregated = aggregated.Add(result);
            }

            var count = NumOps.FromDouble(results.Count);
            var averaged = new Tensor<T>(aggregated.Shape);
            for (int i = 0; i < aggregated.Length; i++)
            {
                averaged[i] = NumOps.Divide(aggregated[i], count);
            }
            return averaged;
        }

        /// <summary>
        /// Gets an alternative reasoning strategy for diversity.
        /// </summary>
        protected virtual ReasoningStrategy GetAlternativeStrategy(ReasoningStrategy current)
        {
            var strategies = Enum.GetValues(typeof(ReasoningStrategy)).Cast<ReasoningStrategy>().Where(s => s != current).ToArray();
            return strategies[Random.Next(strategies.Length)];
        }

        #region IFullModel Implementation

        /// <summary>
        /// Gets model metadata.
        /// </summary>
        public ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = GetModelType(),
                FeatureCount = GetActiveFeatureIndices().Count(),
                Complexity = (int)EstimateComplexity(),
                Description = GetModelDescription(),
                AdditionalInfo = GetAdditionalMetadata()
            };
        }

        /// <summary>
        /// Gets the specific model type.
        /// </summary>
        protected abstract ModelType GetModelType();

        /// <summary>
        /// Gets a description of the model.
        /// </summary>
        protected abstract string GetModelDescription();

        /// <summary>
        /// Estimates the complexity of the model.
        /// </summary>
        protected abstract double EstimateComplexity();

        /// <summary>
        /// Gets additional metadata for the model.
        /// </summary>
        protected virtual Dictionary<string, object> GetAdditionalMetadata()
        {
            return new Dictionary<string, object>
            {
                ["MaxReasoningDepth"] = MaxReasoningDepth,
                ["SupportsIterativeRefinement"] = SupportsIterativeRefinement,
                ["CurrentStrategy"] = CurrentStrategy.ToString()
            };
        }

        #endregion

        #region IParameterizable Implementation

        /// <summary>
        /// Storage for model parameters as tensors.
        /// </summary>
        protected List<Tensor<T>> Parameters { get; set; } = new List<Tensor<T>>();

        /// <summary>
        /// Gets the model parameters as a flattened vector.
        /// </summary>
        public virtual Vector<T> GetParameters()
        {
            // Flatten all tensor parameters into a single vector
            var allParams = new List<T>();
            foreach (var tensor in Parameters)
            {
                for (int i = 0; i < tensor.Length; i++)
                {
                    allParams.Add(tensor[i]);
                }
            }
            return new Vector<T>(allParams.ToArray());
        }

        /// <summary>
        /// Sets the model parameters from a flattened vector.
        /// </summary>
        public virtual void SetParameters(Vector<T> parameters)
        {
            // Unflatten the vector back into tensors
            int offset = 0;
            var paramList = Parameters.ToList();
            
            for (int i = 0; i < paramList.Count; i++)
            {
                var tensor = paramList[i];
                var size = tensor.Length;
                
                if (offset + size > parameters.Length)
                    throw new ArgumentException("Parameter vector size mismatch");
                
                var newData = new T[size];
                for (int j = 0; j < size; j++)
                {
                    newData[j] = parameters[offset + j];
                }
                
                paramList[i] = new Tensor<T>(tensor.Shape);
                for (int k = 0; k < newData.Length; k++)
                {
                    paramList[i][k] = newData[k];
                }
                offset += size;
            }
            
            Parameters = paramList;
        }

        /// <summary>
        /// Creates a new model with the specified parameters.
        /// </summary>
        public abstract IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters);

        #endregion

        #region IFeatureAware Implementation

        /// <summary>
        /// Gets the indices of active features.
        /// </summary>
        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return _explicitlySetActiveFeatures ?? Enumerable.Empty<int>();
        }

        /// <summary>
        /// Checks if a feature is used by the model.
        /// </summary>
        public bool IsFeatureUsed(int featureIndex)
        {
            if (_explicitlySetActiveFeatures != null)
            {
                return _explicitlySetActiveFeatures.Contains(featureIndex);
            }
            
            // Default implementation: all features are considered active
            return true;
        }

        /// <summary>
        /// Sets the active feature indices.
        /// </summary>
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        {
            _explicitlySetActiveFeatures = new HashSet<int>(featureIndices);
        }

        #endregion

        #region IModelSerializer Implementation

        /// <summary>
        /// Serializes the model to a byte array.
        /// </summary>
        public abstract byte[] Serialize();

        /// <summary>
        /// Deserializes the model from a byte array.
        /// </summary>
        public abstract void Deserialize(byte[] data);

        #endregion

        #region ICloneable Implementation

        /// <summary>
        /// Creates a deep copy of the model.
        /// </summary>
        public abstract IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy();

        /// <summary>
        /// Creates a clone of the model.
        /// </summary>
        public virtual IFullModel<T, Tensor<T>, Tensor<T>> Clone()
        {
            return DeepCopy();
        }

        #endregion

        #region IInterpretableModel Implementation

        protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
        protected Vector<int> _sensitiveFeatures;
        protected readonly List<FairnessMetric> _fairnessMetrics = new();
        protected IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> _baseModel;

        /// <summary>
        /// Gets the global feature importance across all predictions.
        /// </summary>
        public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
        {
        return await InterpretableModelHelper.GetGlobalFeatureImportanceAsync(this, _enabledMethods);
        }

        /// <summary>
        /// Gets the local feature importance for a specific input.
        /// </summary>
        public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Tensor<T> input)
        {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
        }

        /// <summary>
        /// Gets SHAP values for the given inputs.
        /// </summary>
        public virtual async Task<Matrix<T>> GetShapValuesAsync(Tensor<T> inputs)
        {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods);
        }

        /// <summary>
        /// Gets LIME explanation for a specific input.
        /// </summary>
        public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Tensor<T> input, int numFeatures = 10)
        {
        return await InterpretableModelHelper.GetLimeExplanationAsync<T>(_enabledMethods, numFeatures);
        }

        /// <summary>
        /// Gets partial dependence data for specified features.
        /// </summary>
        public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
        {
        return await InterpretableModelHelper.GetPartialDependenceAsync<T>(_enabledMethods, featureIndices, gridResolution);
        }

        /// <summary>
        /// Gets counterfactual explanation for a given input and desired output.
        /// </summary>
        public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Tensor<T> input, Tensor<T> desiredOutput, int maxChanges = 5)
        {
        return await InterpretableModelHelper.GetCounterfactualAsync<T>(_enabledMethods, maxChanges);
        }

        /// <summary>
        /// Gets model-specific interpretability information.
        /// </summary>
        public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
        {
        return await InterpretableModelHelper.GetModelSpecificInterpretabilityAsync(this);
        }

        /// <summary>
        /// Generates a text explanation for a prediction.
        /// </summary>
        public virtual async Task<string> GenerateTextExplanationAsync(Tensor<T> input, Tensor<T> prediction)
        {
        return await InterpretableModelHelper.GenerateTextExplanationAsync(this, input, prediction);
        }

        /// <summary>
        /// Gets feature interaction effects between two features.
        /// </summary>
        public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
        {
        return await InterpretableModelHelper.GetFeatureInteractionAsync<T>(_enabledMethods, feature1Index, feature2Index);
        }

        /// <summary>
        /// Validates fairness metrics for the given inputs.
        /// </summary>
        public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Tensor<T> inputs, int sensitiveFeatureIndex)
        {
        return await InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
        }

        /// <summary>
        /// Gets anchor explanation for a given input.
        /// </summary>
        public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Tensor<T> input, T threshold)
        {
        return await InterpretableModelHelper.GetAnchorExplanationAsync(_enabledMethods, threshold);
        }

        /// <summary>
        /// Sets the base model for interpretability analysis.
        /// </summary>
        public virtual void SetBaseModel(IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> model)
        {
        _baseModel = model ?? throw new ArgumentNullException(nameof(model));
        }

        /// <summary>
        /// Enables specific interpretation methods.
        /// </summary>
        public virtual void EnableMethod(params InterpretationMethod[] methods)
        {
        foreach (var method in methods)
        {
            _enabledMethods.Add(method);
        }
        }

        /// <summary>
        /// Configures fairness evaluation settings.
        /// </summary>
        public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
        {
        _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
        _fairnessMetrics.Clear();
        _fairnessMetrics.AddRange(fairnessMetrics);
        }

        #endregion
    }
}