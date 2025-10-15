using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Base class providing default implementations for interpretability features
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <typeparam name="TInput">The input data type</typeparam>
    /// <typeparam name="TOutput">The output data type</typeparam>
    public abstract class InterpretableModelBase<T, TInput, TOutput> : IInterpretableModel<T, TInput, TOutput>
    {
        protected readonly INumericOperations<T> _ops;
        protected IModel<TInput, TOutput, ModelMetadata<T>> _baseModel;
        protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
        protected Vector<int> _sensitiveFeatures;
        protected readonly List<FairnessMetric> _fairnessMetrics = new();

        protected InterpretableModelBase()
        {
            _ops = MathHelper.GetNumericOperations<T>();
        }

        // Abstract methods that derived classes must implement
        public abstract TOutput Predict(TInput input);
        public abstract Task<TOutput> PredictAsync(TInput input);
        public abstract void Train(TInput inputs, TOutput targets);
        public abstract Task TrainAsync(TInput inputs, TOutput targets);
        public abstract ModelMetadata<T> GetModelMetadata();
        public abstract void SetModelMetadata(ModelMetadata<T> metadata);
        public abstract void Save(string filePath);
        public abstract void Load(string filePath);
        public abstract void Dispose();

        // Virtual interpretability methods with default implementations
        public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
        {
            if (!_enabledMethods.Contains(InterpretationMethod.FeatureImportance))
            {
                throw new InvalidOperationException("Feature importance is not enabled. Call EnableMethod first.");
            }

            // Default implementation using permutation importance
            var importance = new Dictionary<int, T>();
            var metadata = GetModelMetadata();
            
            for (int i = 0; i < metadata.FeatureCount; i++)
            {
                importance[i] = _ops.One; // Placeholder - derived classes should override
            }
            
            return await Task.FromResult(importance);
        }

        public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(TInput input)
        {
            if (!_enabledMethods.Contains(InterpretationMethod.FeatureImportance))
            {
                throw new InvalidOperationException("Feature importance is not enabled. Call EnableMethod first.");
            }

            // Default implementation - derived classes should override for better results
            return await GetGlobalFeatureImportanceAsync();
        }

        public virtual async Task<Matrix<T>> GetShapValuesAsync(TInput inputs)
        {
            if (!_enabledMethods.Contains(InterpretationMethod.SHAP))
            {
                throw new InvalidOperationException("SHAP is not enabled. Call EnableMethod first.");
            }

            // Placeholder implementation - derived classes should override
            var metadata = GetModelMetadata();
            var shapValues = new Matrix<T>(1, metadata.FeatureCount);
            
            for (int i = 0; i < metadata.FeatureCount; i++)
            {
                shapValues[0, i] = _ops.Zero;
            }
            
            return await Task.FromResult(shapValues);
        }

        public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(TInput input, int numFeatures = 10)
        {
            if (!_enabledMethods.Contains(InterpretationMethod.LIME))
            {
                throw new InvalidOperationException("LIME is not enabled. Call EnableMethod first.");
            }

            // Placeholder implementation - derived classes should override
            return await Task.FromResult(new LimeExplanation<T>());
        }

        public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
        {
            if (!_enabledMethods.Contains(InterpretationMethod.PartialDependence))
            {
                throw new InvalidOperationException("Partial dependence is not enabled. Call EnableMethod first.");
            }

            // Placeholder implementation - derived classes should override
            return await Task.FromResult(new PartialDependenceData<T>());
        }

        public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(TInput input, TOutput desiredOutput, int maxChanges = 5)
        {
            if (!_enabledMethods.Contains(InterpretationMethod.Counterfactual))
            {
                throw new InvalidOperationException("Counterfactual explanations are not enabled. Call EnableMethod first.");
            }

            // Placeholder implementation - derived classes should override
            return await Task.FromResult(new CounterfactualExplanation<T>());
        }

        public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
        {
            // Default implementation returns basic model info
            var result = new Dictionary<string, object>();
            var metadata = GetModelMetadata();
            
            result["ModelType"] = metadata.ModelType.ToString();
            result["FeatureCount"] = metadata.FeatureCount;
            result["Complexity"] = metadata.Complexity;
            
            return await Task.FromResult(result);
        }

        public virtual async Task<string> GenerateTextExplanationAsync(TInput input, TOutput prediction)
        {
            // Default implementation
            var metadata = GetModelMetadata();
            return await Task.FromResult($"Model {metadata.ModelType} predicted output based on {metadata.FeatureCount} features.");
        }

        public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
        {
            if (!_enabledMethods.Contains(InterpretationMethod.FeatureInteraction))
            {
                throw new InvalidOperationException("Feature interaction analysis is not enabled. Call EnableMethod first.");
            }

            // Placeholder implementation - derived classes should override
            return await Task.FromResult(_ops.Zero);
        }

        public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(TInput inputs, int sensitiveFeatureIndex)
        {
            if (_fairnessMetrics.Count == 0)
            {
                throw new InvalidOperationException("No fairness metrics configured. Call ConfigureFairness first.");
            }

            // Placeholder implementation - derived classes should override
            return await Task.FromResult(new FairnessMetrics<T>());
        }

        public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(TInput input, T threshold)
        {
            if (!_enabledMethods.Contains(InterpretationMethod.Anchors))
            {
                throw new InvalidOperationException("Anchor explanations are not enabled. Call EnableMethod first.");
            }

            // Placeholder implementation - derived classes should override
            return await Task.FromResult(new AnchorExplanation<T>());
        }

        public virtual void SetBaseModel(IModel<TInput, TOutput, ModelMetadata<T>> model)
        {
            _baseModel = model ?? throw new ArgumentNullException(nameof(model));
        }

        public virtual void EnableMethod(params InterpretationMethod[] methods)
        {
            foreach (var method in methods)
            {
                _enabledMethods.Add(method);
            }
        }

        public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
        {
            _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
            _fairnessMetrics.Clear();
            _fairnessMetrics.AddRange(fairnessMetrics);
        }
    }
}