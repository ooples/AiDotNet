using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Helper class providing default implementations for IInterpretableModel methods.
    /// </summary>
    /// <remarks>
    /// This class provides static methods that can be used by classes implementing IInterpretableModel
    /// to avoid code duplication while working within C#'s single inheritance limitation.
    /// </remarks>
    public static class InterpretableModelHelper
    {
        /// <summary>
        /// Default implementation for GetGlobalFeatureImportanceAsync.
        /// </summary>
        public static async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync<T>(
            IModel<object, object, ModelMetadata<T>> model,
            HashSet<InterpretationMethod> enabledMethods)
        {
            if (!enabledMethods.Contains(InterpretationMethod.FeatureImportance))
            {
                throw new InvalidOperationException("Feature importance is not enabled. Call EnableMethod first.");
            }

            var ops = MathHelper.GetNumericOperations<T>();
            var importance = new Dictionary<int, T>();
            var metadata = model.GetModelMetadata();
            
            for (int i = 0; i < metadata.FeatureCount; i++)
            {
                importance[i] = ops.One; // Placeholder - derived classes should override
            }
            
            return await Task.FromResult(importance);
        }

        /// <summary>
        /// Default implementation for GetLocalFeatureImportanceAsync.
        /// </summary>
        public static async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync<T, TInput>(
            IModel<TInput, object, ModelMetadata<T>> model,
            HashSet<InterpretationMethod> enabledMethods,
            TInput input)
        {
            if (!enabledMethods.Contains(InterpretationMethod.FeatureImportance))
            {
                throw new InvalidOperationException("Feature importance is not enabled. Call EnableMethod first.");
            }

            // Default implementation - derived classes should override for better results
            return await GetGlobalFeatureImportanceAsync(model, enabledMethods);
        }

        /// <summary>
        /// Default implementation for GetShapValuesAsync.
        /// </summary>
        public static async Task<Matrix<T>> GetShapValuesAsync<T>(
            IModel<object, object, ModelMetadata<T>> model,
            HashSet<InterpretationMethod> enabledMethods)
        {
            if (!enabledMethods.Contains(InterpretationMethod.SHAP))
            {
                throw new InvalidOperationException("SHAP is not enabled. Call EnableMethod first.");
            }

            var ops = MathHelper.GetNumericOperations<T>();
            var metadata = model.GetModelMetadata();
            var shapValues = new Matrix<T>(1, metadata.FeatureCount);
            
            for (int i = 0; i < metadata.FeatureCount; i++)
            {
                shapValues[0, i] = ops.Zero;
            }
            
            return await Task.FromResult(shapValues);
        }

        /// <summary>
        /// Default implementation for GetLimeExplanationAsync.
        /// </summary>
        public static async Task<LimeExplanation<T>> GetLimeExplanationAsync<T>(
            HashSet<InterpretationMethod> enabledMethods,
            int numFeatures = 10)
        {
            if (!enabledMethods.Contains(InterpretationMethod.LIME))
            {
                throw new InvalidOperationException("LIME is not enabled. Call EnableMethod first.");
            }

            // Placeholder implementation - derived classes should override
            return await Task.FromResult(new LimeExplanation<T>());
        }

        /// <summary>
        /// Default implementation for GetPartialDependenceAsync.
        /// </summary>
        public static async Task<PartialDependenceData<T>> GetPartialDependenceAsync<T>(
            HashSet<InterpretationMethod> enabledMethods,
            Vector<int> featureIndices,
            int gridResolution = 20)
        {
            if (!enabledMethods.Contains(InterpretationMethod.PartialDependence))
            {
                throw new InvalidOperationException("Partial dependence is not enabled. Call EnableMethod first.");
            }

            // Placeholder implementation - derived classes should override
            return await Task.FromResult(new PartialDependenceData<T>());
        }

        /// <summary>
        /// Default implementation for GetCounterfactualAsync.
        /// </summary>
        public static async Task<CounterfactualExplanation<T>> GetCounterfactualAsync<T>(
            HashSet<InterpretationMethod> enabledMethods,
            int maxChanges = 5)
        {
            if (!enabledMethods.Contains(InterpretationMethod.Counterfactual))
            {
                throw new InvalidOperationException("Counterfactual explanations are not enabled. Call EnableMethod first.");
            }

            // Placeholder implementation - derived classes should override
            return await Task.FromResult(new CounterfactualExplanation<T>());
        }

        /// <summary>
        /// Default implementation for GetModelSpecificInterpretabilityAsync.
        /// </summary>
        public static async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync<T>(
            IModel<object, object, ModelMetadata<T>> model)
        {
            var result = new Dictionary<string, object>();
            var metadata = model.GetModelMetadata();
            
            result["ModelType"] = metadata.ModelType.ToString();
            result["FeatureCount"] = metadata.FeatureCount;
            result["Complexity"] = metadata.Complexity;
            
            return await Task.FromResult(result);
        }

        /// <summary>
        /// Default implementation for GenerateTextExplanationAsync.
        /// </summary>
        public static async Task<string> GenerateTextExplanationAsync<T, TInput, TOutput>(
            IModel<TInput, TOutput, ModelMetadata<T>> model,
            TInput input,
            TOutput prediction)
        {
            var metadata = model.GetModelMetadata();
            return await Task.FromResult($"Model {metadata.ModelType} predicted output based on {metadata.FeatureCount} features.");
        }

        /// <summary>
        /// Default implementation for GetFeatureInteractionAsync.
        /// </summary>
        public static async Task<T> GetFeatureInteractionAsync<T>(
            HashSet<InterpretationMethod> enabledMethods,
            int feature1Index,
            int feature2Index)
        {
            if (!enabledMethods.Contains(InterpretationMethod.FeatureInteraction))
            {
                throw new InvalidOperationException("Feature interaction analysis is not enabled. Call EnableMethod first.");
            }

            var ops = MathHelper.GetNumericOperations<T>();
            return await Task.FromResult(ops.Zero);
        }

        /// <summary>
        /// Default implementation for ValidateFairnessAsync.
        /// </summary>
        public static async Task<FairnessMetrics<T>> ValidateFairnessAsync<T>(
            List<FairnessMetric> fairnessMetrics)
        {
            if (fairnessMetrics.Count == 0)
            {
                throw new InvalidOperationException("No fairness metrics configured. Call ConfigureFairness first.");
            }

            // Placeholder implementation - derived classes should override
            return await Task.FromResult(new FairnessMetrics<T>());
        }

        /// <summary>
        /// Default implementation for GetAnchorExplanationAsync.
        /// </summary>
        public static async Task<AnchorExplanation<T>> GetAnchorExplanationAsync<T>(
            HashSet<InterpretationMethod> enabledMethods,
            T threshold)
        {
            if (!enabledMethods.Contains(InterpretationMethod.Anchors))
            {
                throw new InvalidOperationException("Anchor explanations are not enabled. Call EnableMethod first.");
            }

            // Placeholder implementation - derived classes should override
            return await Task.FromResult(new AnchorExplanation<T>());
        }
    }
}