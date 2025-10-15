using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Enums;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for model interpretability and explainability
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <typeparam name="TInput">The input data type</typeparam>
    /// <typeparam name="TOutput">The output data type</typeparam>
    public interface IInterpretableModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
    {
        /// <summary>
        /// Gets global feature importance scores
        /// </summary>
        /// <returns>Dictionary mapping feature index to importance score</returns>
        Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync();

        /// <summary>
        /// Gets local feature importance for a specific prediction
        /// </summary>
        /// <param name="input">Input sample</param>
        /// <returns>Feature importance scores for this prediction</returns>
        Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(TInput input);

        /// <summary>
        /// Generates SHAP (SHapley Additive exPlanations) values
        /// </summary>
        /// <param name="inputs">Input samples</param>
        /// <returns>SHAP values for each sample and feature</returns>
        Task<Matrix<T>> GetShapValuesAsync(TInput inputs);

        /// <summary>
        /// Generates LIME (Local Interpretable Model-agnostic Explanations)
        /// </summary>
        /// <param name="input">Input sample to explain</param>
        /// <param name="numFeatures">Number of top features to include</param>
        /// <returns>LIME explanation</returns>
        Task<LimeExplanation<T>> GetLimeExplanationAsync(TInput input, int numFeatures = 10);

        /// <summary>
        /// Gets partial dependence for specified features
        /// </summary>
        /// <param name="featureIndices">Indices of features to analyze</param>
        /// <param name="gridResolution">Number of points in the grid</param>
        /// <returns>Partial dependence data</returns>
        Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20);

        /// <summary>
        /// Generates counterfactual explanations
        /// </summary>
        /// <param name="input">Original input</param>
        /// <param name="desiredOutput">Desired output</param>
        /// <param name="maxChanges">Maximum number of features to change</param>
        /// <returns>Counterfactual explanation</returns>
        Task<CounterfactualExplanation<T>> GetCounterfactualAsync(TInput input, TOutput desiredOutput, int maxChanges = 5);

        /// <summary>
        /// Gets model-specific interpretability information
        /// </summary>
        /// <returns>Dictionary of interpretability metrics</returns>
        Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync();

        /// <summary>
        /// Generates a text explanation for a prediction
        /// </summary>
        /// <param name="input">Input sample</param>
        /// <param name="prediction">Model prediction</param>
        /// <returns>Human-readable explanation</returns>
        Task<string> GenerateTextExplanationAsync(TInput input, TOutput prediction);

        /// <summary>
        /// Gets interaction effects between features
        /// </summary>
        /// <param name="feature1Index">First feature index</param>
        /// <param name="feature2Index">Second feature index</param>
        /// <returns>Interaction effect strength</returns>
        Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index);

        /// <summary>
        /// Validates model fairness across different groups
        /// </summary>
        /// <param name="inputs">Input samples</param>
        /// <param name="sensitiveFeatureIndex">Index of sensitive feature</param>
        /// <returns>Fairness metrics</returns>
        Task<FairnessMetrics<T>> ValidateFairnessAsync(TInput inputs, int sensitiveFeatureIndex);

        /// <summary>
        /// Gets anchors (sufficient conditions) for predictions
        /// </summary>
        /// <param name="input">Input sample</param>
        /// <param name="threshold">Precision threshold</param>
        /// <returns>Anchor explanation</returns>
        Task<AnchorExplanation<T>> GetAnchorExplanationAsync(TInput input, T threshold);
        
        /// <summary>
        /// Sets the base model to interpret
        /// </summary>
        /// <param name="model">The model to interpret</param>
        void SetBaseModel(IModel<TInput, TOutput, ModelMetadata<T>> model);
        
        /// <summary>
        /// Enables specific interpretation methods
        /// </summary>
        /// <param name="methods">The interpretation methods to enable</param>
        void EnableMethod(params InterpretationMethod[] methods);
        
        /// <summary>
        /// Configures fairness constraints
        /// </summary>
        /// <param name="sensitiveFeatures">Indices of sensitive features</param>
        /// <param name="fairnessMetrics">Fairness metrics to monitor</param>
        void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics);
    }
}