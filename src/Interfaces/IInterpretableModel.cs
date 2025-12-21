using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for models that support interpretability features.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public interface IInterpretableModel<T>
    {
        /// <summary>
        /// Gets the global feature importance across all predictions.
        /// </summary>
        /// <returns>A dictionary mapping feature indices to importance scores.</returns>
        Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync();

        /// <summary>
        /// Gets the local feature importance for a specific input.
        /// </summary>
        /// <param name="input">The input to analyze.</param>
        /// <returns>A dictionary mapping feature indices to importance scores.</returns>
        Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Tensor<T> input);

        /// <summary>
        /// Gets SHAP values for the given inputs.
        /// </summary>
        /// <param name="inputs">The inputs to analyze.</param>
        /// <returns>A matrix containing SHAP values.</returns>
        Task<Matrix<T>> GetShapValuesAsync(Tensor<T> inputs);

        /// <summary>
        /// Gets LIME explanation for a specific input.
        /// </summary>
        /// <param name="input">The input to explain.</param>
        /// <param name="numFeatures">The number of features to include in the explanation.</param>
        /// <returns>A LIME explanation.</returns>
        Task<LimeExplanation<T>> GetLimeExplanationAsync(Tensor<T> input, int numFeatures = 10);

        /// <summary>
        /// Gets partial dependence data for specified features.
        /// </summary>
        /// <param name="featureIndices">The feature indices to analyze.</param>
        /// <param name="gridResolution">The grid resolution to use.</param>
        /// <returns>Partial dependence data.</returns>
        Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20);

        /// <summary>
        /// Gets counterfactual explanation for a given input and desired output.
        /// </summary>
        /// <param name="input">The input to analyze.</param>
        /// <param name="desiredOutput">The desired output.</param>
        /// <param name="maxChanges">The maximum number of changes allowed.</param>
        /// <returns>A counterfactual explanation.</returns>
        Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Tensor<T> input, Tensor<T> desiredOutput, int maxChanges = 5);

        /// <summary>
        /// Gets model-specific interpretability information.
        /// </summary>
        /// <returns>A dictionary of model-specific interpretability information.</returns>
        Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync();

        /// <summary>
        /// Generates a text explanation for a prediction.
        /// </summary>
        /// <param name="input">The input data.</param>
        /// <param name="prediction">The prediction made by the model.</param>
        /// <returns>A text explanation of the prediction.</returns>
        Task<string> GenerateTextExplanationAsync(Tensor<T> input, Tensor<T> prediction);

        /// <summary>
        /// Gets feature interaction effects between two features.
        /// </summary>
        /// <param name="feature1Index">The index of the first feature.</param>
        /// <param name="feature2Index">The index of the second feature.</param>
        /// <returns>The interaction effect value.</returns>
        Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index);

        /// <summary>
        /// Validates fairness metrics for the given inputs.
        /// </summary>
        /// <param name="inputs">The inputs to analyze.</param>
        /// <param name="sensitiveFeatureIndex">The index of the sensitive feature.</param>
        /// <returns>Fairness metrics results.</returns>
        Task<FairnessMetrics<T>> ValidateFairnessAsync(Tensor<T> inputs, int sensitiveFeatureIndex);

        /// <summary>
        /// Gets anchor explanation for a given input.
        /// </summary>
        /// <param name="input">The input to explain.</param>
        /// <param name="threshold">The threshold for anchor construction.</param>
        /// <returns>An anchor explanation.</returns>
        Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Tensor<T> input, T threshold);

        /// <summary>
        /// Sets the base model for interpretability analysis.
        /// </summary>
        /// <typeparam name="TInput">The input type for the model.</typeparam>
        /// <typeparam name="TOutput">The output type for the model.</typeparam>
        /// <param name="model">The base model. Must implement IFullModel.</param>
        void SetBaseModel<TInput, TOutput>(IFullModel<T, TInput, TOutput> model);

        /// <summary>
        /// Enables specific interpretation methods.
        /// </summary>
        /// <param name="methods">The methods to enable.</param>
        void EnableMethod(params InterpretationMethod[] methods);

        /// <summary>
        /// Configures fairness evaluation settings.
        /// </summary>
        /// <param name="sensitiveFeatures">The indices of sensitive features.</param>
        /// <param name="fairnessMetrics">The fairness metrics to evaluate.</param>
        void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics);
    }
}
