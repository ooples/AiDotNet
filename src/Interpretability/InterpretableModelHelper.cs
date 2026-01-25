
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Provides helper methods for interpretable model functionality.
    /// </summary>
    public static class InterpretableModelHelper
    {
        /// <summary>
        /// Gets the global feature importance across all predictions.
        /// </summary>
        /// <typeparam name="T">The numeric type for calculations.</typeparam>
        /// <param name="model">The model to analyze.</param>
        /// <param name="enabledMethods">The set of enabled interpretation methods.</param>
        /// <returns>A dictionary mapping feature indices to importance scores.</returns>
        public static Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync<T>(
            IInterpretableModel<T> model,
            HashSet<InterpretationMethod> enabledMethods)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (enabledMethods == null)
                throw new ArgumentNullException(nameof(enabledMethods));

            if (!enabledMethods.Contains(InterpretationMethod.FeatureImportance))
            {
                throw new InvalidOperationException("FeatureImportance method is not enabled.");
            }

            return model.GetGlobalFeatureImportanceAsync();
        }

        /// <summary>
        /// Gets the local feature importance for a specific input.
        /// </summary>
        /// <typeparam name="T">The numeric type for calculations.</typeparam>
        /// <param name="model">The model to analyze.</param>
        /// <param name="enabledMethods">The set of enabled interpretation methods.</param>
        /// <param name="input">The input to analyze.</param>
        /// <returns>A dictionary mapping feature indices to importance scores.</returns>
        public static Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync<T>(
            IInterpretableModel<T> model,
            HashSet<InterpretationMethod> enabledMethods,
            Tensor<T> input)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (enabledMethods == null)
                throw new ArgumentNullException(nameof(enabledMethods));
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            if (!enabledMethods.Contains(InterpretationMethod.FeatureImportance))
            {
                throw new InvalidOperationException("FeatureImportance method is not enabled.");
            }

            return model.GetLocalFeatureImportanceAsync(input);
        }

        /// <summary>
        /// Gets SHAP values for the given inputs.
        /// </summary>
        /// <typeparam name="T">The numeric type for calculations.</typeparam>
        /// <param name="model">The model to analyze.</param>
        /// <param name="enabledMethods">The set of enabled interpretation methods.</param>
        /// <param name="inputs">The inputs to analyze.</param>
        /// <returns>A matrix containing SHAP values.</returns>
        public static Task<Matrix<T>> GetShapValuesAsync<T>(
            IInterpretableModel<T> model,
            HashSet<InterpretationMethod> enabledMethods,
            Tensor<T> inputs)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (enabledMethods == null)
                throw new ArgumentNullException(nameof(enabledMethods));
            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));

            if (!enabledMethods.Contains(InterpretationMethod.SHAP))
            {
                throw new InvalidOperationException("SHAP method is not enabled.");
            }

            return model.GetShapValuesAsync(inputs);
        }

        /// <summary>
        /// Gets LIME explanation for a specific input.
        /// </summary>
        /// <typeparam name="T">The numeric type for calculations.</typeparam>
        /// <param name="model">The model to analyze.</param>
        /// <param name="enabledMethods">The set of enabled interpretation methods.</param>
        /// <param name="input">The input to explain.</param>
        /// <param name="numFeatures">The number of features to include in the explanation.</param>
        /// <returns>A LIME explanation.</returns>
        public static Task<LimeExplanation<T>> GetLimeExplanationAsync<T>(
            IInterpretableModel<T> model,
            HashSet<InterpretationMethod> enabledMethods,
            Tensor<T> input,
            int numFeatures = 10)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (enabledMethods == null)
                throw new ArgumentNullException(nameof(enabledMethods));
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            if (!enabledMethods.Contains(InterpretationMethod.LIME))
            {
                throw new InvalidOperationException("LIME method is not enabled.");
            }

            return model.GetLimeExplanationAsync(input, numFeatures);
        }

        /// <summary>
        /// Gets partial dependence data for specified features.
        /// </summary>
        /// <typeparam name="T">The numeric type for calculations.</typeparam>
        /// <param name="model">The model to analyze.</param>
        /// <param name="enabledMethods">The set of enabled interpretation methods.</param>
        /// <param name="featureIndices">The feature indices to analyze.</param>
        /// <param name="gridResolution">The grid resolution to use.</param>
        /// <returns>Partial dependence data.</returns>
        public static Task<PartialDependenceData<T>> GetPartialDependenceAsync<T>(
            IInterpretableModel<T> model,
            HashSet<InterpretationMethod> enabledMethods,
            Vector<int> featureIndices,
            int gridResolution = 20)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (enabledMethods == null)
                throw new ArgumentNullException(nameof(enabledMethods));
            if (featureIndices == null)
                throw new ArgumentNullException(nameof(featureIndices));

            if (!enabledMethods.Contains(InterpretationMethod.PartialDependence))
            {
                throw new InvalidOperationException("PartialDependence method is not enabled.");
            }

            return model.GetPartialDependenceAsync(featureIndices, gridResolution);
        }

        /// <summary>
        /// Gets counterfactual explanation for a given input and desired output.
        /// </summary>
        /// <typeparam name="T">The numeric type for calculations.</typeparam>
        /// <param name="model">The model to analyze.</param>
        /// <param name="enabledMethods">The set of enabled interpretation methods.</param>
        /// <param name="input">The input to analyze.</param>
        /// <param name="desiredOutput">The desired output.</param>
        /// <param name="maxChanges">The maximum number of changes allowed.</param>
        /// <returns>A counterfactual explanation.</returns>
        public static Task<CounterfactualExplanation<T>> GetCounterfactualAsync<T>(
            IInterpretableModel<T> model,
            HashSet<InterpretationMethod> enabledMethods,
            Tensor<T> input,
            Tensor<T> desiredOutput,
            int maxChanges = 5)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (enabledMethods == null)
                throw new ArgumentNullException(nameof(enabledMethods));
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (desiredOutput == null)
                throw new ArgumentNullException(nameof(desiredOutput));

            if (!enabledMethods.Contains(InterpretationMethod.Counterfactual))
            {
                throw new InvalidOperationException("Counterfactual method is not enabled.");
            }

            return model.GetCounterfactualAsync(input, desiredOutput, maxChanges);
        }

        /// <summary>
        /// Gets model-specific interpretability information.
        /// </summary>
        /// <typeparam name="T">The numeric type for calculations.</typeparam>
        /// <param name="model">The model to analyze.</param>
        /// <returns>A dictionary of model-specific interpretability information.</returns>
        public static Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync<T>(
            IInterpretableModel<T> model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            return model.GetModelSpecificInterpretabilityAsync();
        }

        /// <summary>
        /// Generates a text explanation for a prediction.
        /// </summary>
        /// <typeparam name="T">The numeric type for calculations.</typeparam>
        /// <param name="model">The model to analyze.</param>
        /// <param name="input">The input data.</param>
        /// <param name="prediction">The prediction made by the model.</param>
        /// <returns>A text explanation of the prediction.</returns>
        public static Task<string> GenerateTextExplanationAsync<T>(
            IInterpretableModel<T> model,
            Tensor<T> input,
            Tensor<T> prediction)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (prediction == null)
                throw new ArgumentNullException(nameof(prediction));

            return model.GenerateTextExplanationAsync(input, prediction);
        }

        /// <summary>
        /// Gets feature interaction effects between two features.
        /// </summary>
        /// <typeparam name="T">The numeric type for calculations.</typeparam>
        /// <param name="enabledMethods">The set of enabled interpretation methods.</param>
        /// <param name="feature1Index">The index of the first feature.</param>
        /// <param name="feature2Index">The index of the second feature.</param>
        /// <returns>The interaction effect value.</returns>
        public static Task<T> GetFeatureInteractionAsync<T>(
            HashSet<InterpretationMethod> enabledMethods,
            int feature1Index,
            int feature2Index)
        {
            if (enabledMethods == null)
                throw new ArgumentNullException(nameof(enabledMethods));

            if (!enabledMethods.Contains(InterpretationMethod.FeatureInteraction))
            {
                throw new InvalidOperationException("FeatureInteraction method is not enabled.");
            }

            // Return placeholder implementation - return zero for numeric type T
            var numOps = MathHelper.GetNumericOperations<T>();
            return Task.FromResult(numOps.Zero);
        }

        /// <summary>
        /// Validates fairness metrics for the given inputs.
        /// </summary>
        /// <typeparam name="T">The numeric type for calculations.</typeparam>
        /// <param name="fairnessMetrics">The fairness metrics to validate.</param>
        /// <returns>Fairness metrics results.</returns>
        public static Task<FairnessMetrics<T>> ValidateFairnessAsync<T>(
            List<FairnessMetric> fairnessMetrics)
        {
            if (fairnessMetrics == null)
                throw new ArgumentNullException(nameof(fairnessMetrics));

            // Return placeholder implementation with zero values for all metrics
            var numOps = MathHelper.GetNumericOperations<T>();
            return Task.FromResult(new FairnessMetrics<T>(
                demographicParity: numOps.Zero,
                equalOpportunity: numOps.Zero,
                equalizedOdds: numOps.Zero,
                predictiveParity: numOps.Zero,
                disparateImpact: numOps.Zero,
                statisticalParityDifference: numOps.Zero));
        }

        /// <summary>
        /// Gets anchor explanation for a given input.
        /// </summary>
        /// <typeparam name="T">The numeric type for calculations.</typeparam>
        /// <param name="model">The model to analyze.</param>
        /// <param name="enabledMethods">The set of enabled interpretation methods.</param>
        /// <param name="input">The input to explain.</param>
        /// <param name="threshold">The threshold for anchor construction.</param>
        /// <returns>An anchor explanation.</returns>
        public static Task<AnchorExplanation<T>> GetAnchorExplanationAsync<T>(
            IInterpretableModel<T> model,
            HashSet<InterpretationMethod> enabledMethods,
            Tensor<T> input,
            T threshold)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (enabledMethods == null)
                throw new ArgumentNullException(nameof(enabledMethods));
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            if (!enabledMethods.Contains(InterpretationMethod.Anchor))
            {
                throw new InvalidOperationException("Anchor method is not enabled.");
            }

            return model.GetAnchorExplanationAsync(input, threshold);
        }
    }
}
