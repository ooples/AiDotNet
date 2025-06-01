using System.Collections.Generic;
using System.Threading.Tasks;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for model interpretability and explainability
    /// </summary>
    public interface IInterpretableModel : IModel<Matrix<double>, Vector<double>, ModelMetaData<double>>
    {
        /// <summary>
        /// Gets global feature importance scores
        /// </summary>
        /// <returns>Dictionary mapping feature index to importance score</returns>
        Task<Dictionary<int, double>> GetGlobalFeatureImportanceAsync();

        /// <summary>
        /// Gets local feature importance for a specific prediction
        /// </summary>
        /// <param name="input">Input sample</param>
        /// <returns>Feature importance scores for this prediction</returns>
        Task<Dictionary<int, double>> GetLocalFeatureImportanceAsync(double[] input);

        /// <summary>
        /// Generates SHAP (SHapley Additive exPlanations) values
        /// </summary>
        /// <param name="inputs">Input samples</param>
        /// <returns>SHAP values for each sample and feature</returns>
        Task<double[,]> GetShapValuesAsync(double[][] inputs);

        /// <summary>
        /// Generates LIME (Local Interpretable Model-agnostic Explanations)
        /// </summary>
        /// <param name="input">Input sample to explain</param>
        /// <param name="numFeatures">Number of top features to include</param>
        /// <returns>LIME explanation</returns>
        Task<LimeExplanation> GetLimeExplanationAsync(double[] input, int numFeatures = 10);

        /// <summary>
        /// Gets partial dependence for specified features
        /// </summary>
        /// <param name="featureIndices">Indices of features to analyze</param>
        /// <param name="gridResolution">Number of points in the grid</param>
        /// <returns>Partial dependence data</returns>
        Task<PartialDependenceData> GetPartialDependenceAsync(int[] featureIndices, int gridResolution = 20);

        /// <summary>
        /// Generates counterfactual explanations
        /// </summary>
        /// <param name="input">Original input</param>
        /// <param name="desiredOutput">Desired output</param>
        /// <param name="maxChanges">Maximum number of features to change</param>
        /// <returns>Counterfactual explanation</returns>
        Task<CounterfactualExplanation> GetCounterfactualAsync(double[] input, double desiredOutput, int maxChanges = 5);

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
        Task<string> GenerateTextExplanationAsync(double[] input, double prediction);

        /// <summary>
        /// Gets interaction effects between features
        /// </summary>
        /// <param name="feature1Index">First feature index</param>
        /// <param name="feature2Index">Second feature index</param>
        /// <returns>Interaction effect strength</returns>
        Task<double> GetFeatureInteractionAsync(int feature1Index, int feature2Index);

        /// <summary>
        /// Validates model fairness across different groups
        /// </summary>
        /// <param name="inputs">Input samples</param>
        /// <param name="sensitiveFeatureIndex">Index of sensitive feature</param>
        /// <returns>Fairness metrics</returns>
        Task<FairnessMetrics> ValidateFairnessAsync(double[][] inputs, int sensitiveFeatureIndex);

        /// <summary>
        /// Gets anchors (sufficient conditions) for predictions
        /// </summary>
        /// <param name="input">Input sample</param>
        /// <param name="threshold">Precision threshold</param>
        /// <returns>Anchor explanation</returns>
        Task<AnchorExplanation> GetAnchorExplanationAsync(double[] input, double threshold = 0.95);
    }

    /// <summary>
    /// LIME explanation result
    /// </summary>
    public class LimeExplanation
    {
        public Dictionary<int, double> FeatureWeights { get; set; } = new();
        public double Intercept { get; set; }
        public double LocalScore { get; set; }
        public double Coverage { get; set; }
    }

    /// <summary>
    /// Partial dependence data
    /// </summary>
    public class PartialDependenceData
    {
        public int[] FeatureIndices { get; set; } = new int[0];
        public double[][] Grid { get; set; } = new double[0][];
        public double[] Values { get; set; } = new double[0];
        public double[] IndividualValues { get; set; } = new double[0];
    }

    /// <summary>
    /// Counterfactual explanation
    /// </summary>
    public class CounterfactualExplanation
    {
        public double[] OriginalInput { get; set; } = new double[0];
        public double[] CounterfactualInput { get; set; } = new double[0];
        public Dictionary<int, double> Changes { get; set; } = new();
        public double OriginalPrediction { get; set; }
        public double CounterfactualPrediction { get; set; }
        public double Distance { get; set; }
    }

    /// <summary>
    /// Fairness validation metrics
    /// </summary>
    public class FairnessMetrics
    {
        public double DemographicParity { get; set; }
        public double EqualOpportunity { get; set; }
        public double EqualizingOdds { get; set; }
        public double DisparateImpact { get; set; }
        public Dictionary<string, double> GroupMetrics { get; set; } = new();
    }

    /// <summary>
    /// Anchor explanation
    /// </summary>
    public class AnchorExplanation
    {
        public List<AnchorRule> Rules { get; set; } = new();
        public double Precision { get; set; }
        public double Coverage { get; set; }
    }

    /// <summary>
    /// Single anchor rule
    /// </summary>
    public class AnchorRule
    {
        public int FeatureIndex { get; set; }
        public string Operator { get; set; } = string.Empty;
        public double Value { get; set; }
        public string Description { get; set; } = string.Empty;
    }
}