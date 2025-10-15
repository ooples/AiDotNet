namespace AiDotNet.Enums
{
    /// <summary>
    /// Types of visualizations for model interpretability
    /// </summary>
    public enum VisualizationType
    {
        /// <summary>
        /// Partial dependence plot
        /// </summary>
        PartialDependence,
        
        /// <summary>
        /// Feature importance chart
        /// </summary>
        FeatureImportance,
        
        /// <summary>
        /// SHAP summary plot
        /// </summary>
        ShapSummary,
        
        /// <summary>
        /// LIME explanation
        /// </summary>
        Lime,
        
        /// <summary>
        /// Decision tree visualization
        /// </summary>
        DecisionTree,
        
        /// <summary>
        /// Confusion matrix
        /// </summary>
        ConfusionMatrix,
        
        /// <summary>
        /// ROC curve
        /// </summary>
        RocCurve,
        
        /// <summary>
        /// Learning curve
        /// </summary>
        LearningCurve
    }
}