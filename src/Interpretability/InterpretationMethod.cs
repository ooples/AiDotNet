namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Enumeration of interpretation methods supported by interpretable models.
    /// </summary>
    public enum InterpretationMethod
    {
        /// <summary>
        /// SHAP (SHapley Additive exPlanations) values for feature importance.
        /// </summary>
        SHAP,

        /// <summary>
        /// LIME (Local Interpretable Model-agnostic Explanations) for local explanations.
        /// </summary>
        LIME,

        /// <summary>
        /// Partial dependence plots to show feature effects.
        /// </summary>
        PartialDependence,

        /// <summary>
        /// Counterfactual explanations to understand decision boundaries.
        /// </summary>
        Counterfactual,

        /// <summary>
        /// Anchor explanations for rule-based interpretations.
        /// </summary>
        Anchor,

        /// <summary>
        /// Feature importance analysis.
        /// </summary>
        FeatureImportance,

        /// <summary>
        /// Feature interaction analysis.
        /// </summary>
        FeatureInteraction
    }
}
