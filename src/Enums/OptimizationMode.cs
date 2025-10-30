namespace AiDotNet.Enums
{
    /// <summary>
    /// Specifies the mode of optimization for an optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// OptimizationMode determines what aspects of a model the optimizer will modify during the optimization process.
    /// This can include feature selection (choosing which features to use), parameter adjustment (modifying model parameters),
    /// or both.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as choosing what the optimizer is allowed to change. It can select
    /// which features (input variables) to use, adjust the model's internal parameters, or do both. This gives you
    /// control over how the optimizer improves your model.</para>
    /// </remarks>
    public enum OptimizationMode
    {
        /// <summary>
        /// Optimize only feature selection (which features to include in the model).
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b> In this mode, the optimizer only decides which features (input variables)
        /// should be used in the model. It doesn't change the model's internal parameters.</para>
        /// </remarks>
        FeatureSelectionOnly = 0,

        /// <summary>
        /// Optimize only model parameters (adjust existing model parameters).
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b> In this mode, the optimizer only adjusts the model's internal parameters
        /// (like weights and biases). It doesn't change which features are used.</para>
        /// </remarks>
        ParametersOnly = 1,

        /// <summary>
        /// Optimize both feature selection and model parameters.
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b> In this mode, the optimizer can both select which features to use AND
        /// adjust the model's internal parameters. This gives the optimizer the most flexibility but may take longer.</para>
        /// </remarks>
        Both = 2
    }
}
