namespace AiDotNet.Enums
{
    /// <summary>
    /// Defines the types of parameters that can be used in hyperparameter search
    /// </summary>
    public enum ParameterType
    {
        /// <summary>
        /// Integer parameter type
        /// </summary>
        Integer,

        /// <summary>
        /// Floating point parameter type
        /// </summary>
        Float,

        /// <summary>
        /// Boolean parameter type
        /// </summary>
        Boolean,

        /// <summary>
        /// Categorical parameter type (discrete choices)
        /// </summary>
        Categorical,

        /// <summary>
        /// Continuous parameter type
        /// </summary>
        Continuous
    }
}
