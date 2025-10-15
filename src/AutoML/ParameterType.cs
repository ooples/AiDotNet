namespace AiDotNet.AutoML
{
    /// <summary>
    /// Parameter types for AutoML search
    /// </summary>
    public enum ParameterType
    {
        /// <summary>
        /// Continuous numeric values (e.g., learning rate)
        /// </summary>
        Continuous,
        
        /// <summary>
        /// Integer values (e.g., number of trees)
        /// </summary>
        Integer,
        
        /// <summary>
        /// Categorical values (e.g., activation function types)
        /// </summary>
        Categorical,
        
        /// <summary>
        /// Boolean values (e.g., use bias)
        /// </summary>
        Boolean
    }
}