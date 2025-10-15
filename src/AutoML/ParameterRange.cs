namespace AiDotNet.AutoML
{
    /// <summary>
    /// Represents a parameter range for hyperparameter search
    /// </summary>
    public class ParameterRange
    {
        /// <summary>
        /// Gets or sets the minimum value for the parameter
        /// </summary>
        public object? MinValue { get; set; }
        
        /// <summary>
        /// Gets or sets the maximum value for the parameter
        /// </summary>
        public object? MaxValue { get; set; }
        
        /// <summary>
        /// Gets or sets the type of parameter
        /// </summary>
        public ParameterType Type { get; set; }
        
        /// <summary>
        /// Gets or sets the categorical values for categorical parameters
        /// </summary>
        public object[]? CategoricalValues { get; set; }
        
        /// <summary>
        /// Gets or sets whether to use log scale for continuous parameters
        /// </summary>
        public bool LogScale { get; set; }
    }
}