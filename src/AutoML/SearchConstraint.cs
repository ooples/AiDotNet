namespace AiDotNet.AutoML
{
    /// <summary>
    /// Search constraint for AutoML
    /// </summary>
    public class SearchConstraint
    {
        /// <summary>
        /// Gets or sets the name of the constraint
        /// </summary>
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the type of constraint
        /// </summary>
        public ConstraintType Type { get; set; }
        
        /// <summary>
        /// Gets or sets the value for the constraint
        /// </summary>
        public object? Value { get; set; }
    }
}