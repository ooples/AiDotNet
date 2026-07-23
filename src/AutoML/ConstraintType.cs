namespace AiDotNet.AutoML
{
    /// <summary>
    /// Defines the types of constraints that can be applied to AutoML search.
    /// </summary>
    public enum ConstraintType
    {
        /// <summary>
        /// Range constraint limiting a parameter to a specific range.
        /// </summary>
        Range,

        /// <summary>
        /// Dependency constraint between multiple parameters.
        /// </summary>
        Dependency,

        /// <summary>
        /// Exclusion constraint preventing certain parameter combinations.
        /// </summary>
        Exclusion,

        /// <summary>
        /// Resource constraint limiting compute resources.
        /// </summary>
        Resource,

        /// <summary>
        /// Custom constraint defined by an expression.
        /// </summary>
        Custom
    }
}

