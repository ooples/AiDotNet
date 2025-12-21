using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Represents a counterfactual explanation showing minimal changes needed for a different outcome.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class CounterfactualExplanation<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        /// <summary>
        /// Gets or sets the original input.
        /// </summary>
        public Tensor<T>? OriginalInput { get; set; }

        /// <summary>
        /// Gets or sets the counterfactual input (modified version).
        /// </summary>
        public Tensor<T>? CounterfactualInput { get; set; }

        /// <summary>
        /// Gets or sets the original prediction.
        /// </summary>
        public Tensor<T>? OriginalPrediction { get; set; }

        /// <summary>
        /// Gets or sets the counterfactual prediction.
        /// </summary>
        public Tensor<T>? CounterfactualPrediction { get; set; }

        /// <summary>
        /// Gets or sets the feature changes made.
        /// Keys are feature indices, values are the change amounts.
        /// </summary>
        public Dictionary<int, T> FeatureChanges { get; set; }

        /// <summary>
        /// Gets or sets the total distance between original and counterfactual.
        /// </summary>
        public T Distance { get; set; }

        /// <summary>
        /// Gets or sets the maximum number of changes allowed.
        /// </summary>
        public int MaxChanges { get; set; }

        /// <summary>
        /// Initializes a new instance of the CounterfactualExplanation class.
        /// </summary>
        public CounterfactualExplanation()
        {
            FeatureChanges = new Dictionary<int, T>();
            Distance = NumOps.Zero;
        }
    }
}
