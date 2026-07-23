using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Represents a LIME (Local Interpretable Model-agnostic Explanations) explanation for a prediction.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class LimeExplanation<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        /// <summary>
        /// Gets or sets the feature importance scores for the explanation.
        /// Keys are feature indices, values are importance scores.
        /// </summary>
        public Dictionary<int, T> FeatureImportance { get; set; }

        /// <summary>
        /// Gets or sets the intercept of the linear approximation.
        /// </summary>
        public T Intercept { get; set; }

        /// <summary>
        /// Gets or sets the predicted value for the explained instance.
        /// </summary>
        public T PredictedValue { get; set; }

        /// <summary>
        /// Gets or sets the R-squared score of the local linear approximation.
        /// </summary>
        public T LocalModelScore { get; set; }

        /// <summary>
        /// Gets or sets the number of features used in the explanation.
        /// </summary>
        public int NumFeatures { get; set; }

        /// <summary>
        /// Initializes a new instance of the LimeExplanation class.
        /// </summary>
        public LimeExplanation()
        {
            FeatureImportance = new Dictionary<int, T>();
            Intercept = NumOps.Zero;
            PredictedValue = NumOps.Zero;
            LocalModelScore = NumOps.Zero;
        }
    }
}
