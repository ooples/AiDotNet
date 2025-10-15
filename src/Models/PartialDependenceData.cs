using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.Models
{
    /// <summary>
    /// Partial dependence data showing how features affect predictions
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class PartialDependenceData<T>
    {
        private readonly INumericOperations<T> _ops;
        
        public PartialDependenceData()
        {
            _ops = MathHelper.GetNumericOperations<T>();
            FeatureIndices = Vector<int>.Empty();
            Grid = Matrix<T>.Empty();
            Values = Vector<T>.Empty();
            IndividualValues = Vector<T>.Empty();
        }
        
        /// <summary>
        /// Gets or sets the indices of features being analyzed
        /// </summary>
        public Vector<int> FeatureIndices { get; set; }
        
        /// <summary>
        /// Gets or sets the grid of feature values used for analysis
        /// </summary>
        public Matrix<T> Grid { get; set; }
        
        /// <summary>
        /// Gets or sets the average predicted values across the grid
        /// </summary>
        public Vector<T> Values { get; set; }
        
        /// <summary>
        /// Gets or sets the individual predicted values for each instance
        /// </summary>
        public Vector<T> IndividualValues { get; set; }
    }
}