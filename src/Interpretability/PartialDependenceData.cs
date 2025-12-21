using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Represents partial dependence data showing how features affect predictions.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class PartialDependenceData<T>
    {
        /// <summary>
        /// Gets or sets the feature indices analyzed.
        /// </summary>
        public Vector<int> FeatureIndices { get; set; }

        /// <summary>
        /// Gets or sets the grid values used for each feature.
        /// Keys are feature indices, values are the grid points.
        /// </summary>
        public Dictionary<int, Vector<T>> GridValues { get; set; }

        /// <summary>
        /// Gets or sets the partial dependence values.
        /// </summary>
        public Matrix<T> PartialDependenceValues { get; set; }

        /// <summary>
        /// Gets or sets the grid resolution used.
        /// </summary>
        public int GridResolution { get; set; }

        /// <summary>
        /// Gets or sets individual conditional expectation (ICE) curves if available.
        /// </summary>
        public List<Matrix<T>> IceCurves { get; set; }

        /// <summary>
        /// Initializes a new instance of the PartialDependenceData class.
        /// </summary>
        public PartialDependenceData()
        {
            FeatureIndices = new Vector<int>(0);
            GridValues = new Dictionary<int, Vector<T>>();
            PartialDependenceValues = new Matrix<T>(0, 0);
            IceCurves = new List<Matrix<T>>();
        }
    }
}
