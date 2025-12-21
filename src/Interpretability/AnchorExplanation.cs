using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Represents an anchor explanation providing rule-based interpretations.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class AnchorExplanation<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        /// <summary>
        /// Gets or sets the anchor rules (feature indices and their conditions).
        /// </summary>
        public Dictionary<int, (T Min, T Max)> AnchorRules { get; set; }

        /// <summary>
        /// Gets or sets the precision of the anchor (how often the anchor holds).
        /// </summary>
        public T Precision { get; set; }

        /// <summary>
        /// Gets or sets the coverage of the anchor (fraction of instances covered).
        /// </summary>
        public T Coverage { get; set; }

        /// <summary>
        /// Gets or sets the threshold used for anchor construction.
        /// </summary>
        public T Threshold { get; set; }

        /// <summary>
        /// Gets or sets the features involved in the anchor.
        /// </summary>
        public List<int> AnchorFeatures { get; set; }

        /// <summary>
        /// Gets or sets a human-readable description of the anchor rules.
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Initializes a new instance of the AnchorExplanation class.
        /// </summary>
        public AnchorExplanation()
        {
            AnchorRules = new Dictionary<int, (T Min, T Max)>();
            AnchorFeatures = new List<int>();
            Description = string.Empty;
            Precision = NumOps.Zero;
            Coverage = NumOps.Zero;
            Threshold = NumOps.Zero;
        }
    }
}
