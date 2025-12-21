using System;
using System.Collections.Generic;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Defines a constraint for AutoML search to limit the search space or enforce requirements.
    /// </summary>
    public class SearchConstraint : ICloneable
    {
        /// <summary>
        /// Gets or sets the name of the constraint.
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the type of constraint.
        /// </summary>
        public ConstraintType Type { get; set; }

        /// <summary>
        /// Gets or sets the parameter names involved in this constraint.
        /// </summary>
        public List<string> ParameterNames { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the constraint expression or rule.
        /// </summary>
        public string Expression { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the minimum value for range constraints.
        /// </summary>
        public double? MinValue { get; set; }

        /// <summary>
        /// Gets or sets the maximum value for range constraints.
        /// </summary>
        public double? MaxValue { get; set; }

        /// <summary>
        /// Gets or sets whether this constraint is a hard constraint (must be satisfied) or soft constraint (preferred).
        /// </summary>
        public bool IsHardConstraint { get; set; } = true;

        /// <summary>
        /// Gets or sets additional metadata for the constraint.
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();

        /// <summary>
        /// Creates a clone of this search constraint.
        /// </summary>
        /// <returns>A new SearchConstraint with the same values</returns>
        public object Clone()
        {
            return new SearchConstraint
            {
                Name = Name,
                Type = Type,
                ParameterNames = new List<string>(ParameterNames),
                Expression = Expression,
                MinValue = MinValue,
                MaxValue = MaxValue,
                IsHardConstraint = IsHardConstraint,
                Metadata = new Dictionary<string, object>(Metadata)
            };
        }
    }

}
