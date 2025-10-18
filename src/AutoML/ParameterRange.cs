using AiDotNet.Enums;
using System;
using System.Collections.Generic;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Defines the range and type of a hyperparameter for AutoML search
    /// </summary>
    public class ParameterRange : ICloneable
    {
        /// <summary>
        /// The type of parameter (Integer, Float, Boolean, Categorical, etc.)
        /// </summary>
        public ParameterType Type { get; set; }

        /// <summary>
        /// The minimum value for numeric parameters
        /// </summary>
        public object? MinValue { get; set; }

        /// <summary>
        /// The maximum value for numeric parameters
        /// </summary>
        public object? MaxValue { get; set; }

        /// <summary>
        /// The step size for discrete parameters
        /// </summary>
        public double? Step { get; set; }

        /// <summary>
        /// List of possible values for categorical parameters
        /// </summary>
        public List<object>? CategoricalValues { get; set; }

        /// <summary>
        /// Whether to use logarithmic scale for sampling
        /// </summary>
        public bool UseLogScale { get; set; }

        /// <summary>
        /// Default value for the parameter
        /// </summary>
        public object? DefaultValue { get; set; }

        /// <summary>
        /// Creates a shallow copy of the ParameterRange
        /// </summary>
        public object Clone()
        {
            return new ParameterRange
            {
                Type = Type,
                MinValue = MinValue,
                MaxValue = MaxValue,
                Step = Step,
                CategoricalValues = CategoricalValues != null ? new List<object>(CategoricalValues) : null,
                UseLogScale = UseLogScale,
                DefaultValue = DefaultValue
            };
        }
    }
}
