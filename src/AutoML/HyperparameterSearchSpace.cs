using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;

namespace AiDotNet.AutoML
{
    using System.Collections.Generic;
    using AiDotNet.Enums;

    /// <summary>
    /// Defines the search space for hyperparameter optimization
    /// </summary>
    public class HyperparameterSearchSpace
    {
        private readonly Dictionary<string, ParameterRange> parameters;

        public HyperparameterSearchSpace()
        {
            parameters = new Dictionary<string, ParameterRange>();
        }

        /// <summary>
        /// Adds a continuous parameter to the search space
        /// </summary>
        public HyperparameterSearchSpace AddContinuous(string name, double min, double max)
        {
            parameters[name] = new ParameterRange 
            { 
                MinValue = min, 
                MaxValue = max,
                Type = ParameterType.Continuous
            };
            return this;
        }

        /// <summary>
        /// Adds an integer parameter to the search space
        /// </summary>
        public HyperparameterSearchSpace AddInteger(string name, int min, int max)
        {
            parameters[name] = new ParameterRange 
            { 
                MinValue = min, 
                MaxValue = max,
                Type = ParameterType.Integer
            };
            return this;
        }

        /// <summary>
        /// Adds a categorical parameter to the search space
        /// </summary>
        public HyperparameterSearchSpace AddCategorical(string name, string[] values)
        {
            parameters[name] = new ParameterRange 
            { 
                Type = ParameterType.Categorical,
                CategoricalValues = values.Cast<object>().ToArray()
            };
            return this;
        }

        /// <summary>
        /// Gets the parameter ranges
        /// </summary>
        public Dictionary<string, ParameterRange> GetParameters()
        {
            return new Dictionary<string, ParameterRange>(parameters);
        }

        /// <summary>
        /// Gets the number of parameters in the search space
        /// </summary>
        public int Count => parameters.Count;

        /// <summary>
        /// Checks if a parameter exists in the search space
        /// </summary>
        public bool ContainsParameter(string name)
        {
            return parameters.ContainsKey(name);
        }

        /// <summary>
        /// Gets a specific parameter range
        /// </summary>
        public ParameterRange? GetParameter(string name)
        {
            return parameters.TryGetValue(name, out var range) ? range : null;
        }
    }

}