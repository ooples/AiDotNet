using AiDotNet.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Defines the search space for hyperparameters in AutoML
    /// </summary>
    public class HyperparameterSpace
    {
        private readonly Dictionary<string, ParameterRange> _parameters = new();
        private readonly Random _random;

        /// <summary>
        /// Initializes a new instance of the HyperparameterSpace class
        /// </summary>
        /// <param name="seed">Random seed for reproducibility</param>
        public HyperparameterSpace(int? seed = null)
        {
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
        }

        /// <summary>
        /// Gets the parameter definitions
        /// </summary>
        public IReadOnlyDictionary<string, ParameterRange> Parameters => _parameters;

        /// <summary>
        /// Adds a continuous parameter to the search space
        /// </summary>
        /// <param name="name">Parameter name</param>
        /// <param name="minValue">Minimum value</param>
        /// <param name="maxValue">Maximum value</param>
        /// <param name="logScale">Whether to use log scale</param>
        public void AddContinuous(string name, double minValue, double maxValue, bool logScale = false)
        {
            _parameters[name] = new ParameterRange
            {
                MinValue = minValue,
                MaxValue = maxValue,
                Type = ParameterType.Continuous,
                LogScale = logScale
            };
        }

        /// <summary>
        /// Adds an integer parameter to the search space
        /// </summary>
        /// <param name="name">Parameter name</param>
        /// <param name="minValue">Minimum value</param>
        /// <param name="maxValue">Maximum value</param>
        public void AddInteger(string name, int minValue, int maxValue)
        {
            _parameters[name] = new ParameterRange
            {
                MinValue = minValue,
                MaxValue = maxValue,
                Type = ParameterType.Integer
            };
        }

        /// <summary>
        /// Adds a categorical parameter to the search space
        /// </summary>
        /// <param name="name">Parameter name</param>
        /// <param name="values">Possible values</param>
        public void AddCategorical(string name, params object[] values)
        {
            _parameters[name] = new ParameterRange
            {
                Type = ParameterType.Categorical,
                CategoricalValues = values
            };
        }

        /// <summary>
        /// Adds a boolean parameter to the search space
        /// </summary>
        /// <param name="name">Parameter name</param>
        public void AddBoolean(string name)
        {
            _parameters[name] = new ParameterRange
            {
                Type = ParameterType.Boolean
            };
        }

        /// <summary>
        /// Adds a discrete parameter to the search space (alias for AddCategorical)
        /// </summary>
        /// <param name="name">Parameter name</param>
        /// <param name="values">Possible discrete values</param>
        /// <remarks>
        /// <para>
        /// This method adds a discrete parameter, which can take on any value from a specific set of options.
        /// It's an alias for AddCategorical, providing a more mathematically oriented name for the same concept.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> A discrete parameter can only take specific values from a predefined list.
        /// Think of it like:
        /// - Number of layers: could be 1, 2, 3, 4, or 5 (but not 2.5)
        /// - Activation function: could be "ReLU", "Sigmoid", or "Tanh" (but not "Something else")
        /// - Batch size: could be 16, 32, 64, or 128 (but not 33)
        ///
        /// This is different from continuous parameters which can take any value in a range.
        /// </para>
        /// </remarks>
        public void AddDiscreteParameter(string name, params object[] values)
        {
            AddCategorical(name, values);
        }

        /// <summary>
        /// Adds a continuous parameter to the search space (alias with descriptive name)
        /// </summary>
        /// <param name="name">Parameter name</param>
        /// <param name="minValue">Minimum value</param>
        /// <param name="maxValue">Maximum value</param>
        /// <param name="logScale">Whether to use log scale</param>
        /// <remarks>
        /// <para>
        /// This is an alias for AddContinuous, providing a more explicit method name.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> A continuous parameter can take any value within a specified range.
        /// For example, a learning rate could be any value between 0.001 and 0.1, including
        /// values like 0.00123456 or 0.05789.
        /// </para>
        /// </remarks>
        public void AddContinuousParameter(string name, double minValue, double maxValue, bool logScale = false)
        {
            AddContinuous(name, minValue, maxValue, logScale);
        }

        /// <summary>
        /// Samples a random point from the search space
        /// </summary>
        /// <returns>Dictionary of parameter values</returns>
        public Dictionary<string, object> Sample()
        {
            var result = new Dictionary<string, object>();

            foreach (var kvp in _parameters)
            {
                result[kvp.Key] = SampleParameter(kvp.Value);
            }

            return result;
        }

        /// <summary>
        /// Generates a grid of all possible parameter combinations
        /// </summary>
        /// <param name="stepsPerDimension">Number of steps for continuous parameters</param>
        /// <returns>List of parameter combinations</returns>
        public List<Dictionary<string, object>> GenerateGrid(int stepsPerDimension = 10)
        {
            var parameterValues = new Dictionary<string, List<object>>();

            foreach (var kvp in _parameters)
            {
                parameterValues[kvp.Key] = GenerateParameterValues(kvp.Value, stepsPerDimension);
            }

            return GenerateCombinations(parameterValues);
        }

        /// <summary>
        /// Validates a parameter configuration
        /// </summary>
        /// <param name="parameters">Parameters to validate</param>
        /// <returns>True if valid, false otherwise</returns>
        public bool Validate(Dictionary<string, object> parameters)
        {
            foreach (var kvp in parameters)
            {
                if (!_parameters.ContainsKey(kvp.Key))
                    return false;

                if (!ValidateParameter(kvp.Key, kvp.Value))
                    return false;
            }

            return parameters.Count == _parameters.Count;
        }

        /// <summary>
        /// Gets the total number of combinations for grid search
        /// </summary>
        /// <param name="stepsPerDimension">Number of steps for continuous parameters</param>
        /// <returns>Total combinations</returns>
        public long GetTotalCombinations(int stepsPerDimension = 10)
        {
            long total = 1;

            foreach (var range in _parameters.Values)
            {
                int count = range.Type switch
                {
                    ParameterType.Continuous => stepsPerDimension,
                    ParameterType.Integer => Convert.ToInt32(range.MaxValue) - Convert.ToInt32(range.MinValue) + 1,
                    ParameterType.Categorical => range.CategoricalValues?.Length ?? 0,
                    ParameterType.Boolean => 2,
                    _ => 1
                };

                total *= count;
            }

            return total;
        }

        private object SampleParameter(ParameterRange range)
        {
            return range.Type switch
            {
                ParameterType.Continuous => SampleContinuous(range),
                ParameterType.Integer => SampleInteger(range),
                ParameterType.Categorical => SampleCategorical(range),
                ParameterType.Boolean => _random.NextDouble() > 0.5,
                _ => throw new ArgumentException($"Unknown parameter type: {range.Type}")
            };
        }

        private double SampleContinuous(ParameterRange range)
        {
            double min = Convert.ToDouble(range.MinValue);
            double max = Convert.ToDouble(range.MaxValue);

            if (range.LogScale)
            {
                double logMin = Math.Log(min);
                double logMax = Math.Log(max);
                double logValue = logMin + _random.NextDouble() * (logMax - logMin);
                return Math.Exp(logValue);
            }
            else
            {
                return min + _random.NextDouble() * (max - min);
            }
        }

        private int SampleInteger(ParameterRange range)
        {
            int min = Convert.ToInt32(range.MinValue);
            int max = Convert.ToInt32(range.MaxValue);
            return _random.Next(min, max + 1);
        }

        private object SampleCategorical(ParameterRange range)
        {
            if (range.CategoricalValues == null || range.CategoricalValues.Length == 0)
                throw new InvalidOperationException("Categorical parameter has no values");

            int index = _random.Next(range.CategoricalValues.Length);
            return range.CategoricalValues[index];
        }

        private List<object> GenerateParameterValues(ParameterRange range, int steps)
        {
            return range.Type switch
            {
                ParameterType.Continuous => GenerateContinuousValues(range, steps),
                ParameterType.Integer => GenerateIntegerValues(range),
                ParameterType.Categorical => range.CategoricalValues?.ToList() ?? new List<object>(),
                ParameterType.Boolean => new List<object> { false, true },
                _ => new List<object>()
            };
        }

        private List<object> GenerateContinuousValues(ParameterRange range, int steps)
        {
            var values = new List<object>();
            double min = Convert.ToDouble(range.MinValue);
            double max = Convert.ToDouble(range.MaxValue);

            if (range.LogScale)
            {
                double logMin = Math.Log(min);
                double logMax = Math.Log(max);
                double logStep = (logMax - logMin) / (steps - 1);

                for (int i = 0; i < steps; i++)
                {
                    values.Add(Math.Exp(logMin + i * logStep));
                }
            }
            else
            {
                double step = (max - min) / (steps - 1);
                for (int i = 0; i < steps; i++)
                {
                    values.Add(min + i * step);
                }
            }

            return values;
        }

        private List<object> GenerateIntegerValues(ParameterRange range)
        {
            var values = new List<object>();
            int min = Convert.ToInt32(range.MinValue);
            int max = Convert.ToInt32(range.MaxValue);

            for (int i = min; i <= max; i++)
            {
                values.Add(i);
            }

            return values;
        }

        private List<Dictionary<string, object>> GenerateCombinations(Dictionary<string, List<object>> parameterValues)
        {
            var keys = parameterValues.Keys.ToList();
            var combinations = new List<Dictionary<string, object>>();
            GenerateCombinationsRecursive(parameterValues, keys, 0, new Dictionary<string, object>(), combinations);
            return combinations;
        }

        private void GenerateCombinationsRecursive(
            Dictionary<string, List<object>> parameterValues,
            List<string> keys,
            int keyIndex,
            Dictionary<string, object> current,
            List<Dictionary<string, object>> results)
        {
            if (keyIndex >= keys.Count)
            {
                results.Add(new Dictionary<string, object>(current));
                return;
            }

            string key = keys[keyIndex];
            foreach (var value in parameterValues[key])
            {
                current[key] = value;
                GenerateCombinationsRecursive(parameterValues, keys, keyIndex + 1, current, results);
            }
            current.Remove(key);
        }

        private bool ValidateParameter(string name, object value)
        {
            if (!_parameters.TryGetValue(name, out var range))
                return false;

            switch (range.Type)
            {
                case ParameterType.Continuous:
                    double d = Convert.ToDouble(value);
                    return d >= Convert.ToDouble(range.MinValue) && d <= Convert.ToDouble(range.MaxValue);

                case ParameterType.Integer:
                    int i = Convert.ToInt32(value);
                    return i >= Convert.ToInt32(range.MinValue) && i <= Convert.ToInt32(range.MaxValue);

                case ParameterType.Categorical:
                    return range.CategoricalValues?.Contains(value) ?? false;

                case ParameterType.Boolean:
                    return value is bool;

                default:
                    return false;
            }
        }
    }
}
