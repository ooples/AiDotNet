using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.InferenceOptimization
{
    /// <summary>
    /// Thread-safe registry for managing custom operators with automatic fallback
    /// </summary>
    public sealed class CustomOperatorRegistry
    {
        private static readonly Lazy<CustomOperatorRegistry> _instance =
            new Lazy<CustomOperatorRegistry>(() => new CustomOperatorRegistry());

        private readonly ConcurrentDictionary<string, List<ICustomOperator>> _operators;
        private readonly ConcurrentDictionary<string, ICustomOperator> _selectedOperators;

        /// <summary>
        /// Gets the singleton instance of the registry
        /// </summary>
        public static CustomOperatorRegistry Instance => _instance.Value;

        private CustomOperatorRegistry()
        {
            _operators = new ConcurrentDictionary<string, List<ICustomOperator>>();
            _selectedOperators = new ConcurrentDictionary<string, ICustomOperator>();
        }

        /// <summary>
        /// Registers a custom operator
        /// </summary>
        public void Register(ICustomOperator op)
        {
            if (op == null)
                throw new ArgumentNullException(nameof(op));

            _operators.AddOrUpdate(
                op.Name,
                _ => new List<ICustomOperator> { op },
                (_, list) =>
                {
                    lock (list)
                    {
                        list.Add(op);
                        list.Sort((a, b) => b.Priority.CompareTo(a.Priority));
                    }
                    return list;
                });

            // Clear cached selection to force re-evaluation
            _selectedOperators.TryRemove(op.Name, out _);
        }

        /// <summary>
        /// Gets the best available operator for the given name
        /// </summary>
        public ICustomOperator? GetOperator(string name)
        {
            if (string.IsNullOrEmpty(name))
                throw new ArgumentException("Operator name cannot be null or empty", nameof(name));

            return _selectedOperators.GetOrAdd(name, key =>
            {
                if (!_operators.TryGetValue(key, out var candidates))
                    return new NullOperator();

                lock (candidates)
                {
                    // Find the highest priority supported operator
                    var result = candidates.FirstOrDefault(op => op.IsSupported());
                    return result ?? new NullOperator();
                }
            }) is NullOperator ? null : _selectedOperators[name];
        }

        /// <summary>
        /// Gets a typed operator
        /// </summary>
        public ICustomOperator<T>? GetOperator<T>(string name) where T : struct
        {
            return GetOperator(name) as ICustomOperator<T>;
        }

        /// <summary>
        /// Internal marker type for null operators
        /// </summary>
        private sealed class NullOperator : ICustomOperator
        {
            public string Name => string.Empty;
            public string Version => string.Empty;
            public int Priority => int.MinValue;
            public bool IsSupported() => false;
            public double EstimatedSpeedup() => 0;
        }

        /// <summary>
        /// Checks if an operator is available
        /// </summary>
        public bool HasOperator(string name)
        {
            return GetOperator(name) != null;
        }

        /// <summary>
        /// Unregisters all operators with the given name
        /// </summary>
        public void Unregister(string name)
        {
            _operators.TryRemove(name, out _);
            _selectedOperators.TryRemove(name, out _);
        }

        /// <summary>
        /// Gets all registered operator names
        /// </summary>
        public IEnumerable<string> GetRegisteredOperatorNames()
        {
            return _operators.Keys.ToArray();
        }

        /// <summary>
        /// Gets detailed information about all registered operators
        /// </summary>
        public Dictionary<string, List<OperatorInfo>> GetOperatorInfo()
        {
            var result = new Dictionary<string, List<OperatorInfo>>();

            foreach (var kvp in _operators)
            {
                lock (kvp.Value)
                {
                    result[kvp.Key] = kvp.Value.Select(op => new OperatorInfo
                    {
                        Name = op.Name,
                        Version = op.Version,
                        Priority = op.Priority,
                        IsSupported = op.IsSupported(),
                        EstimatedSpeedup = op.EstimatedSpeedup(),
                        Type = op.GetType().FullName ?? op.GetType().Name
                    }).ToList();
                }
            }

            return result;
        }

        /// <summary>
        /// Clears all registered operators
        /// </summary>
        public void Clear()
        {
            _operators.Clear();
            _selectedOperators.Clear();
        }
    }

    /// <summary>
    /// Information about a registered operator
    /// </summary>
    public class OperatorInfo
    {
        public string Name { get; set; } = string.Empty;
        public string Version { get; set; } = string.Empty;
        public int Priority { get; set; }
        public bool IsSupported { get; set; }
        public double EstimatedSpeedup { get; set; }
        public string Type { get; set; } = string.Empty;
    }
}
