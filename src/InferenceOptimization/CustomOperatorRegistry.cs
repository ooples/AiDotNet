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
        private readonly ConcurrentDictionary<string, SelectedOperatorEntry> _selectedOperators;
        private readonly ConcurrentDictionary<string, long> _operatorVersions;

        /// <summary>
        /// Gets the singleton instance of the registry
        /// </summary>
        public static CustomOperatorRegistry Instance => _instance.Value;

        private CustomOperatorRegistry()
        {
            _operators = new ConcurrentDictionary<string, List<ICustomOperator>>();
            _selectedOperators = new ConcurrentDictionary<string, SelectedOperatorEntry>();
            _operatorVersions = new ConcurrentDictionary<string, long>();
        }

        /// <summary>
        /// Registers a custom operator
        /// </summary>
        public void Register(ICustomOperator op)
        {
            if (op == null)
                throw new ArgumentNullException(nameof(op));

            // Bump the version after the operator set is updated.
            // This avoids stale cached selections without requiring coarse locking.
            void BumpVersion() => _operatorVersions.AddOrUpdate(op.Name, 1, (_, v) => v + 1);

            // Use AddOrUpdate with factory that always creates a new sorted list
            // This ensures thread-safety by never mutating existing lists
            _operators.AddOrUpdate(
                op.Name,
                _ => new List<ICustomOperator> { op },
                (_, existingList) =>
                {
                    // Create a new list with all existing operators plus the new one
                    // This avoids race conditions from modifying the existing list
                    List<ICustomOperator> newList;
                    lock (existingList)
                    {
                        newList = new List<ICustomOperator>(existingList) { op };
                    }
                    newList.Sort((a, b) => b.Priority.CompareTo(a.Priority));
                    return newList;
                });

            BumpVersion();
        }

        /// <summary>
        /// Gets the best available operator for the given name
        /// </summary>
        public ICustomOperator? GetOperator(string name)
        {
            if (string.IsNullOrEmpty(name))
                throw new ArgumentException("Operator name cannot be null or empty", nameof(name));

            while (true)
            {
                long version = _operatorVersions.GetOrAdd(name, 0);

                if (_selectedOperators.TryGetValue(name, out var existing) && existing.Version == version)
                {
                    return existing.Operator is NullOperator ? null : existing.Operator;
                }

                var selected = SelectOperatorOrNull(name);

                // Only publish the cached selection if the operator set version did not change while we were selecting.
                if (_operatorVersions.TryGetValue(name, out var current) && current == version)
                {
                    _selectedOperators[name] = new SelectedOperatorEntry(version, selected);
                    return selected is NullOperator ? null : selected;
                }

                // Operator set changed while selecting; retry to avoid caching a stale choice.
            }
        }

        private ICustomOperator SelectOperatorOrNull(string name)
        {
            if (!_operators.TryGetValue(name, out var candidates))
                return new NullOperator();

            lock (candidates)
            {
                // Find the highest priority supported operator
                var result = candidates.FirstOrDefault(op => op.IsSupported());
                return result ?? new NullOperator();
            }
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
            _operatorVersions.TryRemove(name, out _);
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
            _operatorVersions.Clear();
        }

        private readonly record struct SelectedOperatorEntry(long Version, ICustomOperator Operator);
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
