using System;
using System.Collections.Generic;
using AiDotNet.Enums;

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
        /// Creates a deep copy of the ParameterRange, including deep cloning of reference-type properties
        /// </summary>
        /// <returns>A deep clone of this ParameterRange</returns>
        /// <remarks>
        /// This method performs deep cloning for all properties:
        /// - MinValue, MaxValue, DefaultValue: Deep cloned if they implement ICloneable, otherwise copied by reference (safe for value types and strings)
        /// - CategoricalValues: Each element is deep cloned if it implements ICloneable, otherwise copied by reference
        /// </remarks>
        public object Clone()
        {
            return new ParameterRange
            {
                Type = Type,
                MinValue = DeepCloneObject(MinValue),
                MaxValue = DeepCloneObject(MaxValue),
                Step = Step,
                CategoricalValues = CategoricalValues != null ? DeepCloneList(CategoricalValues) : null,
                UseLogScale = UseLogScale,
                DefaultValue = DeepCloneObject(DefaultValue)
            };
        }

        /// <summary>
        /// Deep clones an object if possible, otherwise returns the object itself.
        /// </summary>
        /// <param name="obj">The object to clone</param>
        /// <returns>A deep clone if the object implements ICloneable, otherwise the original object</returns>
        /// <remarks>
        /// Value types and strings are immutable and safe to return directly.
        /// Reference types implementing ICloneable are deep cloned via their Clone() method.
        /// Other reference types are returned by reference (shallow copy).
        /// </remarks>
        private static object? DeepCloneObject(object? obj)
        {
            if (obj == null)
                return null;

            // Value types and strings are immutable, safe to return as-is
            var type = obj.GetType();
            if (type.IsValueType || obj is string)
                return obj;

            // If object implements ICloneable, use its Clone method
            if (obj is ICloneable cloneable)
                return cloneable.Clone();

            // For other reference types, return by reference (shallow copy)
            // This is safe for immutable types but may cause issues with mutable types
            return obj;
        }

        /// <summary>
        /// Deep clones a list of objects, cloning each element if possible.
        /// </summary>
        /// <param name="list">The list to clone</param>
        /// <returns>A new list with deep-cloned elements</returns>
        private static List<object> DeepCloneList(List<object> list)
        {
            var clonedList = new List<object>(list.Count);
            foreach (var item in list)
            {
                clonedList.Add(DeepCloneObject(item) ?? throw new InvalidOperationException("Cloned object cannot be null"));
            }
            return clonedList;
        }
    }
}
