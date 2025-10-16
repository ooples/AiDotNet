using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Represents a candidate neural architecture
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class ArchitectureCandidate<T>
    {
        /// <summary>
        /// Gets or sets the layers in this architecture
        /// </summary>
        public List<LayerConfiguration<T>> Layers { get; set; }
        
        /// <summary>
        /// Gets or sets the fitness score of this architecture
        /// </summary>
        public T Fitness { get; set; }
        
        /// <summary>
        /// Gets or sets the validation accuracy
        /// </summary>
        public T ValidationAccuracy { get; set; }
        
        /// <summary>
        /// Gets or sets the number of parameters
        /// </summary>
        public int Parameters { get; set; }
        
        /// <summary>
        /// Gets or sets the estimated FLOPs
        /// </summary>
        public long FLOPs { get; set; }
        
        /// <summary>
        /// Gets or sets whether this architecture has been evaluated
        /// </summary>
        public bool IsEvaluated { get; set; }
        
        /// <summary>
        /// Initializes a new instance of the ArchitectureCandidate class
        /// </summary>
        public ArchitectureCandidate()
        {
            Layers = new List<LayerConfiguration<T>>();
            IsEvaluated = false;
            var ops = MathHelper.GetNumericOperations<T>();
            Fitness = ops.Zero;
            ValidationAccuracy = ops.Zero;
        }
        
        /// <summary>
        /// Creates a deep copy of this architecture candidate
        /// </summary>
        /// <returns>A deep copy of the architecture candidate</returns>
        public ArchitectureCandidate<T> Clone()
        {
            return new ArchitectureCandidate<T>
            {
                Layers = Layers.Select(l => l.Clone()).ToList(),
                Fitness = Fitness,
                ValidationAccuracy = ValidationAccuracy,
                Parameters = Parameters,
                FLOPs = FLOPs,
                IsEvaluated = IsEvaluated
            };
        }
    }
}