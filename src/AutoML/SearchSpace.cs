using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Search space definition for neural architecture search
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class SearchSpace<T>
        where T : struct, IComparable<T>, IConvertible, IEquatable<T>
    {
        /// <summary>
        /// Gets or sets the available layer types
        /// </summary>
        public List<LayerType> LayerTypes { get; set; } = new List<LayerType>();
        
        /// <summary>
        /// Gets or sets the available activation functions
        /// </summary>
        public List<ActivationFunction> ActivationFunctions { get; set; } = new List<ActivationFunction>();
        
        /// <summary>
        /// Gets or sets the maximum number of layers
        /// </summary>
        public int MaxLayers { get; set; }
        
        /// <summary>
        /// Gets or sets the maximum units per layer
        /// </summary>
        public int MaxUnitsPerLayer { get; set; }
        
        /// <summary>
        /// Gets or sets the maximum number of filters
        /// </summary>
        public int MaxFilters { get; set; }
        
        /// <summary>
        /// Gets or sets the available kernel sizes
        /// </summary>
        public Vector<int> KernelSizes { get; set; }
        
        /// <summary>
        /// Gets or sets the available dropout rates
        /// </summary>
        public Vector<T> DropoutRates { get; set; }
        
        /// <summary>
        /// Gets or sets the mutation rate for evolutionary strategies
        /// </summary>
        public T MutationRate { get; set; }
        
        /// <summary>
        /// Gets or sets the input dimension
        /// </summary>
        public int InputDimension { get; set; }
        
        /// <summary>
        /// Initializes a new instance of the SearchSpace class
        /// </summary>
        public SearchSpace()
        {
            var ops = MathHelper.GetNumericOperations<T>();
            KernelSizes = new Vector<int>(new[] { 3, 5, 7 });
            DropoutRates = new Vector<T>(new[] { 
                ops.FromDouble(0.0), 
                ops.FromDouble(0.1), 
                ops.FromDouble(0.2), 
                ops.FromDouble(0.3), 
                ops.FromDouble(0.5) 
            });
            MutationRate = ops.FromDouble(0.1);
        }
    }
}