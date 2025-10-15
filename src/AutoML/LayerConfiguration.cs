using System;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Configuration for a single layer in a neural architecture
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class LayerConfiguration<T>
        where T : struct, IComparable<T>, IConvertible, IEquatable<T>
    {
        /// <summary>
        /// Gets or sets the type of layer
        /// </summary>
        public LayerType Type { get; set; }
        
        /// <summary>
        /// Gets or sets the number of units (neurons) in the layer
        /// </summary>
        public int Units { get; set; }
        
        /// <summary>
        /// Gets or sets the number of filters (for convolutional layers)
        /// </summary>
        public int Filters { get; set; }
        
        /// <summary>
        /// Gets or sets the kernel size (for convolutional layers)
        /// </summary>
        public int KernelSize { get; set; }
        
        /// <summary>
        /// Gets or sets the stride (for convolutional layers)
        /// </summary>
        public int Stride { get; set; }
        
        /// <summary>
        /// Gets or sets the pool size (for pooling layers)
        /// </summary>
        public int PoolSize { get; set; }
        
        /// <summary>
        /// Gets or sets the activation function
        /// </summary>
        public ActivationFunction Activation { get; set; }
        
        /// <summary>
        /// Gets or sets the dropout rate
        /// </summary>
        public T DropoutRate { get; set; }
        
        /// <summary>
        /// Gets or sets whether to return sequences (for recurrent layers)
        /// </summary>
        public bool ReturnSequences { get; set; }
        
        /// <summary>
        /// Initializes a new instance of the LayerConfiguration class
        /// </summary>
        public LayerConfiguration()
        {
            var ops = MathHelper.GetNumericOperations<T>();
            DropoutRate = ops.Zero;
            Stride = 1;
            Activation = ActivationFunction.ReLU;
        }
        
        /// <summary>
        /// Creates a deep copy of this layer configuration
        /// </summary>
        /// <returns>A deep copy of the layer configuration</returns>
        public LayerConfiguration<T> Clone()
        {
            return new LayerConfiguration<T>
            {
                Type = Type,
                Units = Units,
                Filters = Filters,
                KernelSize = KernelSize,
                Stride = Stride,
                PoolSize = PoolSize,
                Activation = Activation,
                DropoutRate = DropoutRate,
                ReturnSequences = ReturnSequences
            };
        }
    }
}