using System;
using System.Collections.Generic;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Architecture representation for neural architecture search
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class Architecture<T>
        where T : struct, IComparable<T>, IConvertible, IEquatable<T>
    {
        /// <summary>
        /// Gets or sets the layers in this architecture
        /// </summary>
        public List<LayerConfiguration<T>> Layers { get; set; }
        
        /// <summary>
        /// Initializes a new instance of the Architecture class
        /// </summary>
        public Architecture()
        {
            Layers = new List<LayerConfiguration<T>>();
        }
    }
}