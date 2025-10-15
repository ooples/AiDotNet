using AiDotNet.LinearAlgebra;
using System;

namespace AiDotNet.Extensions
{
    /// <summary>
    /// Extension methods for iterating over tensor elements efficiently.
    /// </summary>
    public static class TensorIterationExtensions
    {
        /// <summary>
        /// Applies an action to each element of the tensor.
        /// </summary>
        public static void ForEach<T>(this Tensor<T> tensor, Action<int[], T> action)
        {
            var indices = new int[tensor.Shape.Length];
            ForEachRecursive(tensor, indices, 0, action);
        }
        
        /// <summary>
        /// Applies a function to each element and stores the result in a new tensor.
        /// </summary>
        public static Tensor<T> Map<T>(this Tensor<T> tensor, Func<T, T> func)
        {
            var result = new Tensor<T>(tensor.Shape);
            tensor.ForEach((indices, value) => result[indices] = func(value));
            return result;
        }
        
        /// <summary>
        /// Applies a function to each element with its indices and stores the result in a new tensor.
        /// </summary>
        public static Tensor<T> MapWithIndices<T>(this Tensor<T> tensor, Func<int[], T, T> func)
        {
            var result = new Tensor<T>(tensor.Shape);
            tensor.ForEach((indices, value) => result[indices] = func(indices, value));
            return result;
        }
        
        /// <summary>
        /// Sets all elements of the tensor using a function based on indices.
        /// </summary>
        public static void SetAll<T>(this Tensor<T> tensor, Func<int[], T> valueFunc)
        {
            var indices = new int[tensor.Shape.Length];
            SetAllRecursive(tensor, indices, 0, valueFunc);
        }
        
        /// <summary>
        /// Gets a flattened view of the tensor for iteration by index.
        /// </summary>
        public static T GetFlat<T>(this Tensor<T> tensor, int flatIndex)
        {
            var indices = GetMultiDimensionalIndices(flatIndex, tensor.Shape);
            return tensor[indices];
        }
        
        /// <summary>
        /// Sets a value using a flattened index.
        /// </summary>
        public static void SetFlat<T>(this Tensor<T> tensor, int flatIndex, T value)
        {
            var indices = GetMultiDimensionalIndices(flatIndex, tensor.Shape);
            tensor[indices] = value;
        }
        
        private static void ForEachRecursive<T>(Tensor<T> tensor, int[] indices, int dimension, Action<int[], T> action)
        {
            if (dimension == tensor.Shape.Length)
            {
                action(indices, tensor[indices]);
                return;
            }
            
            for (int i = 0; i < tensor.Shape[dimension]; i++)
            {
                indices[dimension] = i;
                ForEachRecursive(tensor, indices, dimension + 1, action);
            }
        }
        
        private static void SetAllRecursive<T>(Tensor<T> tensor, int[] indices, int dimension, Func<int[], T> valueFunc)
        {
            if (dimension == tensor.Shape.Length)
            {
                tensor[indices] = valueFunc(indices);
                return;
            }
            
            for (int i = 0; i < tensor.Shape[dimension]; i++)
            {
                indices[dimension] = i;
                SetAllRecursive(tensor, indices, dimension + 1, valueFunc);
            }
        }
        
        private static int[] GetMultiDimensionalIndices(int flatIndex, int[] shape)
        {
            var indices = new int[shape.Length];
            int remaining = flatIndex;
            
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                indices[i] = remaining % shape[i];
                remaining /= shape[i];
            }
            
            return indices;
        }
    }
}