using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Example demonstrating the INumericOperations pattern for generic numeric operations.
    /// </summary>
    public class NumericOperationsExample
    {
        /// <summary>
        /// Demonstrates how to use INumericOperations with a generic class.
        /// </summary>
        public static void Run()
        {
            Console.WriteLine("Numeric Operations Example");
            Console.WriteLine("=========================");
            
            // Example with double
            Console.WriteLine("\nVector Addition (double):");
            var doubleVector1 = new NumericVector<double>(new double[] { 1.5, 2.5, 3.5 });
            var doubleVector2 = new NumericVector<double>(new double[] { 0.5, 1.0, 1.5 });
            var doubleResult = doubleVector1.Add(doubleVector2);
            Console.WriteLine($"Vector1: {doubleVector1}");
            Console.WriteLine($"Vector2: {doubleVector2}");
            Console.WriteLine($"Result: {doubleResult}");
            
            // Example with float
            Console.WriteLine("\nVector Addition (float):");
            var floatVector1 = new NumericVector<float>(new float[] { 1.5f, 2.5f, 3.5f });
            var floatVector2 = new NumericVector<float>(new float[] { 0.5f, 1.0f, 1.5f });
            var floatResult = floatVector1.Add(floatVector2);
            Console.WriteLine($"Vector1: {floatVector1}");
            Console.WriteLine($"Vector2: {floatVector2}");
            Console.WriteLine($"Result: {floatResult}");
            
            // Example of scaling a vector
            Console.WriteLine("\nVector Scaling (double):");
            var scaledVector = doubleVector1.Scale(2.0);
            Console.WriteLine($"Original: {doubleVector1}");
            Console.WriteLine($"Scaled by 2.0: {scaledVector}");
            
            // Example of computing dot product
            Console.WriteLine("\nDot Product (double):");
            var dotProduct = doubleVector1.DotProduct(doubleVector2);
            Console.WriteLine($"Vector1: {doubleVector1}");
            Console.WriteLine($"Vector2: {doubleVector2}");
            Console.WriteLine($"Dot Product: {dotProduct}");
            
            // Example of computing magnitude
            Console.WriteLine("\nVector Magnitude:");
            var magnitude = doubleVector1.Magnitude();
            Console.WriteLine($"Vector: {doubleVector1}");
            Console.WriteLine($"Magnitude: {magnitude}");
        }
    }

    /// <summary>
    /// A simple vector class that works with any numeric type using INumericOperations.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    public class NumericVector<T>
    {
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
        
        /// <summary>
        /// The elements of the vector.
        /// </summary>
        private readonly T[] _elements;
        
        /// <summary>
        /// Gets the length of the vector.
        /// </summary>
        public int Length => _elements.Length;
        
        /// <summary>
        /// Initializes a new instance of the NumericVector class.
        /// </summary>
        /// <param name="elements">The elements of the vector.</param>
        public NumericVector(T[] elements)
        {
            _elements = elements ?? throw new ArgumentNullException(nameof(elements));
        }
        
        /// <summary>
        /// Adds another vector to this vector.
        /// </summary>
        /// <param name="other">The vector to add.</param>
        /// <returns>A new vector that is the sum of this vector and the other vector.</returns>
        /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
        public NumericVector<T> Add(NumericVector<T> other)
        {
            if (Length != other.Length)
                throw new ArgumentException("Vectors must have the same length.");
                
            var result = new T[Length];
            for (int i = 0; i < Length; i++)
            {
                result[i] = NumOps.Add(_elements[i], other._elements[i]);
            }
            
            return new NumericVector<T>(result);
        }
        
        /// <summary>
        /// Scales this vector by a scalar value.
        /// </summary>
        /// <param name="scalar">The scalar value to multiply by.</param>
        /// <returns>A new vector that is this vector scaled by the scalar value.</returns>
        public NumericVector<T> Scale(T scalar)
        {
            var result = new T[Length];
            for (int i = 0; i < Length; i++)
            {
                result[i] = NumOps.Multiply(_elements[i], scalar);
            }
            
            return new NumericVector<T>(result);
        }
        
        /// <summary>
        /// Computes the dot product of this vector and another vector.
        /// </summary>
        /// <param name="other">The other vector.</param>
        /// <returns>The dot product of the two vectors.</returns>
        /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
        public T DotProduct(NumericVector<T> other)
        {
            if (Length != other.Length)
                throw new ArgumentException("Vectors must have the same length.");
                
            var result = NumOps.Zero;
            for (int i = 0; i < Length; i++)
            {
                result = NumOps.Add(result, NumOps.Multiply(_elements[i], other._elements[i]));
            }
            
            return result;
        }
        
        /// <summary>
        /// Computes the magnitude (Euclidean norm) of this vector.
        /// </summary>
        /// <returns>The magnitude of this vector.</returns>
        public T Magnitude()
        {
            var sumOfSquares = NumOps.Zero;
            for (int i = 0; i < Length; i++)
            {
                sumOfSquares = NumOps.Add(sumOfSquares, NumOps.Square(_elements[i]));
            }
            
            return NumOps.Sqrt(sumOfSquares);
        }
        
        /// <summary>
        /// Returns a string representation of this vector.
        /// </summary>
        public override string ToString()
        {
            return "[" + string.Join(", ", _elements) + "]";
        }
        
        /// <summary>
        /// Gets the element at the specified index.
        /// </summary>
        public T this[int index] => _elements[index];
    }
}