using System;
using System.Numerics;
using AiDotNet.Interfaces;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.NestedLearning
{
    /// <summary>
    /// Implementation of Context Flow mechanism for nested learning.
    /// Maintains distinct information pathways and update rates for each optimization level.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class ContextFlow<T> : IContextFlow<T>
        where T : struct, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>
    {
        private readonly int _numLevels;
        private readonly int _contextDimension;
        private Tensor<T>[] _contextStates;
        private Matrix<T>[] _transformationMatrices;
        private readonly Random _random;

        /// <summary>
        /// Initializes a new Context Flow mechanism.
        /// </summary>
        /// <param name="contextDimension">Dimension of context vectors</param>
        /// <param name="numLevels">Number of nested optimization levels</param>
        /// <param name="seed">Random seed for initialization</param>
        public ContextFlow(int contextDimension, int numLevels = 3, int? seed = null)
        {
            _contextDimension = contextDimension;
            _numLevels = numLevels;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();

            // Initialize context states
            _contextStates = new Tensor<T>[numLevels];
            _transformationMatrices = new Matrix<T>[numLevels];

            for (int i = 0; i < numLevels; i++)
            {
                // Initialize context states to zero
                _contextStates[i] = Tensor<T>.CreateFromArray(
                    new T[contextDimension],
                    new[] { contextDimension });

                // Initialize transformation matrices with Xavier initialization
                _transformationMatrices[i] = CreateMatrix.Random<T>(
                    contextDimension, contextDimension, _random);

                T scale = T.Sqrt(T.CreateChecked(2.0 / contextDimension));
                _transformationMatrices[i] *= scale;
            }
        }

        /// <inheritdoc/>
        public Tensor<T> PropagateContext(Tensor<T> input, int currentLevel)
        {
            if (currentLevel < 0 || currentLevel >= _numLevels)
            {
                throw new ArgumentException($"Invalid level: {currentLevel}");
            }

            // Transform input through level-specific transformation
            var inputVector = Vector<T>.Build.DenseOfArray(input.ToArray());
            var transformed = _transformationMatrices[currentLevel] * inputVector;

            // Update context state (exponential moving average)
            var currentContext = Vector<T>.Build.DenseOfArray(_contextStates[currentLevel].ToArray());
            T momentum = T.CreateChecked(0.9); // Momentum factor

            var newContext = currentContext * momentum + transformed * (T.One - momentum);

            _contextStates[currentLevel] = Tensor<T>.CreateFromArray(
                newContext.ToArray(),
                new[] { _contextDimension });

            return _contextStates[currentLevel];
        }

        /// <inheritdoc/>
        public Tensor<T> ComputeContextGradients(Tensor<T> upstreamGradient, int level)
        {
            if (level < 0 || level >= _numLevels)
            {
                throw new ArgumentException($"Invalid level: {level}");
            }

            // Backpropagate through transformation matrix
            var gradVector = Vector<T>.Build.DenseOfArray(upstreamGradient.ToArray());
            var contextGrad = _transformationMatrices[level].Transpose() * gradVector;

            return Tensor<T>.CreateFromArray(
                contextGrad.ToArray(),
                new[] { _contextDimension });
        }

        /// <inheritdoc/>
        public void UpdateFlow(Tensor<T>[] gradients, T[] learningRates)
        {
            if (gradients.Length != _numLevels)
            {
                throw new ArgumentException(
                    $"Number of gradients ({gradients.Length}) must match number of levels ({_numLevels})");
            }

            if (learningRates.Length != _numLevels)
            {
                throw new ArgumentException(
                    $"Number of learning rates ({learningRates.Length}) must match number of levels ({_numLevels})");
            }

            for (int i = 0; i < _numLevels; i++)
            {
                var gradVector = Vector<T>.Build.DenseOfArray(gradients[i].ToArray());
                var contextVector = Vector<T>.Build.DenseOfArray(_contextStates[i].ToArray());

                // Compute gradient for transformation matrix (outer product)
                var matrixGrad = Matrix<T>.Build.DenseOfColumnVectors(gradVector) *
                                Matrix<T>.Build.DenseOfRowVectors(contextVector);

                // Update transformation matrix
                _transformationMatrices[i] -= learningRates[i] * matrixGrad;
            }
        }

        /// <inheritdoc/>
        public Tensor<T> GetContextState(int level)
        {
            if (level < 0 || level >= _numLevels)
            {
                throw new ArgumentException($"Invalid level: {level}");
            }

            return _contextStates[level];
        }

        /// <inheritdoc/>
        public void Reset()
        {
            for (int i = 0; i < _numLevels; i++)
            {
                _contextStates[i] = Tensor<T>.CreateFromArray(
                    new T[_contextDimension],
                    new[] { _contextDimension });
            }
        }

        /// <inheritdoc/>
        public int NumberOfLevels => _numLevels;

        /// <summary>
        /// Gets the transformation matrices for each level (for inspection/debugging).
        /// </summary>
        public Matrix<T>[] GetTransformationMatrices() => _transformationMatrices;
    }
}
