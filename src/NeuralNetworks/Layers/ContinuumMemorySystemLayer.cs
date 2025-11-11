using System;
using System.Numerics;
using AiDotNet.Interfaces;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers
{
    /// <summary>
    /// Continuum Memory System (CMS) layer - implements a spectrum of memory modules
    /// operating at different frequencies for nested learning.
    /// Based on Google's Nested Learning paradigm.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class ContinuumMemorySystemLayer<T> : LayerBase<T>
        where T : struct, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>,
            IFloatingPointIeee754<T>, ITrigonometricFunctions<T>
    {
        private readonly int _memoryDim;
        private readonly int _numFrequencyLevels;
        private readonly T[] _decayRates;
        private Tensor<T>[] _memoryStates;
        private Tensor<T>?[] _queryCache;
        private Tensor<T>?[] _keyCache;
        private Tensor<T>?[] _valueCache;

        // Learnable projection matrices for each frequency level
        private Matrix<T>[] _queryProjections;
        private Matrix<T>[] _keyProjections;
        private Matrix<T>[] _valueProjections;
        private Matrix<T>[] _outputProjections;

        // Gradients
        private Matrix<T>[]? _queryProjectionsGrad;
        private Matrix<T>[]? _keyProjectionsGrad;
        private Matrix<T>[]? _valueProjectionsGrad;
        private Matrix<T>[]? _outputProjectionsGrad;

        /// <summary>
        /// Initializes a new instance of the ContinuumMemorySystemLayer.
        /// </summary>
        /// <param name="inputShape">Shape of input tensor</param>
        /// <param name="memoryDim">Dimension of memory representations</param>
        /// <param name="numFrequencyLevels">Number of frequency levels in the continuum</param>
        /// <param name="decayRates">Decay rate for each frequency level (optional)</param>
        public ContinuumMemorySystemLayer(
            int[] inputShape,
            int memoryDim,
            int numFrequencyLevels = 3,
            T[]? decayRates = null)
            : base(inputShape, new[] { memoryDim }, null, null)
        {
            _memoryDim = memoryDim;
            _numFrequencyLevels = numFrequencyLevels;

            // Initialize decay rates - higher levels decay slower
            _decayRates = decayRates ?? CreateDefaultDecayRates(numFrequencyLevels);

            // Initialize memory states for each frequency level
            _memoryStates = new Tensor<T>[numFrequencyLevels];
            _queryCache = new Tensor<T>?[numFrequencyLevels];
            _keyCache = new Tensor<T>?[numFrequencyLevels];
            _valueCache = new Tensor<T>?[numFrequencyLevels];

            int inputDim = inputShape[0];

            // Initialize projection matrices for each level
            _queryProjections = new Matrix<T>[numFrequencyLevels];
            _keyProjections = new Matrix<T>[numFrequencyLevels];
            _valueProjections = new Matrix<T>[numFrequencyLevels];
            _outputProjections = new Matrix<T>[numFrequencyLevels];

            for (int i = 0; i < numFrequencyLevels; i++)
            {
                // Initialize memory states with zeros
                _memoryStates[i] = Tensor<T>.CreateFromArray(new T[memoryDim], new[] { memoryDim });

                // Xavier initialization for projection matrices
                _queryProjections[i] = CreateMatrix.Random<T>(memoryDim, inputDim, new Random());
                _keyProjections[i] = CreateMatrix.Random<T>(memoryDim, inputDim, new Random());
                _valueProjections[i] = CreateMatrix.Random<T>(memoryDim, inputDim, new Random());
                _outputProjections[i] = CreateMatrix.Random<T>(inputDim, memoryDim, new Random());

                // Scale by Xavier initialization factor
                T scale = T.CreateChecked(T.Sqrt(T.CreateChecked(2.0) / T.CreateChecked(inputDim + memoryDim)));
                _queryProjections[i] *= scale;
                _keyProjections[i] *= scale;
                _valueProjections[i] *= scale;
                _outputProjections[i] *= scale;
            }

            InitializeParameters();
        }

        private T[] CreateDefaultDecayRates(int numLevels)
        {
            var rates = new T[numLevels];
            for (int i = 0; i < numLevels; i++)
            {
                // Exponentially decreasing decay rates: level 0 decays fastest, highest level slowest
                double rate = Math.Pow(0.5, i + 1); // 0.5, 0.25, 0.125, ...
                rates[i] = T.CreateChecked(rate);
            }
            return rates;
        }

        private void InitializeParameters()
        {
            // Flatten all projection matrices into parameter vector
            int totalParams = 4 * _numFrequencyLevels * _memoryDim * GetInputShape()[0];
            Parameters = Vector<T>.Build.Dense(totalParams);

            int offset = 0;
            for (int i = 0; i < _numFrequencyLevels; i++)
            {
                CopyMatrixToVector(_queryProjections[i], Parameters, ref offset);
                CopyMatrixToVector(_keyProjections[i], Parameters, ref offset);
                CopyMatrixToVector(_valueProjections[i], Parameters, ref offset);
                CopyMatrixToVector(_outputProjections[i], Parameters, ref offset);
            }
        }

        private void CopyMatrixToVector(Matrix<T> matrix, Vector<T> vector, ref int offset)
        {
            for (int i = 0; i < matrix.RowCount; i++)
            {
                for (int j = 0; j < matrix.ColumnCount; j++)
                {
                    vector[offset++] = matrix[i, j];
                }
            }
        }

        /// <summary>
        /// Forward pass through the CMS layer.
        /// </summary>
        public override Tensor<T> Forward(Tensor<T> input)
        {
            LastInput = input;

            // Aggregate outputs from all frequency levels
            Tensor<T>? output = null;

            for (int level = 0; level < _numFrequencyLevels; level++)
            {
                // Project input to query, key, value spaces
                var query = ProjectToQuerySpace(input, level);
                var key = ProjectToKeySpace(input, level);
                var value = ProjectToValueSpace(input, level);

                _queryCache[level] = query;
                _keyCache[level] = key;
                _valueCache[level] = value;

                // Compute attention scores with memory
                var scores = ComputeAttentionScores(query, _memoryStates[level]);

                // Retrieve from memory using attention
                var retrieved = ApplyAttention(scores, _memoryStates[level]);

                // Update memory state with decay
                UpdateMemoryState(level, value);

                // Project back to output space
                var levelOutput = ProjectToOutputSpace(retrieved, level);

                if (output == null)
                {
                    output = levelOutput;
                }
                else
                {
                    output = AddTensors(output, levelOutput);
                }
            }

            LastOutput = output!;
            return LastOutput;
        }

        private Tensor<T> ProjectToQuerySpace(Tensor<T> input, int level)
        {
            var inputVec = TensorToVector(input);
            var projected = _queryProjections[level] * inputVec;
            return VectorToTensor(projected);
        }

        private Tensor<T> ProjectToKeySpace(Tensor<T> input, int level)
        {
            var inputVec = TensorToVector(input);
            var projected = _keyProjections[level] * inputVec;
            return VectorToTensor(projected);
        }

        private Tensor<T> ProjectToValueSpace(Tensor<T> input, int level)
        {
            var inputVec = TensorToVector(input);
            var projected = _valueProjections[level] * inputVec;
            return VectorToTensor(projected);
        }

        private Tensor<T> ProjectToOutputSpace(Tensor<T> memory, int level)
        {
            var memVec = TensorToVector(memory);
            var projected = _outputProjections[level] * memVec;
            return VectorToTensor(projected);
        }

        private Tensor<T> ComputeAttentionScores(Tensor<T> query, Tensor<T> memory)
        {
            // Simplified dot-product attention
            var queryVec = TensorToVector(query);
            var memoryVec = TensorToVector(memory);

            T score = queryVec.DotProduct(memoryVec);
            T sqrtDim = T.Sqrt(T.CreateChecked(_memoryDim));
            score /= sqrtDim; // Scale by sqrt(dim)

            // Apply softmax (simplified for single memory state)
            T expScore = T.Exp(score);
            return Tensor<T>.CreateFromArray(new[] { expScore }, new[] { 1 });
        }

        private Tensor<T> ApplyAttention(Tensor<T> scores, Tensor<T> memory)
        {
            // With single memory state, attention just scales the memory
            T score = scores.ToArray()[0];
            var memVec = TensorToVector(memory);
            var result = memVec * score;
            return VectorToTensor(result);
        }

        private void UpdateMemoryState(int level, Tensor<T> newValue)
        {
            // Exponential moving average with level-specific decay
            T decay = _decayRates[level];
            T oneMinusDecay = T.One - decay;

            var currentMemVec = TensorToVector(_memoryStates[level]);
            var newValVec = TensorToVector(newValue);

            // memory = decay * memory + (1 - decay) * newValue
            var updated = currentMemVec * decay + newValVec * oneMinusDecay;
            _memoryStates[level] = VectorToTensor(updated);
        }

        /// <summary>
        /// Backward pass through the CMS layer.
        /// </summary>
        public override Tensor<T> Backward(Tensor<T> outputGradient)
        {
            // Initialize gradient accumulators
            _queryProjectionsGrad = new Matrix<T>[_numFrequencyLevels];
            _keyProjectionsGrad = new Matrix<T>[_numFrequencyLevels];
            _valueProjectionsGrad = new Matrix<T>[_numFrequencyLevels];
            _outputProjectionsGrad = new Matrix<T>[_numFrequencyLevels];

            Tensor<T>? inputGradient = null;

            for (int level = 0; level < _numFrequencyLevels; level++)
            {
                // Backprop through output projection
                var outputProjGrad = ComputeOutputProjectionGradient(outputGradient, level);
                _outputProjectionsGrad[level] = outputProjGrad;

                // Backprop through attention and memory operations
                var levelInputGrad = ComputeLevelInputGradient(outputGradient, level);

                if (inputGradient == null)
                {
                    inputGradient = levelInputGrad;
                }
                else
                {
                    inputGradient = AddTensors(inputGradient, levelInputGrad);
                }

                // Compute projection gradients
                _queryProjectionsGrad[level] = ComputeProjectionGradient(_queryCache[level]!, LastInput!);
                _keyProjectionsGrad[level] = ComputeProjectionGradient(_keyCache[level]!, LastInput!);
                _valueProjectionsGrad[level] = ComputeProjectionGradient(_valueCache[level]!, LastInput!);
            }

            // Update parameter gradients
            UpdateParameterGradients();

            return inputGradient!;
        }

        private Matrix<T> ComputeOutputProjectionGradient(Tensor<T> outputGrad, int level)
        {
            var outGradVec = TensorToVector(outputGrad);
            var memoryVec = TensorToVector(_memoryStates[level]);
            return Matrix<T>.Build.DenseOfColumnVectors(outGradVec) * Matrix<T>.Build.DenseOfRowVectors(memoryVec);
        }

        private Tensor<T> ComputeLevelInputGradient(Tensor<T> outputGrad, int level)
        {
            var outGradVec = TensorToVector(outputGrad);
            var inputGrad = _outputProjections[level].Transpose() * outGradVec;
            return VectorToTensor(inputGrad);
        }

        private Matrix<T> ComputeProjectionGradient(Tensor<T> projected, Tensor<T> input)
        {
            var projVec = TensorToVector(projected);
            var inputVec = TensorToVector(input);
            return Matrix<T>.Build.DenseOfColumnVectors(projVec) * Matrix<T>.Build.DenseOfRowVectors(inputVec);
        }

        private void UpdateParameterGradients()
        {
            int totalParams = Parameters.Count;
            ParameterGradients = Vector<T>.Build.Dense(totalParams);

            int offset = 0;
            for (int i = 0; i < _numFrequencyLevels; i++)
            {
                CopyMatrixToVector(_queryProjectionsGrad![i], ParameterGradients, ref offset);
                CopyMatrixToVector(_keyProjectionsGrad![i], ParameterGradients, ref offset);
                CopyMatrixToVector(_valueProjectionsGrad![i], ParameterGradients, ref offset);
                CopyMatrixToVector(_outputProjectionsGrad![i], ParameterGradients, ref offset);
            }
        }

        public override void UpdateParameters(Vector<T> updates)
        {
            Parameters -= updates;

            // Update matrices from parameter vector
            int offset = 0;
            for (int i = 0; i < _numFrequencyLevels; i++)
            {
                CopyVectorToMatrix(Parameters, _queryProjections[i], ref offset);
                CopyVectorToMatrix(Parameters, _keyProjections[i], ref offset);
                CopyVectorToMatrix(Parameters, _valueProjections[i], ref offset);
                CopyVectorToMatrix(Parameters, _outputProjections[i], ref offset);
            }
        }

        private void CopyVectorToMatrix(Vector<T> vector, Matrix<T> matrix, ref int offset)
        {
            for (int i = 0; i < matrix.RowCount; i++)
            {
                for (int j = 0; j < matrix.ColumnCount; j++)
                {
                    matrix[i, j] = vector[offset++];
                }
            }
        }

        private Vector<T> TensorToVector(Tensor<T> tensor)
        {
            return Vector<T>.Build.DenseOfArray(tensor.ToArray());
        }

        private Tensor<T> VectorToTensor(Vector<T> vector)
        {
            return Tensor<T>.CreateFromArray(vector.ToArray(), new[] { vector.Count });
        }

        private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
        {
            var arrA = a.ToArray();
            var arrB = b.ToArray();
            var result = new T[arrA.Length];
            for (int i = 0; i < arrA.Length; i++)
            {
                result[i] = arrA[i] + arrB[i];
            }
            return Tensor<T>.CreateFromArray(result, a.Shape);
        }

        /// <summary>
        /// Gets the memory states for diagnostic purposes.
        /// </summary>
        public Tensor<T>[] GetMemoryStates() => _memoryStates;

        /// <summary>
        /// Resets all memory states to zero.
        /// </summary>
        public void ResetMemory()
        {
            for (int i = 0; i < _numFrequencyLevels; i++)
            {
                _memoryStates[i] = Tensor<T>.CreateFromArray(new T[_memoryDim], new[] { _memoryDim });
            }
        }

        /// <summary>
        /// Consolidates memories across frequency levels (mimics biological memory consolidation).
        /// </summary>
        public void ConsolidateMemory()
        {
            // Transfer information from faster to slower frequency levels
            for (int i = 0; i < _numFrequencyLevels - 1; i++)
            {
                var fastMem = TensorToVector(_memoryStates[i]);
                var slowMem = TensorToVector(_memoryStates[i + 1]);

                // Blend fast memory into slow memory
                T blendFactor = T.CreateChecked(0.1); // 10% transfer rate
                var blended = slowMem * (T.One - blendFactor) + fastMem * blendFactor;

                _memoryStates[i + 1] = VectorToTensor(blended);
            }
        }
    }
}
