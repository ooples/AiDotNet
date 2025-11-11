using System;
using System.Numerics;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.NestedLearning
{
    /// <summary>
    /// Hope architecture - a self-modifying recurrent variant for nested learning.
    /// Implements unbounded levels of in-context learning through self-referential optimization.
    /// Based on Google's Nested Learning paradigm.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class HopeNetwork<T> : NeuralNetworkBase<T>
        where T : struct, IFloatingPoint<T>, IPowerFunctions<T>, IExponentialFunctions<T>,
            IFloatingPointIeee754<T>, ITrigonometricFunctions<T>
    {
        private readonly int _hiddenDim;
        private readonly int _numCMSLevels;
        private readonly int _numRecurrentLayers;

        private ContinuumMemorySystemLayer<T>[] _cmsBlocks;
        private RecurrentLayer<T>[] _recurrentLayers;
        private DenseLayer<T>? _outputLayer;

        // Self-modification state
        private Tensor<T>? _metaState;
        private int _adaptationStep;

        /// <summary>
        /// Initializes a new Hope network.
        /// </summary>
        /// <param name="architecture">Neural network architecture configuration</param>
        /// <param name="optimizer">Optional optimizer</param>
        /// <param name="lossFunction">Optional loss function</param>
        /// <param name="hiddenDim">Hidden dimension size</param>
        /// <param name="numCMSLevels">Number of CMS frequency levels</param>
        /// <param name="numRecurrentLayers">Number of recurrent layers</param>
        public HopeNetwork(
            NeuralNetworkArchitecture<T> architecture,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            ILossFunction<T>? lossFunction = null,
            int hiddenDim = 256,
            int numCMSLevels = 3,
            int numRecurrentLayers = 2)
            : base(architecture, lossFunction, maxGradNorm: 1.0)
        {
            _hiddenDim = hiddenDim;
            _numCMSLevels = numCMSLevels;
            _numRecurrentLayers = numRecurrentLayers;
            _adaptationStep = 0;

            InitializeLayers();
        }

        /// <summary>
        /// Initializes the Hope network layers.
        /// </summary>
        protected override void InitializeLayers()
        {
            Layers.Clear();

            // Initialize CMS blocks (one per frequency level)
            _cmsBlocks = new ContinuumMemorySystemLayer<T>[_numCMSLevels];
            for (int i = 0; i < _numCMSLevels; i++)
            {
                _cmsBlocks[i] = new ContinuumMemorySystemLayer<T>(
                    inputShape: new[] { _hiddenDim },
                    memoryDim: _hiddenDim,
                    numFrequencyLevels: _numCMSLevels);

                Layers.Add(_cmsBlocks[i]);
            }

            // Initialize recurrent layers for temporal processing
            _recurrentLayers = new RecurrentLayer<T>[_numRecurrentLayers];
            for (int i = 0; i < _numRecurrentLayers; i++)
            {
                _recurrentLayers[i] = new RecurrentLayer<T>(
                    inputShape: new[] { _hiddenDim },
                    outputUnits: _hiddenDim,
                    activation: ActivationFunction.Tanh);

                Layers.Add(_recurrentLayers[i]);
            }

            // Initialize meta-state for self-modification
            _metaState = Tensor<T>.CreateFromArray(new T[_hiddenDim], new[] { _hiddenDim });
        }

        /// <summary>
        /// Forward pass through Hope network with self-referential optimization.
        /// </summary>
        public override Tensor<T> Forward(Tensor<T> input)
        {
            var current = input;

            // Self-modification: adjust processing based on meta-state
            if (_metaState != null)
            {
                current = ApplySelfModification(current, _metaState);
            }

            // Process through CMS blocks at different frequency levels
            Tensor<T>? cmsOutput = null;
            for (int i = 0; i < _numCMSLevels; i++)
            {
                var levelOutput = _cmsBlocks[i].Forward(current);

                if (cmsOutput == null)
                {
                    cmsOutput = levelOutput;
                }
                else
                {
                    cmsOutput = AddTensors(cmsOutput, levelOutput);
                }
            }

            current = cmsOutput!;

            // Process through recurrent layers
            foreach (var recurrentLayer in _recurrentLayers)
            {
                current = recurrentLayer.Forward(current);
            }

            // Update meta-state based on current processing
            UpdateMetaState(current);

            // If we have an output layer, use it
            if (_outputLayer != null)
            {
                current = _outputLayer.Forward(current);
            }

            _adaptationStep++;

            return current;
        }

        /// <summary>
        /// Backward pass through Hope network.
        /// </summary>
        public override Tensor<T> Backward(Tensor<T> outputGradient)
        {
            var gradient = outputGradient;

            // Backprop through output layer if present
            if (_outputLayer != null)
            {
                gradient = _outputLayer.Backward(gradient);
            }

            // Backprop through recurrent layers (in reverse)
            for (int i = _numRecurrentLayers - 1; i >= 0; i--)
            {
                gradient = _recurrentLayers[i].Backward(gradient);
            }

            // Backprop through CMS blocks
            Tensor<T>? totalGradient = null;
            for (int i = _numCMSLevels - 1; i >= 0; i--)
            {
                var cmsGrad = _cmsBlocks[i].Backward(gradient);

                if (totalGradient == null)
                {
                    totalGradient = cmsGrad;
                }
                else
                {
                    totalGradient = AddTensors(totalGradient, cmsGrad);
                }
            }

            return totalGradient!;
        }

        /// <summary>
        /// Applies self-modification to input based on meta-state.
        /// This implements the self-referential optimization aspect of Hope.
        /// </summary>
        private Tensor<T> ApplySelfModification(Tensor<T> input, Tensor<T> metaState)
        {
            var inputArray = input.ToArray();
            var metaArray = metaState.ToArray();

            int minLen = Math.Min(inputArray.Length, metaArray.Length);
            var modified = new T[inputArray.Length];

            for (int i = 0; i < inputArray.Length; i++)
            {
                if (i < minLen)
                {
                    // Modulate input with meta-state
                    T modulationFactor = T.One + metaArray[i] * T.CreateChecked(0.1);
                    modified[i] = inputArray[i] * modulationFactor;
                }
                else
                {
                    modified[i] = inputArray[i];
                }
            }

            return Tensor<T>.CreateFromArray(modified, input.Shape);
        }

        /// <summary>
        /// Updates the meta-state based on current processing.
        /// This enables the network to adapt its behavior over time.
        /// </summary>
        private void UpdateMetaState(Tensor<T> currentState)
        {
            if (_metaState == null) return;

            var currentArray = currentState.ToArray();
            var metaArray = _metaState.ToArray();

            int minLen = Math.Min(currentArray.Length, metaArray.Length);

            // Exponential moving average with slow adaptation
            T adaptationRate = T.CreateChecked(0.01);

            for (int i = 0; i < minLen; i++)
            {
                metaArray[i] = metaArray[i] * (T.One - adaptationRate) +
                              currentArray[i] * adaptationRate;
            }

            _metaState = Tensor<T>.CreateFromArray(metaArray, _metaState.Shape);
        }

        /// <summary>
        /// Performs memory consolidation across all CMS blocks.
        /// Should be called periodically during training.
        /// </summary>
        public void ConsolidateMemory()
        {
            foreach (var cmsBlock in _cmsBlocks)
            {
                cmsBlock.ConsolidateMemory();
            }
        }

        /// <summary>
        /// Resets all memory in CMS blocks and meta-state.
        /// </summary>
        public void ResetMemory()
        {
            foreach (var cmsBlock in _cmsBlocks)
            {
                cmsBlock.ResetMemory();
            }

            _metaState = Tensor<T>.CreateFromArray(new T[_hiddenDim], new[] { _hiddenDim });
            _adaptationStep = 0;
        }

        /// <summary>
        /// Resets recurrent layer states.
        /// </summary>
        public void ResetRecurrentState()
        {
            foreach (var recurrentLayer in _recurrentLayers)
            {
                recurrentLayer.ResetState();
            }
        }

        /// <summary>
        /// Adds an output layer to the Hope network.
        /// </summary>
        /// <param name="outputDim">Output dimension</param>
        /// <param name="activation">Activation function</param>
        public void AddOutputLayer(int outputDim, ActivationFunction activation = ActivationFunction.Linear)
        {
            _outputLayer = new DenseLayer<T>(
                inputShape: new[] { _hiddenDim },
                outputUnits: outputDim,
                activation: activation);

            Layers.Add(_outputLayer);
        }

        /// <summary>
        /// Gets the current meta-state (for inspection/debugging).
        /// </summary>
        public Tensor<T>? GetMetaState() => _metaState;

        /// <summary>
        /// Gets the number of adaptation steps performed.
        /// </summary>
        public int AdaptationStep => _adaptationStep;

        /// <summary>
        /// Gets the CMS blocks (for inspection/debugging).
        /// </summary>
        public ContinuumMemorySystemLayer<T>[] GetCMSBlocks() => _cmsBlocks;

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
    }
}
