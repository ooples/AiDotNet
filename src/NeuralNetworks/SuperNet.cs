using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Attributes;
using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Validation;

using AiDotNet.Models.Parameters;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// SuperNet implementation for gradient-based neural architecture search (DARTS).
    /// Implements a differentiable architecture search by maintaining architecture parameters (alpha)
    /// and network weights simultaneously.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    /// <remarks>
    /// <para><b>For Beginners:</b> A SuperNet is a "network of all possible networks." It
    /// contains every candidate architecture within a single large network, with learnable
    /// weights that determine which operations are most important. During architecture search,
    /// the SuperNet trains these weights using gradient descent, and the final architecture
    /// is derived by selecting the operations with the highest weights. This is the core
    /// mechanism behind DARTS-style neural architecture search.</para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var searchSpace = new SearchSpaceBase&lt;float&gt;();
    /// var superNet = new SuperNet&lt;float&gt;(searchSpace, numNodes: 4, inputSize: 784, outputSize: 10);
    /// superNet.ForwardPass(inputTensor);
    /// var architecture = superNet.DeriveArchitecture();
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.General)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelTask(ModelTask.Classification)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Understanding and Simplifying One-Shot Architecture Search", "https://arxiv.org/abs/1810.03522")]
    public partial class SuperNet<T> : ModelBase<T, Tensor<T>, Tensor<T>>
    {
        private readonly SearchSpaceBase<T> _searchSpace;
        private readonly int _numNodes;
        private readonly int _numOperations;
        private readonly Random _random; // Shared Random instance to avoid time-based seeding issues

        // Architecture parameters (alpha) - learnable parameters that determine operation weights
        [TrainableParameter]
        private readonly List<Matrix<T>> _architectureParams;

        // Network weights - parameters for each operation
        [TrainableParameter(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Fit)]
        private readonly Dictionary<string, Vector<T>> _weights;

        // Gradients
        [Scratch]
        private readonly List<Matrix<T>> _architectureGradients;
        [Scratch]
        private readonly Dictionary<string, Vector<T>> _weightGradients;

        // Model metadata
        private int _inputSize;
        private int _outputSize;

        // IInterpretableModel fields
        private readonly HashSet<InterpretationMethod> _enabledMethods = new();
        [Buffer]
        private Vector<int>? _sensitiveFeatures;
        private readonly List<FairnessMetric> _fairnessMetrics = new();
        private IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>>? _baseModel;

        /// <summary>
        /// The default loss function used by this model for gradient computation.
        /// </summary>
        private readonly ILossFunction<T> _defaultLossFunction;

        // The candidate operations, one per search-space entry and in its order: the name as the search
        // space spells it, and what that name denotes on this cell's feature axis.
        private readonly string[] _operationNames;
        private readonly Operation[] _operations;

        // The shape each operation weight takes on the tape, keyed like _weights.
        private readonly Dictionary<string, int[]> _weightShapes = new Dictionary<string, int[]>();

        // Zero-copy tensor views of the architecture matrices and weight vectors, one per storage object.
        // The optimizers key their state (Adam moments, momentum velocity) by tensor identity, so a view
        // rebuilt every step would silently restart them; a restored matrix or vector gets a fresh view.
        private readonly ConditionalWeakTable<Matrix<T>, Tensor<T>> _architectureViews = new ConditionalWeakTable<Matrix<T>, Tensor<T>>();
        private readonly ConditionalWeakTable<Vector<T>, Tensor<T>> _weightViews = new ConditionalWeakTable<Vector<T>, Tensor<T>>();

        // Liu et al. 2019, appendix A.1.1: Adam for the architecture (3e-4, betas (0.5, 0.999), weight decay
        // 1e-3) and momentum SGD for the weights (0.025, momentum 0.9, weight decay 3e-4).
        private const double ArchitectureLearningRate = 3e-4;
        private const double ArchitectureWeightDecay = 1e-3;
        private const double WeightLearningRate = 0.025;
        private const double WeightMomentum = 0.9;
        private const double WeightDecay = 3e-4;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _architectureOptimizer;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _weightOptimizer;

        private static readonly string[] DefaultOperationNames = { "identity", "conv3x3", "conv5x5", "maxpool3x3", "avgpool3x3" };


        public string[] FeatureNames { get; set; } = Array.Empty<string>();
        /// <summary>
        /// Gets the default loss function used by this model for gradient computation.
        /// </summary>
        /// <remarks>
        /// <para>
        /// For SuperNet (Neural Architecture Search), the default loss function is Mean Squared Error (MSE),
        /// which is used for computing both architecture and weight gradients.
        /// </para>
        /// </remarks>
        public override ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

        /// <summary>
        /// Initializes a new SuperNet for differentiable architecture search.
        /// </summary>
        /// <param name="searchSpace">The search space defining available operations</param>
        /// <param name="numNodes">Number of nodes in the architecture</param>
        /// <param name="lossFunction">Optional loss function to use for training. If null, uses Mean Squared Error (MSE) for neural architecture search.</param>
        public SuperNet(SearchSpaceBase<T> searchSpace, int numNodes = 4, ILossFunction<T>? lossFunction = null)
        {
            _searchSpace = searchSpace ?? throw new ArgumentNullException(nameof(searchSpace));
            if (numNodes <= 0)
                throw new ArgumentOutOfRangeException(nameof(numNodes), numNodes, "A SuperNet cell needs at least one node.");

            _numNodes = numNodes;
            _operationNames = (searchSpace.Operations is { Count: > 0 } names ? names : (IList<string>)DefaultOperationNames).ToArray();
            _operations = _operationNames.Select(ParseOperation).ToArray();
            _numOperations = _operations.Length;
            _random = RandomHelper.CreateSeededRandom(42); // Initialize with seed for reproducibility

            // Architecture parameters start at zero, "which implies equal amount of attention (after taking
            // the softmax) over all possible ops" (Liu et al. 2019, appendix A.1.1). Node j mixes every earlier
            // node, the input included, so its matrix has j + 1 rows.
            _architectureParams = new List<Matrix<T>>();
            _architectureGradients = new List<Matrix<T>>();
            for (int i = 0; i < _numNodes; i++)
            {
                _architectureParams.Add(new Matrix<T>(i + 1, _numOperations));
                _architectureGradients.Add(new Matrix<T>(i + 1, _numOperations));
            }

            // Every operation weight exists from construction: none depends on the input width, so the
            // parameter count is fixed before any forward pass.
            _weights = new Dictionary<string, Vector<T>>();
            _weightGradients = new Dictionary<string, Vector<T>>();
            ReconcileWeightsWithPlan();

            // Initialize default loss function (MSE for SuperNet)
            _defaultLossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        }

        /// <summary>
        /// Forward pass through the SuperNet with mixed operations
        /// </summary>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            var rows = ToRows(input, out int[] originalShape);
            _inputSize = rows.Shape[1];
            _outputSize = _inputSize;

            var result = Forward(rows);
            return originalShape.Length == 2 ? result : result.Reshape(originalShape);
        }

        /// <summary>
        /// One step of first-order differentiable architecture search on a single batch.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Liu et al. 2019, Algorithm 1 with xi = 0: descend the loss with respect to the architecture,
        /// then with respect to the weights. The paper takes the first on held-out validation data;
        /// Train receives one batch, so both steps use it. Use <see cref="TrainStep"/> to keep the split.
        /// </para>
        /// <para>
        /// This used to throw NotSupportedException, so a SuperNet returned by a NAS search could not be
        /// trained by the model builder that received it.
        /// </para>
        /// </remarks>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            TrainStep(input, expectedOutput, input, expectedOutput);
        }

        /// <summary>
        /// One step of first-order DARTS: the architecture on the validation batch, then the weights on the
        /// training batch (Liu et al. 2019, Algorithm 1 with xi = 0).
        /// </summary>
        /// <returns>The validation loss the architecture step descended and the training loss the weight
        /// step descended, each including its weight-decay term.</returns>
        public (T ArchitectureLoss, T WeightLoss) TrainStep(
            Tensor<T> trainInput, Tensor<T> trainTarget, Tensor<T> validationInput, Tensor<T> validationTarget)
        {
            if (trainInput is null) throw new ArgumentNullException(nameof(trainInput));
            if (trainTarget is null) throw new ArgumentNullException(nameof(trainTarget));
            if (validationInput is null) throw new ArgumentNullException(nameof(validationInput));
            if (validationTarget is null) throw new ArgumentNullException(nameof(validationTarget));

            T architectureLoss = StepOnTape(
                validationInput, validationTarget, ArchitectureSources(), ArchitectureWeightDecay, ArchitectureOptimizer);
            T weightLoss = StepOnTape(trainInput, trainTarget, WeightSources(), WeightDecay, WeightOptimizer);
            return (architectureLoss, weightLoss);
        }

        /// <summary>
        /// Computes validation loss for architecture parameter updates
        /// </summary>
        public T ComputeValidationLoss(Tensor<T> valData, Tensor<T> valLabels)
            => ScalarValue(TaskLoss(valData, valLabels, _defaultLossFunction));

        /// <summary>
        /// Computes training loss for weight updates
        /// </summary>
        public T ComputeTrainingLoss(Tensor<T> trainData, Tensor<T> trainLabels)
            => ScalarValue(TaskLoss(trainData, trainLabels, _defaultLossFunction));

        /// <summary>
        /// Computes the exact gradient of the loss with respect to the architecture parameters, into
        /// <see cref="GetArchitectureGradients"/>.
        /// </summary>
        /// <remarks>
        /// Taken on the tape. It used to be a central finite difference: two forward passes for every
        /// architecture scalar, and an approximation.
        /// </remarks>
        public void BackwardArchitecture(Tensor<T> input, Tensor<T> target)
        {
            var sources = ArchitectureSources();
            Dictionary<Tensor<T>, Tensor<T>> gradients;
            using (var tape = new GradientTape<T>())
            {
                gradients = tape.ComputeGradients(TaskLoss(input, target, _defaultLossFunction), sources, false);
            }

            for (int node = 0; node < _architectureParams.Count; node++)
            {
                var destination = _architectureGradients[node];
                gradients.TryGetValue(sources[node], out var gradient);
                for (int r = 0; r < destination.Rows; r++)
                {
                    for (int c = 0; c < destination.Columns; c++)
                    {
                        destination[r, c] = gradient is null ? NumOps.Zero : gradient[r, c];
                    }
                }
            }
        }

        /// <summary>
        /// Computes the exact gradient of <paramref name="lossFunction"/> with respect to the operation
        /// weights, into <see cref="GetWeightGradients"/>.
        /// </summary>
        public void BackwardWeights(Tensor<T> input, Tensor<T> target, ILossFunction<T> lossFunction)
        {
            if (lossFunction is null) throw new ArgumentNullException(nameof(lossFunction));

            var keys = _weights.Keys.ToList();
            var sources = keys.Select(WeightView).ToList();
            Dictionary<Tensor<T>, Tensor<T>> gradients;
            using (var tape = new GradientTape<T>())
            {
                gradients = tape.ComputeGradients(TaskLoss(input, target, lossFunction), sources, false);
            }

            for (int i = 0; i < keys.Count; i++)
            {
                var destination = _weightGradients[keys[i]];
                gradients.TryGetValue(sources[i], out var gradient);
                for (int e = 0; e < destination.Length; e++)
                {
                    destination[e] = gradient is null ? NumOps.Zero : gradient.GetFlat(e);
                }
            }
        }

        /// <summary>
        /// The cell's forward pass, shared by <see cref="Predict"/> and every training entry point: node j
        /// is the softmax(alpha_j)-weighted sum of every candidate operation applied to every earlier node
        /// (Liu et al. 2019, eqs. 1 and 2), and the cell's output is its last node.
        /// </summary>
        private Tensor<T> Forward(Tensor<T> rows)
        {
            int batch = rows.Shape[0];
            int features = rows.Shape[1];
            var nodes = new List<Tensor<T>>(_numNodes + 1) { rows };

            for (int node = 0; node < _numNodes; node++)
            {
                var mixture = Engine.Softmax(ArchitectureView(node), axis: 1);
                Tensor<T>? output = null;
                for (int from = 0; from <= node; from++)
                {
                    for (int op = 0; op < _numOperations; op++)
                    {
                        // The zero operation takes its share of the softmax and contributes nothing.
                        if (_operations[op].Kind == OperationKind.Zero) continue;

                        var weight = Engine.TensorTile(
                            Engine.TensorSlice(mixture, new[] { from, op }, new[] { 1, 1 }), new[] { batch, features });
                        var term = Engine.TensorMultiply(weight, ApplyOperation(nodes[from], node, from, op));
                        output = output is null ? term : Engine.TensorAdd(output, term);
                    }
                }

                nodes.Add(output ?? new Tensor<T>(new[] { batch, features }));
            }

            return nodes[nodes.Count - 1];
        }

        /// <summary>Flattens any input to [batch, features], remembering the caller's shape.</summary>
        private static Tensor<T> ToRows(Tensor<T> input, out int[] originalShape)
        {
            if (input is null) throw new ArgumentNullException(nameof(input));

            originalShape = input.Shape.ToArray();
            if (input.Shape.Length == 1)
                return input.Reshape(new[] { 1, input.Shape[0] });
            if (input.Shape.Length == 2)
                return input;

            int features = input.Shape[input.Shape.Length - 1];
            return input.Reshape(new[] { input.Length / features, features });
        }

        private Tensor<T> TaskLoss(Tensor<T> input, Tensor<T> target, ILossFunction<T> lossFunction)
            => lossFunction.ComputeTapeLoss(Forward(ToRows(input, out _)), ToRows(target, out _));

        private T ScalarValue(Tensor<T> tensor) => tensor.Length > 0 ? tensor.GetFlat(0) : NumOps.Zero;

        /// <summary>
        /// One optimizer step on the tape: the task loss plus coupled weight decay (lambda / 2) * sum(theta^2),
        /// whose gradient lambda * theta is exactly PyTorch's weight_decay for both Adam and SGD.
        /// </summary>
        private T StepOnTape(
            Tensor<T> input, Tensor<T> target, IReadOnlyList<Tensor<T>> sources, double weightDecay,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer)
        {
            Dictionary<Tensor<T>, Tensor<T>> gradients;
            Tensor<T> loss;
            using (var tape = new GradientTape<T>())
            {
                loss = Engine.Reshape(TaskLoss(input, target, _defaultLossFunction), new[] { 1 });
                foreach (var parameter in sources)
                {
                    var squares = Engine.ReduceSum(Engine.TensorMultiply(parameter, parameter), null, keepDims: false);
                    loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(
                        Engine.Reshape(squares, new[] { 1 }), NumOps.FromDouble(weightDecay / 2.0)));
                }

                gradients = tape.ComputeGradients(loss, sources, false);
            }

            T lossValue = ScalarValue(loss);
            optimizer.Step(new TapeStepContext<T>(sources, gradients, lossValue));
            return lossValue;
        }

        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> ArchitectureOptimizer
            => _architectureOptimizer ??= new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = ArchitectureLearningRate,
                    Beta1 = 0.5,
                    Beta2 = 0.999,
                });

        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> WeightOptimizer
            => _weightOptimizer ??= new MomentumOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new MomentumOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = WeightLearningRate,
                    InitialMomentum = WeightMomentum,
                });

        private List<Tensor<T>> ArchitectureSources()
            => Enumerable.Range(0, _architectureParams.Count).Select(ArchitectureView).ToList();

        private List<Tensor<T>> WeightSources() => _weights.Keys.Select(WeightView).ToList();

        private Tensor<T> ArchitectureView(int node)
            => _architectureViews.GetValue(_architectureParams[node],
                matrix => Tensor<T>.FromMemory(matrix.AsWritableMemory(), new[] { matrix.Rows, matrix.Columns }));

        private Tensor<T> WeightView(string key)
        {
            int[] shape = _weightShapes[key];
            return _weightViews.GetValue(_weights[key], vector => Tensor<T>.FromMemory(vector.AsWritableMemory(), shape));
        }

        /// <summary>
        /// Gets architecture parameters for optimization
        /// </summary>
        public List<Matrix<T>> GetArchitectureParameters()
        {
            return _architectureParams;
        }

        /// <summary>
        /// Gets architecture gradients
        /// </summary>
        public List<Matrix<T>> GetArchitectureGradients()
        {
            return _architectureGradients;
        }

        /// <summary>
        /// Gets weight parameters for optimization
        /// </summary>
        public Dictionary<string, Vector<T>> GetWeightParameters()
        {
            return _weights;
        }

        /// <summary>
        /// Gets weight gradients
        /// </summary>
        public Dictionary<string, Vector<T>> GetWeightGradients()
        {
            return _weightGradients;
        }

        /// <summary>
        /// Computes gradients of the loss function with respect to model parameters WITHOUT updating parameters.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="target">The target/expected output tensor.</param>
        /// <param name="lossFunction">The loss function to use. If null, uses the model's default loss function.</param>
        /// <returns>A vector containing gradients with respect to all model parameters (both architecture and weights).</returns>
        /// <exception cref="ArgumentNullException">If input or target is null.</exception>
        /// <remarks>
        /// <para>
        /// For SuperNet, this computes gradients for weight parameters only (not architecture parameters).
        /// Architecture parameters are updated separately in DARTS using validation data.
        /// The method uses the existing BackwardWeights method and collects gradients from all layers.
        /// </para>
        /// <para><b>For Beginners:</b>
        /// SuperNet has two types of parameters:
        /// - Architecture parameters (α): which operations to use
        /// - Weight parameters (w): the actual neural network weights
        ///
        /// This method computes gradients for the weight parameters based on training data.
        /// In DARTS, architecture parameters are optimized separately on validation data.
        /// </para>
        /// </remarks>
        public override Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (target == null)
                throw new ArgumentNullException(nameof(target));

            // Use the effective loss function (supplied or default)
            var effectiveLoss = lossFunction ?? _defaultLossFunction;

            // Use BackwardWeights to compute gradients for weight parameters
            BackwardWeights(input, target, effectiveLoss);

            // Collect all gradients into a single vector
            var gradients = new List<T>();

            // Add architecture parameter gradients as ZEROS (not computed in this method)
            // Architecture parameters are optimized separately in DARTS on validation data
            // We include zeros here to maintain consistent vector length with GetParameters()
            var zero = NumOps.FromDouble(0.0);
            foreach (var alpha in _architectureParams)
            {
                for (int i = 0; i < alpha.Rows; i++)
                    for (int j = 0; j < alpha.Columns; j++)
                        gradients.Add(zero);  // Zero gradient since not computed here
            }

            // Add weight gradients (freshly computed by BackwardWeights above)
            foreach (var weightGrad in _weightGradients.Values)
            {
                for (int i = 0; i < weightGrad.Length; i++)
                    gradients.Add(weightGrad[i]);
            }

            return new Vector<T>(gradients.ToArray());
        }

        /// <summary>
        /// Applies pre-computed gradients to update the model parameters.
        /// </summary>
        /// <param name="gradients">The gradient vector to apply.</param>
        /// <param name="learningRate">The learning rate for the update.</param>
        /// <exception cref="ArgumentNullException">If gradients is null.</exception>
        /// <exception cref="ArgumentException">If gradient vector length doesn't match parameter count.</exception>
        /// <remarks>
        /// <para>
        /// Updates both architecture and weight parameters using: θ = θ - learningRate * gradients
        /// </para>
        /// <para><b>For Beginners:</b>
        /// This method applies the gradient updates to both:
        /// - Architecture parameters (which operations are selected)
        /// - Weight parameters (the neural network weights)
        ///
        /// In DARTS, you typically call this with different learning rates for
        /// architecture and weight parameters.
        /// </para>
        /// </remarks>
        public override void ApplyGradients(Vector<T> gradients, T learningRate)
        {
            if (gradients == null)
                throw new ArgumentNullException(nameof(gradients));

            var currentParams = GetParameters();

            if (gradients.Length != currentParams.Length)
            {
                throw new ArgumentException(
                    $"Gradient vector length ({gradients.Length}) must match parameter count ({currentParams.Length})",
                    nameof(gradients));
            }

            int idx = 0;

            // Update architecture parameters
            foreach (var alpha in _architectureParams)
            {
                for (int i = 0; i < alpha.Rows; i++)
                {
                    for (int j = 0; j < alpha.Columns; j++)
                    {
                        T update = NumOps.Multiply(learningRate, gradients[idx++]);
                        alpha[i, j] = NumOps.Subtract(alpha[i, j], update);
                    }
                }
            }

            // Update weights
            foreach (var key in _weights.Keys.ToList())
            {
                var weight = _weights[key];
                for (int i = 0; i < weight.Length; i++)
                {
                    T update = NumOps.Multiply(learningRate, gradients[idx++]);
                    weight[i] = NumOps.Subtract(weight[i], update);
                }
            }
        }

        /// <summary>
        /// Derives discrete architecture from continuous parameters (argmax selection)
        /// </summary>
        public Architecture<T> DeriveArchitecture()
        {
            var architecture = new Architecture<T>();

            for (int nodeIdx = 0; nodeIdx < _numNodes; nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];
                var softmaxWeights = ApplySoftmax(alpha);

                // For each previous node connection, select operation with highest weight
                for (int prevNodeIdx = 0; prevNodeIdx <= nodeIdx; prevNodeIdx++)
                {
                    // The strongest non-zero operation (Liu et al. 2019, section 2.4). The zero operation marks
                    // an absent edge, so it is only chosen when it is the only candidate.
                    int bestOpIdx = -1;
                    for (int opIdx = 0; opIdx < _numOperations; opIdx++)
                    {
                        if (_operations[opIdx].Kind == OperationKind.Zero) continue;
                        if (bestOpIdx < 0 || NumOps.GreaterThan(softmaxWeights[prevNodeIdx, opIdx], softmaxWeights[prevNodeIdx, bestOpIdx]))
                        {
                            bestOpIdx = opIdx;
                        }
                    }

                    if (bestOpIdx < 0) bestOpIdx = 0;

                    // Add selected operation to architecture
                    var operation = GetOperationName(bestOpIdx);
                    architecture.AddOperation(nodeIdx, prevNodeIdx, operation);
                }
            }

            return architecture;
        }

        /// <summary>
        /// Apply softmax to architecture parameters
        /// </summary>
        private Matrix<T> ApplySoftmax(Matrix<T> alpha)
        {
            var result = new Matrix<T>(alpha.Rows, alpha.Columns);

            for (int row = 0; row < alpha.Rows; row++)
            {
                // Compute softmax for this row
                T maxVal = alpha[row, 0];
                for (int col = 1; col < alpha.Columns; col++)
                {
                    if (NumOps.GreaterThan(alpha[row, col], maxVal))
                        maxVal = alpha[row, col];
                }

                // Compute exp(x - max) for numerical stability
                T sumExp = NumOps.Zero;
                var expValues = new T[alpha.Columns];
                for (int col = 0; col < alpha.Columns; col++)
                {
                    expValues[col] = NumOps.Exp(NumOps.Subtract(alpha[row, col], maxVal));
                    sumExp = NumOps.Add(sumExp, expValues[col]);
                }

                // Normalize
                for (int col = 0; col < alpha.Columns; col++)
                {
                    result[row, col] = NumOps.Divide(expValues[col], sumExp);
                }
            }

            return result;
        }

        /// <summary>
        /// Applies candidate operation <paramref name="op"/> on the edge from node <paramref name="from"/> to
        /// node <paramref name="node"/>, to a [batch, features] tensor, on the active tape.
        /// </summary>
        /// <remarks>
        /// The cell works on a feature vector, so each named operation is its one-channel analogue along the
        /// feature axis: a KxK convolution is a width-K kernel, a KxK pool is a width-K window, and the
        /// residual, inverted-residual and squeeze-and-excitation blocks compose those. Depthwise, separable
        /// and grouped convolutions coincide with a plain one on a single channel. These replace constant
        /// scalings that only carried the names ("MaxPool" was x * 0.9).
        /// </remarks>
        private Tensor<T> ApplyOperation(Tensor<T> x, int node, int from, int op)
        {
            var operation = _operations[op];
            string key = WeightKey(node, from, op, string.Empty);
            int batch = x.Shape[0];
            int features = x.Shape[1];
            switch (operation.Kind)
            {
                case OperationKind.Identity:
                    return x;

                case OperationKind.Convolution:
                    return Convolve(x, WeightView(key + "kernel"), operation.KernelSize, operation.Dilation);

                case OperationKind.MaxPool:
                    return MaxPool(x, operation.KernelSize);

                case OperationKind.AveragePool:
                    return AveragePool(x, operation.KernelSize);

                case OperationKind.ResidualBasic:
                {
                    // He et al. 2016: relu(x + conv(relu(conv(x)))).
                    var inner = Engine.ReLU(Convolve(x, WeightView(key + "kernel1"), operation.KernelSize, 1));
                    return Engine.ReLU(Engine.TensorAdd(x, Convolve(inner, WeightView(key + "kernel2"), operation.KernelSize, 1)));
                }

                case OperationKind.ResidualBottleneck:
                {
                    // He et al. 2016: 1x1 reduce, KxK, 1x1 expand, each followed by ReLU, around the skip.
                    var reduced = Engine.ReLU(Convolve(x, WeightView(key + "reduce"), 1, 1));
                    var mixed = Engine.ReLU(Convolve(reduced, WeightView(key + "kernel"), operation.KernelSize, 1));
                    return Engine.ReLU(Engine.TensorAdd(x, Convolve(mixed, WeightView(key + "expand"), 1, 1)));
                }

                case OperationKind.InvertedResidual:
                {
                    // Sandler et al. 2018: 1x1 expansion to E channels, ReLU6, depthwise KxK, ReLU6, linear 1x1
                    // projection back, and the skip because input and output match.
                    int expansion = operation.Expansion;
                    var six = NumOps.FromDouble(6.0);
                    var expanded = Engine.TensorClamp(
                        Engine.Conv1D(Engine.Reshape(x, new[] { batch, 1, features }), WeightView(key + "expand")),
                        NumOps.Zero, six);
                    var depthwiseKernels = WeightView(key + "depthwise");
                    var channels = new Tensor<T>[expansion];
                    for (int c = 0; c < expansion; c++)
                    {
                        channels[c] = Engine.Conv1D(
                            Engine.TensorSlice(expanded, new[] { 0, c, 0 }, new[] { batch, 1, features }),
                            Engine.TensorSlice(depthwiseKernels, new[] { c, 0, 0 }, new[] { 1, 1, operation.KernelSize }),
                            stride: 1, padding: (operation.KernelSize - 1) / 2);
                    }

                    var depthwise = Engine.TensorClamp(Engine.TensorConcatenate(channels, axis: 1), NumOps.Zero, six);
                    var projected = Engine.Conv1D(depthwise, WeightView(key + "project"));
                    return Engine.TensorAdd(x, Engine.Reshape(projected, new[] { batch, features }));
                }

                case OperationKind.SqueezeExcitation:
                {
                    // Hu et al. 2018 on one channel: squeeze over the feature axis, a two-layer excitation, and
                    // a sigmoid gate that rescales the input.
                    var squeezed = Engine.ReduceMean(x, new[] { 1 }, keepDims: true);
                    var hidden = Engine.ReLU(Affine(squeezed, key + "w1", key + "b1"));
                    var gate = Engine.Sigmoid(Affine(hidden, key + "w2", key + "b2"));
                    return Engine.TensorMultiply(x, Engine.TensorTile(gate, new[] { 1, features }));
                }

                default:
                    return new Tensor<T>(new[] { batch, features });
            }
        }

        /// <summary>A width-K convolution along the feature axis with "same" padding.</summary>
        private Tensor<T> Convolve(Tensor<T> x, Tensor<T> kernel, int kernelSize, int dilation)
        {
            int batch = x.Shape[0];
            int features = x.Shape[1];
            var output = Engine.Conv1D(
                Engine.Reshape(x, new[] { batch, 1, features }), kernel,
                stride: 1, padding: dilation * (kernelSize - 1) / 2, dilation: dilation);
            return Engine.Reshape(output, new[] { batch, features });
        }

        /// <summary>w * s + b for a [batch, 1] tensor and scalar weights.</summary>
        private Tensor<T> Affine(Tensor<T> s, string weightKey, string biasKey)
        {
            int batch = s.Shape[0];
            return Engine.TensorAdd(
                Engine.TensorMultiply(s, Engine.TensorTile(WeightView(weightKey), new[] { batch, 1 })),
                Engine.TensorTile(WeightView(biasKey), new[] { batch, 1 }));
        }

        /// <summary>
        /// A width-K max over neighbouring features, stride 1. Shifting in the edge value is the same as
        /// ignoring padding, because the maximum already includes that value.
        /// </summary>
        private Tensor<T> MaxPool(Tensor<T> x, int kernelSize)
        {
            int left = (kernelSize - 1) / 2;
            Tensor<T>? result = null;
            for (int offset = -left; offset < kernelSize - left; offset++)
            {
                var shifted = Shift(x, offset, replicateEdge: true);
                result = result is null ? shifted : Engine.TensorMax(result, shifted);
            }

            return result ?? x;
        }

        /// <summary>
        /// A width-K mean over neighbouring features, stride 1, averaging only the positions that exist
        /// (count_include_pad = False, as in DARTS).
        /// </summary>
        private Tensor<T> AveragePool(Tensor<T> x, int kernelSize)
        {
            int batch = x.Shape[0];
            int features = x.Shape[1];
            int left = (kernelSize - 1) / 2;
            Tensor<T>? sum = null;
            var reciprocal = new Tensor<T>(new[] { 1, features });
            for (int f = 0; f < features; f++)
            {
                int count = 0;
                for (int offset = -left; offset < kernelSize - left; offset++)
                {
                    if (f + offset >= 0 && f + offset < features) count++;
                }

                reciprocal[0, f] = NumOps.FromDouble(1.0 / count);
            }

            for (int offset = -left; offset < kernelSize - left; offset++)
            {
                var shifted = Shift(x, offset, replicateEdge: false);
                sum = sum is null ? shifted : Engine.TensorAdd(sum, shifted);
            }

            return Engine.TensorMultiply(sum ?? x, Engine.TensorTile(reciprocal, new[] { batch, 1 }));
        }

        /// <summary>
        /// out[f] = x[f + offset] along the feature axis, filling positions past either edge with zero or
        /// with the edge value.
        /// </summary>
        private Tensor<T> Shift(Tensor<T> x, int offset, bool replicateEdge)
        {
            int batch = x.Shape[0];
            int features = x.Shape[1];
            int magnitude = Math.Min(Math.Abs(offset), features);
            if (offset == 0 || magnitude == 0) return x;

            Tensor<T> Fill(int edgeColumn) => replicateEdge
                ? Engine.TensorTile(Engine.TensorSlice(x, new[] { 0, edgeColumn }, new[] { batch, 1 }), new[] { 1, magnitude })
                : new Tensor<T>(new[] { batch, magnitude });

            if (magnitude == features) return Fill(offset > 0 ? features - 1 : 0);

            return offset > 0
                ? Engine.TensorConcatenate(new[]
                {
                    Engine.TensorSlice(x, new[] { 0, magnitude }, new[] { batch, features - magnitude }),
                    Fill(features - 1),
                }, axis: 1)
                : Engine.TensorConcatenate(new[]
                {
                    Fill(0),
                    Engine.TensorSlice(x, new[] { 0, 0 }, new[] { batch, features - magnitude }),
                }, axis: 1);
        }

        private static string WeightKey(int node, int from, int op, string part) => $"node{node}_from{from}_op{op}_{part}";

        /// <summary>
        /// Makes <see cref="_weights"/> hold exactly the planned operation weights, in plan order: a restored
        /// weight of the right length is kept, a missing one is initialised, and any other entry is dropped.
        /// </summary>
        private void ReconcileWeightsWithPlan()
        {
            var restored = new Dictionary<string, Vector<T>>(_weights);
            _weights.Clear();
            _weightGradients.Clear();
            _weightShapes.Clear();
            for (int node = 0; node < _numNodes; node++)
            {
                for (int from = 0; from <= node; from++)
                {
                    for (int op = 0; op < _numOperations; op++)
                    {
                        foreach (var (part, shape, fanIn) in WeightParts(_operations[op]))
                        {
                            string key = WeightKey(node, from, op, part);
                            int length = shape.Aggregate(1, (a, b) => a * b);
                            if (!restored.TryGetValue(key, out var weight) || weight.Length != length)
                            {
                                // PyTorch's default for convolutions and linear layers: U(-1/sqrt(fan_in),
                                // 1/sqrt(fan_in)); biases start at zero.
                                weight = new Vector<T>(length);
                                double bound = fanIn > 0 ? 1.0 / Math.Sqrt(fanIn) : 0.0;
                                for (int i = 0; i < length; i++)
                                {
                                    weight[i] = NumOps.FromDouble((_random.NextDouble() * 2.0 - 1.0) * bound);
                                }
                            }

                            _weights[key] = weight;
                            _weightGradients[key] = new Vector<T>(length);
                            _weightShapes[key] = shape;
                        }
                    }
                }
            }
        }

        /// <summary>The weights one operation owns: name suffix, tape shape, and fan-in for initialisation.</summary>
        private static IEnumerable<(string Part, int[] Shape, int FanIn)> WeightParts(Operation operation)
        {
            int k = operation.KernelSize;
            int e = operation.Expansion;
            switch (operation.Kind)
            {
                case OperationKind.Convolution:
                    yield return ("kernel", new[] { 1, 1, k }, k);
                    break;
                case OperationKind.ResidualBasic:
                    yield return ("kernel1", new[] { 1, 1, k }, k);
                    yield return ("kernel2", new[] { 1, 1, k }, k);
                    break;
                case OperationKind.ResidualBottleneck:
                    yield return ("reduce", new[] { 1, 1, 1 }, 1);
                    yield return ("kernel", new[] { 1, 1, k }, k);
                    yield return ("expand", new[] { 1, 1, 1 }, 1);
                    break;
                case OperationKind.InvertedResidual:
                    yield return ("expand", new[] { e, 1, 1 }, 1);
                    yield return ("depthwise", new[] { e, 1, k }, k);
                    yield return ("project", new[] { 1, e, 1 }, e);
                    break;
                case OperationKind.SqueezeExcitation:
                    yield return ("w1", new[] { 1, 1 }, 1);
                    yield return ("b1", new[] { 1, 1 }, 0);
                    yield return ("w2", new[] { 1, 1 }, 1);
                    yield return ("b2", new[] { 1, 1 }, 0);
                    break;
            }
        }

        private static readonly Regex SizedOperationName = new Regex(
            @"^(?<stem>[a-z_]+?)_?(?<k>[0-9])x\k<k>(?:_e(?<e>[0-9]+))?$",
            RegexOptions.CultureInvariant, TimeSpan.FromMilliseconds(100));

        /// <summary>
        /// Resolves a search-space name to the operation it denotes on this cell's feature axis.
        /// </summary>
        /// <exception cref="NotSupportedException">The name has no meaning on a feature vector, for example
        /// attention; it used to run silently as the identity.</exception>
        private static Operation ParseOperation(string name)
        {
            string normalized = (name ?? string.Empty).Trim().ToLowerInvariant();
            switch (normalized)
            {
                case "none":
                case "zero":
                    return new Operation(OperationKind.Zero);
                case "identity":
                case "skip":
                case "skip_connect":
                    return new Operation(OperationKind.Identity);
                case "residual_block_basic":
                    return new Operation(OperationKind.ResidualBasic, kernelSize: 3);
                case "residual_block_bottleneck":
                    return new Operation(OperationKind.ResidualBottleneck, kernelSize: 3);
                case "se_block":
                    return new Operation(OperationKind.SqueezeExcitation);
            }

            var match = SizedOperationName.Match(normalized);
            if (match.Success)
            {
                int k = int.Parse(match.Groups["k"].Value, CultureInfo.InvariantCulture);
                string stem = match.Groups["stem"].Value.TrimEnd('_');
                bool odd = k % 2 == 1;
                switch (stem)
                {
                    case "conv" or "sep_conv" or "separable_conv" or "depthwise_conv" or "grouped_conv" when odd:
                        return new Operation(OperationKind.Convolution, kernelSize: k);
                    case "dil_conv" or "dilated_conv" when odd:
                        return new Operation(OperationKind.Convolution, kernelSize: k, dilation: 2);
                    case "maxpool" or "max_pool":
                        return new Operation(OperationKind.MaxPool, kernelSize: k);
                    case "avgpool" or "avg_pool":
                        return new Operation(OperationKind.AveragePool, kernelSize: k);
                    case "inverted_residual" when odd && match.Groups["e"].Success:
                        return new Operation(OperationKind.InvertedResidual, kernelSize: k,
                            expansion: int.Parse(match.Groups["e"].Value, CultureInfo.InvariantCulture));
                }
            }

            throw new NotSupportedException(
                $"SuperNet has no operation for the search-space entry '{name}'. Its cell works on a feature " +
                "vector, where it supports: none/zero, identity/skip/skip_connect, convKxK with K odd (also " +
                "sep_conv, separable_conv, depthwise_conv, grouped_conv and dil_conv/dilated_conv), " +
                "maxpoolKxK/max_pool_KxK, avgpoolKxK/avg_pool_KxK, residual_block_basic, " +
                "residual_block_bottleneck, inverted_residual_KxK_eE and se_block.");
        }

        /// <summary>Gets the search space's own name for an operation index.</summary>
        private string GetOperationName(int opIdx)
            => opIdx >= 0 && opIdx < _operationNames.Length ? _operationNames[opIdx] : _operationNames[0];

        /// <summary>What a search-space operation name denotes on this cell's feature axis.</summary>
        private enum OperationKind
        {
            Zero,
            Identity,
            Convolution,
            MaxPool,
            AveragePool,
            ResidualBasic,
            ResidualBottleneck,
            InvertedResidual,
            SqueezeExcitation,
        }

        /// <summary>One candidate operation: its kind and the sizes its name states.</summary>
        private readonly struct Operation
        {
            public Operation(OperationKind kind, int kernelSize = 1, int dilation = 1, int expansion = 1)
            {
                Kind = kind;
                KernelSize = kernelSize;
                Dilation = dilation;
                Expansion = expansion;
            }

            public OperationKind Kind { get; }

            public int KernelSize { get; }

            public int Dilation { get; }

            public int Expansion { get; }
        }

        // Replaced by the declared parameter source below. Removed under AIDN082.

        // Replaced by the declared parameter source below. Removed under AIDN082.

        public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
        {
            var clone = (SuperNet<T>)Clone();
            clone.SetParameters(parameters);
            return clone;
        }

        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Description = "Differentiable Architecture Search SuperNet",
                FeatureCount = _inputSize,
                Complexity = _numNodes,
                AdditionalInfo = new Dictionary<string, object>
                {
                    ["NumNodes"] = _numNodes,
                    ["NumOperations"] = _numOperations,
                    ["ParameterCount"] = ParameterCount
                }
            };
        }

        public override void SaveModel(string filePath)
        {
            Helpers.ModelPersistenceGuard.EnforceBeforeSave();

            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

            // Validate path security: prevent directory traversal attacks
            // Use canonicalized path and ensure it is within the current working directory
            var fullPath = System.IO.Path.GetFullPath(filePath);

            // Additional validation: ensure the resolved path doesn't escape the working directory
            var currentDirectory = System.IO.Path.GetFullPath(Environment.CurrentDirectory);
            // Ensure trailing separator for strict directory containment (prevents /app vs /app-data bypass)
            var currentDirWithSep = currentDirectory.EndsWith(System.IO.Path.DirectorySeparatorChar.ToString())
                ? currentDirectory
                : currentDirectory + System.IO.Path.DirectorySeparatorChar;
            if (!fullPath.StartsWith(currentDirWithSep, StringComparison.OrdinalIgnoreCase))
                throw new UnauthorizedAccessException($"Attempted to save model outside of the current directory. Path: {fullPath}");

            using var fs = new System.IO.FileStream(fullPath, System.IO.FileMode.Create);
            using var writer = new System.IO.BinaryWriter(fs);

            writer.Write(_numNodes);
            writer.Write(_numOperations);
            writer.Write(_inputSize);
            writer.Write(_outputSize);

            // Serialize architecture parameters
            writer.Write(_architectureParams.Count);
            foreach (var alpha in _architectureParams)
            {
                writer.Write(alpha.Rows);
                writer.Write(alpha.Columns);
                for (int i = 0; i < alpha.Rows; i++)
                {
                    for (int j = 0; j < alpha.Columns; j++)
                    {
                        writer.Write(Convert.ToDouble(alpha[i, j]));
                    }
                }
            }

            // Serialize weights
            writer.Write(_weights.Count);
            foreach (var kvp in _weights)
            {
                writer.Write(kvp.Key);
                writer.Write(kvp.Value.Length);
                for (int i = 0; i < kvp.Value.Length; i++)
                {
                    writer.Write(Convert.ToDouble(kvp.Value[i]));
                }
            }
        }
        public override void LoadModel(string filePath)
        {
            Helpers.ModelPersistenceGuard.EnforceBeforeLoad();

            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

            // Validate path security: prevent directory traversal attacks
            // Use canonicalized path and ensure it is within the current working directory
            var fullPath = System.IO.Path.GetFullPath(filePath);

            // Additional validation: ensure the resolved path doesn't escape the working directory
            var currentDirectory = System.IO.Path.GetFullPath(Environment.CurrentDirectory);
            // Ensure trailing separator for strict directory containment (prevents /app vs /app-data bypass)
            var currentDirWithSep = currentDirectory.EndsWith(System.IO.Path.DirectorySeparatorChar.ToString())
                ? currentDirectory
                : currentDirectory + System.IO.Path.DirectorySeparatorChar;
            if (!fullPath.StartsWith(currentDirWithSep, StringComparison.OrdinalIgnoreCase))
                throw new UnauthorizedAccessException($"Attempted to load model from outside the current directory. Path: {fullPath}");

            if (!System.IO.File.Exists(fullPath))
                throw new System.IO.FileNotFoundException($"Model file not found: {filePath}");

            using var fs = new System.IO.FileStream(fullPath, System.IO.FileMode.Open);
            using var reader = new System.IO.BinaryReader(fs);

            // Deserialize _numNodes and _numOperations (read-only fields need reflection or constructor)
            var numNodes = reader.ReadInt32();
            var numOperations = reader.ReadInt32();

            // Validate that deserialized structure matches this instance
            if (numNodes != _numNodes || numOperations != _numOperations)
            {
                throw new InvalidOperationException(
                    $"Model file structure mismatch: file has numNodes={numNodes}, numOperations={numOperations}, " +
                    $"but this instance has numNodes={_numNodes}, numOperations={_numOperations}.");
            }

            _inputSize = reader.ReadInt32();
            _outputSize = reader.ReadInt32();

            // Deserialize architecture parameters
            int alphaCount = reader.ReadInt32();
            _architectureParams.Clear();
            _architectureGradients.Clear();
            for (int idx = 0; idx < alphaCount; idx++)
            {
                int rows = reader.ReadInt32();
                int cols = reader.ReadInt32();
                var alpha = new Matrix<T>(rows, cols);
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        alpha[i, j] = NumOps.FromDouble(reader.ReadDouble());
                    }
                }
                _architectureParams.Add(alpha);
                _architectureGradients.Add(new Matrix<T>(rows, cols));
            }

            // Deserialize weights
            int weightCount = reader.ReadInt32();
            _weights.Clear();
            _weightGradients.Clear();
            for (int idx = 0; idx < weightCount; idx++)
            {
                string key = reader.ReadString();
                int length = reader.ReadInt32();
                var weight = new Vector<T>(length);
                for (int i = 0; i < length; i++)
                {
                    weight[i] = NumOps.FromDouble(reader.ReadDouble());
                }
                _weights[key] = weight;
                _weightGradients[key] = new Vector<T>(length);
            }

            ReconcileWeightsWithPlan();
        }
        /// <summary>
        /// Declares the two collections the generator cannot place: the per-node architecture
        /// matrices and the string-keyed weight table.
        /// </summary>
        /// <param name="state">The registry to declare into.</param>
        /// <remarks>
        /// Both fields are readonly, so each setter refills the existing instance rather than
        /// replacing it - the same thing the hand-written Deserialize did with Clear() then Add().
        /// <para>
        /// _numNodes and _numOperations are readonly construction config; the hand-written pair
        /// wrote them only to VALIDATE on read, throwing when they disagreed, and the recorded
        /// constructor replays them. _architectureGradients is rebuilt to match the restored
        /// architecture shape, exactly as the old Deserialize rebuilt it, because a gradient is
        /// scratch from the last backward pass rather than model state.
        /// </para>
        /// </remarks>
        protected override void RegisterState(ModelStateRegistry<T> state)
        {
            base.RegisterState(state);

            state.Declare(
                "SuperNet._architectureParams",
                () => _architectureParams,
                v =>
                {
                    _architectureParams.Clear();
                    _architectureGradients.Clear();
                    if (v is null) return;
                    foreach (var alpha in v)
                    {
                        _architectureParams.Add(alpha);
                        _architectureGradients.Add(new Matrix<T>(alpha.Rows, alpha.Columns));
                    }
                });

            state.Declare(
                "SuperNet._weights",
                () => _weights,
                v =>
                {
                    _weights.Clear();
                    if (v is not null)
                    {
                        foreach (var pair in v) _weights[pair.Key] = pair.Value;
                    }

                    ReconcileWeightsWithPlan();
                });
        }

        public override Dictionary<string, T> GetFeatureImportance() => new Dictionary<string, T>();
        public override IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputSize);
        public override bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _inputSize;
        public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }

        #region IInterpretableModel Implementation

        /// <summary>
        /// Gets the operation importance for SuperNet architecture search.
        /// Returns importance scores for architectural operations rather than input features.
        /// </summary>
        /// <param name="inputs">Input tensor (required for interface compliance; not used in this implementation)</param>
        /// <returns>Dictionary mapping operation indices to their importance scores</returns>
        /// <remarks>
        /// <para>
        /// <b>Note:</b> SuperNet reinterprets "feature importance" as "operation importance" in the context of Neural Architecture Search (NAS).
        /// The returned dictionary maps operation indices (0=identity, 1=conv3x3, 2=conv5x5, etc.) to their importance scores,
        /// calculated by aggregating the absolute values of architecture parameters across all nodes.
        /// </para>
        /// <para>
        /// The 'inputs' parameter is required for IInterpretableModel interface compliance but is not used.
        /// SuperNet analyzes operation importance based on learned architecture parameters rather than input data.
        /// </para>
        /// </remarks>
        public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync(Tensor<T> inputs)
        {
            var importance = new Dictionary<int, T>();

            // For SuperNet, we analyze operation importance rather than input feature importance
            // Each operation index represents a different architectural operation (identity, conv3x3, etc.)
            for (int opIdx = 0; opIdx < _numOperations; opIdx++)
            {
                T sum = NumOps.Zero;

                // Aggregate importance across all nodes and connections
                foreach (var alpha in _architectureParams)
                {
                    // Sum absolute values of architecture parameters for this operation
                    for (int i = 0; i < alpha.Rows; i++)
                    {
                        if (opIdx < alpha.Columns)
                        {
                            sum = NumOps.Add(sum, NumOps.Abs(alpha[i, opIdx]));
                        }
                    }
                }

                importance[opIdx] = sum;
            }

            return await Task.FromResult(importance);
        }

        /// <summary>
        /// Gets the local feature importance for a specific input.
        /// Provides importance based on softmax weights, analyzing which operations are most active.
        /// </summary>
        public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Tensor<T> input)
        {
            var importance = new Dictionary<int, T>();

            // For local importance, we use softmax-transformed architecture parameters
            // to determine which operations are most active for this specific input
            for (int opIdx = 0; opIdx < _numOperations; opIdx++)
            {
                T sum = NumOps.Zero;

                // Apply softmax and aggregate weights for each operation
                foreach (var alpha in _architectureParams)
                {
                    var softmaxWeights = ApplySoftmax(alpha);

                    for (int i = 0; i < softmaxWeights.Rows; i++)
                    {
                        if (opIdx < softmaxWeights.Columns)
                        {
                            sum = NumOps.Add(sum, softmaxWeights[i, opIdx]);
                        }
                    }
                }

                importance[opIdx] = sum;
            }

            return await Task.FromResult(importance);
        }

        /// <summary>
        /// Gets SHAP values for the given inputs.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<Matrix<T>> GetShapValuesAsync(Tensor<T> inputs)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "SHAP values are not supported for SuperNet architecture search models. " +
                "SuperNet uses differentiable architecture search and does not have traditional feature attribution.");
        }

        /// <summary>
        /// Gets LIME explanation for a specific input.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Tensor<T> input, int numFeatures = 10)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "LIME explanations are not supported for SuperNet architecture search models. " +
                "Use GetGlobalFeatureImportanceAsync or GetLocalFeatureImportanceAsync instead.");
        }

        /// <summary>
        /// Gets partial dependence data for specified features.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "Partial dependence plots are not supported for SuperNet architecture search models. " +
                "SuperNet focuses on architecture optimization rather than feature-level analysis.");
        }

        /// <summary>
        /// Gets counterfactual explanation for a given input and desired output.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Tensor<T> input, Tensor<T> desiredOutput, int maxChanges = 5)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "Counterfactual explanations are not supported for SuperNet architecture search models. " +
                "SuperNet is designed for architecture search, not instance-level counterfactuals.");
        }

        /// <summary>
        /// Gets model-specific interpretability information for SuperNet.
        /// Returns architecture parameters and their importance.
        /// </summary>
        public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
        {
            var info = new Dictionary<string, object>
            {
                ["ModelType"] = "SuperNet (Differentiable Architecture Search)",
                ["NumNodes"] = _numNodes,
                ["NumOperations"] = _numOperations,
                // Dictionary<string, object> can box a long natively;
                // ToFlatVectorSize is reserved for places that genuinely
                // need an int (Vector<T> allocation, int-indexed APIs).
                // Storing the un-narrowed long lets >int.MaxValue
                // models surface their true count via the
                // interpretability dictionary without throwing here.
                // Closes review-comment #1271.vDPV.
                ["ParameterCount"] = ParameterCount,
                ["ArchitectureParameterCount"] = _architectureParams.Sum(a => a.Rows * a.Columns),
                ["WeightParameterCount"] = _weights.Values.Sum(w => w.Length),
                ["InputSize"] = _inputSize,
                ["OutputSize"] = _outputSize
            };

            // Add architecture parameter statistics
            var archStats = new List<Dictionary<string, object>>();
            for (int i = 0; i < _architectureParams.Count; i++)
            {
                var alpha = _architectureParams[i];
                var softmax = ApplySoftmax(alpha);

                var nodeStats = new Dictionary<string, object>
                {
                    ["NodeIndex"] = i,
                    ["Rows"] = alpha.Rows,
                    ["Columns"] = alpha.Columns,
                    ["ParameterCount"] = alpha.Rows * alpha.Columns
                };

                archStats.Add(nodeStats);
            }

            info["ArchitectureNodes"] = archStats;

            return await Task.FromResult(info);
        }

        /// <summary>
        /// Generates a text explanation for a prediction.
        /// Provides a description of which operations are most important in the SuperNet.
        /// </summary>
        public virtual async Task<string> GenerateTextExplanationAsync(Tensor<T> input, Tensor<T> prediction)
        {
            var explanation = $"SuperNet Architecture Search Model:\n";
            explanation += $"- Network contains {_numNodes} nodes with {_numOperations} operations each\n";
            explanation += $"- Total parameters: {ParameterCount}\n";
            explanation += $"- Architecture is determined by learned softmax weights over operations\n\n";

            explanation += "Most important architectural decisions:\n";

            // Identify most important nodes based on architecture parameters
            for (int nodeIdx = 0; nodeIdx < Math.Min(3, _numNodes); nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];
                var softmax = ApplySoftmax(alpha);

                // Find the operation with highest weight
                if (softmax.Rows > 0 && softmax.Columns > 0)
                {
                    int bestOp = 0;
                    T bestWeight = softmax[0, 0];

                    for (int i = 0; i < softmax.Rows; i++)
                    {
                        for (int j = 0; j < softmax.Columns; j++)
                        {
                            if (NumOps.GreaterThan(softmax[i, j], bestWeight))
                            {
                                bestWeight = softmax[i, j];
                                bestOp = j;
                            }
                        }
                    }

                    explanation += $"- Node {nodeIdx}: {GetOperationName(bestOp)} operation is dominant\n";
                }
                else
                {
                    explanation += $"- Node {nodeIdx}: No operations available (empty softmax matrix)\n";
                }
            }

            return await Task.FromResult(explanation);
        }

        /// <summary>
        /// Gets feature interaction effects between two features.
        /// Analyzes interactions between operations based on architecture parameter correlations.
        /// </summary>
        public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
        {
            // In SuperNet context, feature indices represent operation indices
            if (feature1Index < 0 || feature1Index >= _numOperations ||
                feature2Index < 0 || feature2Index >= _numOperations)
            {
                throw new ArgumentOutOfRangeException(
                    $"Feature indices must be in the range [0, {_numOperations - 1}]. " +
                    $"Received feature1Index={feature1Index}, feature2Index={feature2Index}.");
            }

            // Calculate correlation between two operations across all architecture parameters
            T sum1 = NumOps.Zero;
            T sum2 = NumOps.Zero;
            T sumProduct = NumOps.Zero;
            T sumSquares1 = NumOps.Zero;
            T sumSquares2 = NumOps.Zero;
            int count = 0;

            foreach (var alpha in _architectureParams)
            {
                for (int i = 0; i < alpha.Rows; i++)
                {
                    if (feature1Index < alpha.Columns && feature2Index < alpha.Columns)
                    {
                        T val1 = alpha[i, feature1Index];
                        T val2 = alpha[i, feature2Index];

                        sum1 = NumOps.Add(sum1, val1);
                        sum2 = NumOps.Add(sum2, val2);
                        sumProduct = NumOps.Add(sumProduct, NumOps.Multiply(val1, val2));
                        sumSquares1 = NumOps.Add(sumSquares1, NumOps.Multiply(val1, val1));
                        sumSquares2 = NumOps.Add(sumSquares2, NumOps.Multiply(val2, val2));
                        count++;
                    }
                }
            }

            if (count == 0)
            {
                return NumOps.Zero;
            }

            // Calculate correlation coefficient
            T n = NumOps.FromDouble(count);
            T numerator = NumOps.Subtract(
                NumOps.Multiply(n, sumProduct),
                NumOps.Multiply(sum1, sum2)
            );

            T denom1 = NumOps.Subtract(
                NumOps.Multiply(n, sumSquares1),
                NumOps.Multiply(sum1, sum1)
            );

            T denom2 = NumOps.Subtract(
                NumOps.Multiply(n, sumSquares2),
                NumOps.Multiply(sum2, sum2)
            );

            T denominator = NumOps.Multiply(denom1, denom2);

            // Avoid division by zero
            if (NumOps.Equals(denominator, NumOps.Zero))
            {
                return NumOps.Zero;
            }

            T correlation = NumOps.Divide(numerator, NumOps.Sqrt(denominator));

            return await Task.FromResult(correlation);
        }

        /// <summary>
        /// Validates fairness metrics for the given inputs.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Tensor<T> inputs, int sensitiveFeatureIndex)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "Fairness validation is not supported for SuperNet architecture search models. " +
                "SuperNet focuses on architecture optimization rather than fairness evaluation.");
        }

        /// <summary>
        /// Gets anchor explanation for a given input.
        /// Not supported for SuperNet architecture search models.
        /// </summary>
        public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Tensor<T> input, T threshold)
        {
            await Task.CompletedTask;
            throw new NotSupportedException(
                "Anchor explanations are not supported for SuperNet architecture search models. " +
                "SuperNet focuses on architecture optimization rather than instance-level explanations.");
        }

        /// <summary>
        /// Sets the base model for interpretability analysis.
        /// </summary>
        public virtual void SetBaseModel(IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> model)
        {
            Guard.NotNull(model);
            _baseModel = model;
        }

        /// <summary>
        /// Enables specific interpretation methods.
        /// </summary>
        public virtual void EnableMethod(params InterpretationMethod[] methods)
        {
            if (methods == null)
                return;

            foreach (var method in methods)
            {
                _enabledMethods.Add(method);
            }
        }

        /// <summary>
        /// Configures fairness evaluation settings.
        /// </summary>
        public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
        {
            Guard.NotNull(sensitiveFeatures);
            _sensitiveFeatures = sensitiveFeatures;
            _fairnessMetrics.Clear();
            if (fairnessMetrics != null)
            {
                _fairnessMetrics.AddRange(fairnessMetrics);
            }
        }

        #endregion

        /// <summary>
        /// Saves the SuperNet's current state (architecture parameters and weights) to a stream.
        /// </summary>
        /// <param name="stream">The stream to write the model state to.</param>
        /// <remarks>
        /// <para>
        /// This method serializes all the information needed to recreate the SuperNet's current state,
        /// including architecture parameters, operation weights, and model configuration.
        /// It uses the existing Serialize method and writes the data to the provided stream.
        /// </para>
        /// <para><b>For Beginners:</b> This is like creating a snapshot of your neural architecture search model.
        ///
        /// When you call SaveState:
        /// - All architecture parameters (alpha values) are written to the stream
        /// - All operation weights are saved
        /// - The model's configuration and structure are preserved
        ///
        /// This is particularly useful for:
        /// - Checkpointing during neural architecture search
        /// - Saving the best architecture found during search
        /// - Knowledge distillation from SuperNet to final architecture
        /// - Resuming interrupted architecture search
        ///
        /// You can later use LoadState to restore the model to this exact state.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
        /// <exception cref="IOException">Thrown when there's an error writing to the stream.</exception>
        public override void SaveState(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            if (!stream.CanWrite)
                throw new ArgumentException("Stream must be writable.", nameof(stream));

            try
            {
                var data = this.Serialize();
                stream.Write(data, 0, data.Length);
                stream.Flush();
            }
            catch (IOException ex)
            {
                throw new IOException($"Failed to save SuperNet state to stream: {ex.Message}", ex);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Unexpected error while saving SuperNet state: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Loads the SuperNet's state (architecture parameters and weights) from a stream.
        /// </summary>
        /// <param name="stream">The stream to read the model state from.</param>
        /// <remarks>
        /// <para>
        /// This method deserializes SuperNet state that was previously saved with SaveState,
        /// restoring all architecture parameters, operation weights, and configuration.
        /// It uses the existing Deserialize method after reading data from the stream.
        /// </para>
        /// <para><b>For Beginners:</b> This is like loading a saved snapshot of your neural architecture search model.
        ///
        /// When you call LoadState:
        /// - All architecture parameters (alpha values) are read from the stream
        /// - All operation weights are restored
        /// - The model is configured to match the saved state
        ///
        /// After loading, the model can:
        /// - Continue architecture search from where it left off
        /// - Make predictions using the restored architecture
        /// - Be used for further optimization or deployment
        ///
        /// This is essential for:
        /// - Resuming interrupted architecture search
        /// - Loading the best architecture found during search
        /// - Deploying searched architectures to production
        /// - Knowledge distillation workflows
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
        /// <exception cref="IOException">Thrown when there's an error reading from the stream.</exception>
        /// <exception cref="InvalidOperationException">Thrown when the stream contains invalid or incompatible data.</exception>
        public override void LoadState(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            if (!stream.CanRead)
                throw new ArgumentException("Stream must be readable.", nameof(stream));

            try
            {
                using var ms = new MemoryStream();
                stream.CopyTo(ms);
                var data = ms.ToArray();

                if (data.Length == 0)
                    throw new InvalidOperationException("Stream contains no data.");

                this.Deserialize(data);
            }
            catch (IOException ex)
            {
                throw new IOException($"Failed to read SuperNet state from stream: {ex.Message}", ex);
            }
            catch (InvalidOperationException)
            {
                // Re-throw InvalidOperationException from Deserialize
                throw;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Failed to deserialize SuperNet state. The stream may contain corrupted or incompatible data: {ex.Message}", ex);
            }
        }

    }
}
