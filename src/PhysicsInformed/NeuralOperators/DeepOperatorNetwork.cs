using AiDotNet.LearningRateSchedulers;
using System;
using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.PhysicsInformed.Options;

namespace AiDotNet.PhysicsInformed.NeuralOperators
{
    /// <summary>
    /// Implements Deep Operator Network (DeepONet) for learning operators.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// DeepONet is another approach to learning operators (like FNO), but with a different architecture.
    ///
    /// Universal Approximation Theorem for Operators:
    /// Just as neural networks can approximate any function, DeepONet can approximate any operator!
    /// This is based on a theorem by Chen and Chen (1995).
    ///
    /// The Key Idea - Decomposition:
    /// DeepONet represents an operator G as:
    /// G(u)(y) = Σᵢ bᵢ(u) * tᵢ(y) + b₀
    ///
    /// Where:
    /// - u is the input function
    /// - y is the query location
    /// - bᵢ(u) are "basis functions" of the input (learned by Branch Net)
    /// - tᵢ(y) are "basis functions" of the location (learned by Trunk Net)
    ///
    /// Architecture:
    /// DeepONet has TWO networks:
    ///
    /// 1. Branch Network:
    ///    - Input: The entire input function u(x) (sampled at sensors)
    ///    - Output: Coefficients b₁, b₂, ..., bₚ
    ///    - Role: Encodes information about the input function
    ///
    /// 2. Trunk Network:
    ///    - Input: Query location y (where we want to evaluate output)
    ///    - Output: Basis functions t₁(y), t₂(y), ..., tₚ(y)
    ///    - Role: Encodes spatial/temporal patterns
    ///
    /// 3. Combination:
    ///    - Output: G(u)(y) = b · t + b₀ = Σᵢ bᵢ * tᵢ(y) + b₀
    ///    - b₀ is a learned scalar bias (Lu et al. 2021, Eq. 2), which the paper reports lowers
    ///      both training and test error
    ///    - Simple dot product of the two network outputs
    ///
    /// Analogy:
    /// Think of it like a bilinear form or low-rank factorization:
    /// - Branch net learns "what" information matters in the input
    /// - Trunk net learns "where" patterns occur spatially
    /// - Their interaction gives the output
    ///
    /// Example - Heat Equation:
    /// Problem: Given initial temperature u(x,0), find temperature u(x,t)
    ///
    /// Branch Net:
    /// - Input: u(x,0) sampled at many points → [u(x₁,0), u(x₂,0), ..., u(xₙ,0)]
    /// - Learns: "This initial condition is smooth/peaked/oscillatory"
    /// - Output: Coefficients [b₁, b₂, ..., bₚ]
    ///
    /// Trunk Net:
    /// - Input: (x, t) where we want to know the temperature
    /// - Learns: Spatial-temporal basis functions
    /// - Output: Basis values [t₁(x,t), t₂(x,t), ..., tₚ(x,t)]
    ///
    /// Result:
    /// u(x,t) = Σᵢ bᵢ * tᵢ(x,t)
    ///
    /// Key Advantages:
    /// 1. Sensor flexibility: Can use different sensor locations at test time
    /// 2. Query flexibility: Can evaluate at any location y
    /// 3. Theoretical foundation: Universal approximation theorem
    /// 4. Efficient: Once trained, very fast evaluation
    /// 5. Interpretable: Decomposition into branch/trunk has clear meaning
    ///
    /// Comparison with FNO:
    /// DeepONet:
    /// - Works on unstructured data (any sensor locations)
    /// - More flexible for irregular domains
    /// - Requires specifying sensor locations
    /// - Good for problems with sparse/irregular data
    ///
    /// FNO:
    /// - Works on structured grids
    /// - Uses FFT (very efficient)
    /// - Resolution-invariant
    /// - Good for periodic/regular problems
    ///
    /// Both are powerful, choice depends on your problem!
    ///
    /// Applications:
    /// - Same as FNO: PDEs, climate, fluids, etc.
    /// - Particularly good for:
    ///   * Inverse problems (finding unknown parameters)
    ///   * Problems with sparse measurements
    ///   * Irregular geometries
    ///   * Multi-scale phenomena
    ///
    /// Historical Note:
    /// DeepONet was introduced by Lu et al. (2021) and has been highly successful
    /// in learning solution operators for PDEs with theoretical guarantees.
    /// </remarks>
    /// <example>
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(
    ///     inputType: InputType.OneDimensional,
    ///     taskType: NeuralNetworkTaskType.Regression,
    ///     inputSize: 100, outputSize: 1);
    /// var branchArch = new NeuralNetworkArchitecture&lt;float&gt;(
    ///     inputType: InputType.OneDimensional,
    ///     taskType: NeuralNetworkTaskType.Regression,
    ///     inputSize: 100, outputSize: 128);
    /// var trunkArch = new NeuralNetworkArchitecture&lt;float&gt;(
    ///     inputType: InputType.OneDimensional,
    ///     taskType: NeuralNetworkTaskType.Regression,
    ///     inputSize: 2, outputSize: 128);
    /// var deepONet = new DeepOperatorNetwork&lt;float&gt;(arch, branchArch, trunkArch);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Science)]
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.PhysicsInformed)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators", "https://doi.org/10.1038/s42256-021-00302-5", Year = 2021, Authors = "Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, George Em Karniadakis")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    [PaperOptimizer(OptimizerKind.Adam, LearningRate = 0.001,
                    Source = "Lu et al. 2021, Sec. 4: in all problems the Adam optimizer is used with a "
                            + "learning rate of 0.001, the number of iterations chosen to guarantee "
                            + "convergence. The 0.01 and 0.0001 rates elsewhere in the paper belong to its "
                            + "learning-rate ablation study, not to the trained model.")]
    public partial class DeepOperatorNetwork<T> : NeuralNetworkBase<T>
    {

        // InitializeLayers already does Layers.AddRange(_branchNet.Layers) followed by Layers.AddRange(_trunkNet.Layers), so the base walk over Layers ALREADY produces the branch-then-trunk parameters the hand-written surfaces rebuilt by hand, in the same order.
        private readonly DeepOperatorNetworkOptions _options;

        private readonly NeuralNetworkBase<T> _branchNet;
        private readonly NeuralNetworkBase<T> _trunkNet;
        private readonly int _latentDimension; // Dimension of the latent space (number of basis functions)

        // The constructor's own architectures, held so the clone plan can replay the constructor. Without
        // them a clone rebuilt the trunk from the overall architecture (input width 6 where the trunk
        // takes 2) and every clone rejected its input.
        private readonly NeuralNetworkArchitecture<T> _branchArchitecture;
        private readonly NeuralNetworkArchitecture<T> _trunkArchitecture;
        private readonly int _numSensors;
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;

        // The learned output bias b0 of Lu et al. 2021, Eq. 2: G(u)(y) ~ sum_k b_k t_k + b0.
        // Zero-initialized, so an untrained model's output is exactly the branch-trunk product.
        [TrainableParameter]
        private Tensor<T> _outputBias;

        /// <summary>
        /// Initializes a new instance of DeepONet.
        /// </summary>
        /// <param name="architecture">The overall architecture (mainly for metadata).</param>
        /// <param name="branchArchitecture">Architecture for the branch network.</param>
        /// <param name="trunkArchitecture">Architecture for the trunk network.</param>
        /// <remarks>
        /// For Beginners:
        ///
        /// Parameters:
        ///
        /// latentDimension (p): Number of basis functions
        /// - Controls the expressiveness of the operator
        /// - Higher p = more expressive but more parameters
        /// - Typical: 100-400
        /// - Like the rank in low-rank matrix factorization
        ///
        /// numSensors: How many points to sample the input function
        /// - More sensors = more information about input
        /// - Must be enough to capture important features
        /// - Typical: 50-200
        /// - Can use different sensor locations at train vs. test time!
        ///
        /// Branch Network:
        /// - Input size: numSensors (values of input function at sensors)
        /// - Output size: latentDimension (p)
        /// - Architecture: Deep feedforward network
        /// - Typical: 3-5 layers, 100-200 neurons per layer
        ///
        /// Trunk Network:
        /// - Input size: dimension of query location (e.g., 2 for (x,y), 3 for (x,y,t))
        /// - Output size: latentDimension (p)
        /// - Architecture: Deep feedforward network
        /// - Typical: 3-5 layers, 100-200 neurons per layer
        /// </remarks>
        public DeepOperatorNetwork(NeuralNetworkArchitecture<T> architecture,
            NeuralNetworkArchitecture<T> branchArchitecture,
            NeuralNetworkArchitecture<T> trunkArchitecture,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            DeepOperatorNetworkOptions? options = null)
            : base(architecture ?? throw new ArgumentNullException(nameof(architecture)),
                NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            options ??= new DeepOperatorNetworkOptions();
            _options = options;
            Options = _options;

            if (branchArchitecture == null)
            {
                throw new ArgumentNullException(nameof(branchArchitecture));
            }

            if (trunkArchitecture == null)
            {
                throw new ArgumentNullException(nameof(trunkArchitecture));
            }

            if (options.LatentDimension <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(options.LatentDimension), "Latent dimension must be positive.");
            }

            if (options.NumSensors <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(options.NumSensors), "Number of sensors must be positive.");
            }

            _latentDimension = options.LatentDimension;
            _branchArchitecture = branchArchitecture;
            _trunkArchitecture = trunkArchitecture;
            _numSensors = options.NumSensors;
            _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _usesDefaultOptimizer = optimizer == null;
            _outputBias = new Tensor<T>(new[] { 1 });

            // Create branch network
            var branchNetArchitecture = EnsureOutputSize(branchArchitecture, options.LatentDimension, "Branch");
            _branchNet = new FeedForwardNeuralNetwork<T>(branchNetArchitecture);

            // Create trunk network
            var trunkNetArchitecture = EnsureOutputSize(trunkArchitecture, options.LatentDimension, "Trunk");
            _trunkNet = new FeedForwardNeuralNetwork<T>(trunkNetArchitecture);

            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            ClearLayers();
            // Register branch and trunk network layers for proper parameter management
            Layers.AddRange(_branchNet.Layers);
            Layers.AddRange(_trunkNet.Layers);
        }

        private static NeuralNetworkArchitecture<T> EnsureOutputSize(
            NeuralNetworkArchitecture<T> architecture,
            int outputSize,
            string name)
        {
            if (architecture == null)
            {
                throw new ArgumentNullException(nameof(architecture));
            }

            if (architecture.OutputSize == outputSize)
            {
                return architecture;
            }

            if (architecture.Layers != null && architecture.Layers.Count > 0)
            {
                throw new ArgumentException($"{name} architecture output size must be {outputSize} when custom layers are provided.");
            }

            return new NeuralNetworkArchitecture<T>(
                architecture.InputType,
                architecture.TaskType,
                architecture.Complexity,
                architecture.InputSize,
                architecture.InputHeight,
                architecture.InputWidth,
                architecture.InputDepth,
                outputSize,
                null,
                architecture.ShouldReturnFullSequence);
        }

        /// <summary>
        /// Evaluates the operator: G(u)(y) = branch(u) · trunk(y) + b0.
        /// </summary>
        /// <param name="inputFunction">Values of input function at sensor locations [numSensors].</param>
        /// <param name="queryLocation">Location where to evaluate output [spatialDim].</param>
        /// <returns>Output value at the query location.</returns>
        /// <remarks>
        /// For Beginners:
        /// This is the forward pass of DeepONet.
        ///
        /// Steps:
        /// 1. Pass input function values through branch network → get coefficients b
        /// 2. Pass query location through trunk network → get basis functions t
        /// 3. Compute dot product: output = b · t
        ///
        /// Example:
        /// - inputFunction = [0.5, 0.7, 0.3, ...] (100 values)
        ///   → branch net → [b₁, b₂, ..., b₁₂₈] (128 coefficients)
        ///
        /// - queryLocation = [0.3, 0.5] (x=0.3, y=0.5)
        ///   → trunk net → [t₁, t₂, ..., t₁₂₈] (128 basis values)
        ///
        /// - output = b₁*t₁ + b₂*t₂ + ... + b₁₂₈*t₁₂₈ (single number)
        ///
        /// 4. Add the learned bias b0 (Lu et al. 2021, Eq. 2)
        ///
        /// To get output at many locations, call EvaluateMultiple (branch computed once per batch).
        /// </remarks>
        public T Evaluate(Vector<T> inputFunction, Vector<T> queryLocation)
        {
            Guard.NotNull(queryLocation);
            var query = Engine.Reshape(Tensor<T>.FromVector(queryLocation), new[] { 1, queryLocation.Length });
            return EvaluateMultiple(inputFunction, query)[0];
        }

        /// <summary>
        /// Evaluates the operator at multiple query locations efficiently.
        /// </summary>
        /// <param name="inputFunction">Input function values at sensors.</param>
        /// <param name="queryLocations">Multiple query locations [numQueries, spatialDim].</param>
        /// <returns>Output values at all query locations [numQueries].</returns>
        /// <remarks>
        /// For Beginners:
        /// Once you encode the input function via the branch network, you can query the solution at many locations
        /// in one batched pass. The result includes the learned bias b0 of Lu et al. (2021), Eq. 2, exactly as
        /// Predict does.
        /// </remarks>
        public Vector<T> EvaluateMultiple(Vector<T> inputFunction, Tensor<T> queryLocations)
        {
            Guard.NotNull(inputFunction);
            Guard.NotNull(queryLocations);
            if (inputFunction.Length != _numSensors)
            {
                throw new ArgumentException($"Input function must have {_numSensors} sensor values.", nameof(inputFunction));
            }

            if (queryLocations.Rank != 2)
            {
                throw new ArgumentException("Query locations must be [numQueries, spatialDim].", nameof(queryLocations));
            }

            if (queryLocations.Shape[0] == 0)
            {
                return new Vector<T>(0);
            }

            var sensors = Engine.Reshape(Tensor<T>.FromVector(inputFunction), new[] { 1, _numSensors });
            var queries = Engine.Reshape(queryLocations, new[] { 1, queryLocations.Shape[0], queryLocations.Shape[1] });
            return Predict(PackRows(sensors, queries)).ToVector();
        }

        /// <summary>
        /// Packs sensors [S, m] and queries [S, Q, d] into the [S * Q, m + d] rows the operator reads, the sensors of
        /// sample s repeated on each of its Q rows.
        /// </summary>
        private Tensor<T> PackRows(Tensor<T> sensors, Tensor<T> queries)
        {
            int trunkInputSize = _trunkNet.Architecture.CalculatedInputSize;
            if (trunkInputSize <= 0)
            {
                throw new InvalidOperationException("Trunk network input size must be positive.");
            }

            if (queries.Shape[2] != trunkInputSize)
            {
                throw new ArgumentException($"Query locations must have {trunkInputSize} values per query.", nameof(queries));
            }

            int samples = sensors.Shape[0];
            int count = queries.Shape[1];
            var repeated = Engine.TensorTile(Engine.Reshape(sensors, new[] { samples, 1, _numSensors }), new[] { 1, count, 1 });
            var rows = Engine.TensorConcatenate(new[] { repeated, queries }, 2);
            return Engine.Reshape(rows, new[] { samples * count, _numSensors + trunkInputSize });
        }

        /// <summary>
        /// Trains DeepONet on input-output function pairs.
        /// </summary>
        /// <param name="inputFunctions">Training input functions [numSamples, numSensors].</param>
        /// <param name="queryLocations">Query locations for each sample [numSamples, numQueries, spatialDim].</param>
        /// <param name="targetValues">Target output values [numSamples, numQueries].</param>
        /// <param name="epochs">Number of training epochs.</param>
        /// <param name="learningRate">The learning rate of the default Adam optimizer.</param>
        /// <param name="verbose">Whether to print the loss every 10 epochs.</param>
        /// <returns>Training history.</returns>
        /// <remarks>
        /// For Beginners:
        /// Each step evaluates the full operator G(u)(y) = sum_k b_k(u) t_k(y) + b0 at one sample's query
        /// locations, takes the loss against the targets and updates the branch net, the trunk net and the bias b0
        /// together, as in Lu et al. (2021). You can use different query locations for each training sample.
        /// </remarks>
        public TrainingHistory<T> Train(
            Tensor<T> inputFunctions,
            Tensor<T> queryLocations,
            Tensor<T> targetValues,
            int epochs = 100,
            double learningRate = 0.001,
            bool verbose = true)
        {
            Guard.NotNull(inputFunctions);
            Guard.NotNull(queryLocations);
            Guard.NotNull(targetValues);
            var history = new TrainingHistory<T>();
            var lossFunction = LossFunction ?? new MeanSquaredErrorLoss<T>();

            if (inputFunctions.Rank != 2 || inputFunctions.Shape[1] != _numSensors)
            {
                throw new ArgumentException($"Input functions must be [numSamples, {_numSensors}].", nameof(inputFunctions));
            }

            int numSamples = inputFunctions.Shape[0];
            if (queryLocations.Rank != 3 || queryLocations.Shape[0] != numSamples)
            {
                throw new ArgumentException("Query locations must be [numSamples, numQueries, spatialDim].", nameof(queryLocations));
            }

            int numQueries = queryLocations.Shape[1];
            if (targetValues.Rank != 2 || targetValues.Shape[0] != numSamples || targetValues.Shape[1] != numQueries)
            {
                throw new ArgumentException("Target values must be [numSamples, numQueries].", nameof(targetValues));
            }

            if (numSamples == 0)
            {
                return history;
            }

            if (numQueries == 0)
            {
                throw new ArgumentException("Query locations must include at least one query per sample.", nameof(queryLocations));
            }

            var rows = PackRows(inputFunctions, queryLocations);
            var targets = Engine.Reshape(targetValues, new[] { numSamples * numQueries, 1 });
            int width = rows.Shape[1];

            if (_usesDefaultOptimizer)
            {
                var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = learningRate
                };
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
            }

            SetTrainingMode(true);
            _branchNet.SetTrainingMode(true);
            _trunkNet.SetTrainingMode(true);
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(true);
            }

            try
            {
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    var losses = new List<T>(numSamples);
                    for (int i = 0; i < numSamples; i++)
                    {
                        // One step over the full operator, bias included: the parameter set is the model's own,
                        // so b0 trains with the branch and trunk weights.
                        var sampleRows = Engine.TensorSlice(rows, new[] { i * numQueries, 0 }, new[] { numQueries, width });
                        var sampleTargets = Engine.TensorSlice(targets, new[] { i * numQueries, 0 }, new[] { numQueries, 1 });
                        losses.Add(TrainCustomTapeStep(
                            sampleRows, sampleTargets, ForwardOperator, lossFunction.ComputeTapeLoss, _optimizer));
                    }

                    var lossTensor = Tensor<T>.FromVector(Vector<T>.FromList(losses));
                    T avgLoss = Engine.ReduceMean(lossTensor, new[] { 0 }, keepDims: false).AsSpan()[0];
                    history.AddEpoch(avgLoss);

                    if (verbose && epoch % 10 == 0)
                    {
                        Console.WriteLine($"Epoch {epoch}/{epochs}, Loss: {avgLoss}");
                    }
                }
            }
            finally
            {
                ClearNetworkGradients(_branchNet);
                ClearNetworkGradients(_trunkNet);
                foreach (var layer in Layers)
                {
                    layer.SetTrainingMode(false);
                }

                _trunkNet.SetTrainingMode(false);
                _branchNet.SetTrainingMode(false);
                SetTrainingMode(false);
            }

            return history;
        }

        /// <summary>
        /// Makes a prediction using the DeepONet for a batch of input/query pairs.
        /// </summary>
        /// <param name="input">Tensor containing input function values followed by query locations.</param>
        /// <returns>Predicted output tensor.</returns>
        protected override Tensor<T> PredictCore(Tensor<T> input)
        {
            bool wasTraining = IsTrainingMode;
            SetTrainingMode(false);
            try
            {
                return ForwardOperator(input);
            }
            finally
            {
                SetTrainingMode(wasTraining);
            }
        }

        /// <summary>
        /// Evaluates G(u)(y) = sum_k b_k(u) t_k(y) + b0 for a batch of [sensors | query] rows.
        /// </summary>
        /// <remarks>
        /// Prediction and training both evaluate this one function. Training used to walk Layers - the
        /// branch layers followed by the trunk layers - as a single chain over the whole concatenated
        /// input, which is not this operator. The split is an engine slice and the sub-networks' layers
        /// run directly, so the gradient tape reaches both networks and the bias.
        /// </remarks>
        private Tensor<T> ForwardOperator(Tensor<T> input)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (input.Rank != 2)
            {
                throw new ArgumentException("DeepONet expects a 2D input tensor.", nameof(input));
            }

            int trunkInputSize = _trunkNet.Architecture.CalculatedInputSize;
            if (trunkInputSize <= 0)
            {
                throw new InvalidOperationException("Trunk network input size must be positive.");
            }

            int expectedWidth = _numSensors + trunkInputSize;
            if (input.Shape[1] != expectedWidth)
            {
                throw new ArgumentException($"Expected input width {expectedWidth} (sensors + query), got {input.Shape[1]}.", nameof(input));
            }

            int batch = input.Shape[0];
            var branchInput = Engine.TensorSlice(input, new[] { 0, 0 }, new[] { batch, _numSensors });
            var trunkInput = Engine.TensorSlice(input, new[] { 0, _numSensors }, new[] { batch, trunkInputSize });

            var coefficients = RunLayers(_branchNet, branchInput);
            var basis = RunLayers(_trunkNet, trunkInput);
            if (coefficients.Rank != 2)
            {
                coefficients = Engine.Reshape(coefficients, new[] { batch, _latentDimension });
            }

            if (basis.Rank != 2)
            {
                basis = Engine.Reshape(basis, new[] { batch, _latentDimension });
            }

            var summed = Engine.ReduceSum(Engine.TensorMultiply(coefficients, basis), new[] { 1 }, keepDims: true);
            var bias = Engine.TensorTile(Engine.Reshape(_outputBias, new[] { 1, 1 }), new[] { batch, 1 });
            return Engine.TensorAdd(summed, bias);
        }

        private static Tensor<T> RunLayers(NeuralNetworkBase<T> network, Tensor<T> input)
        {
            var current = input;
            foreach (var layer in network.Layers)
            {
                current = layer.Forward(current);
            }

            return current;
        }

    // UpdateParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
        /// <inheritdoc />
        /// <remarks>
        /// The gradients of the last step in the layout of <see cref="NeuralNetworkBase{T}.GetParameters"/>: the
        /// branch layers, the trunk layers, then the output bias b0, whose gradient is dL/db0 = sum over the outputs of
        /// dL/dG (Lu et al. 2021, Eq. 2). It concatenated the sub-networks' own gradients, one entry short of the
        /// parameters and with no b0 gradient. Before any training step every gradient is zero.
        /// </remarks>
        public override Vector<T> GetGradients()
        {
            var gradients = GetParameterGradients();
            int parameterCount = GetParameters().Length;
            return gradients.Length == parameterCount ? gradients : new Vector<T>(parameterCount);
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The input is <c>[batch, sensors + query]</c>; a <c>[batch]</c> target is read as <c>[batch, 1]</c>.
        /// </remarks>
        protected override (Tensor<T> Input, Tensor<T>? Target) PrepareTrainingPair(Tensor<T> input, Tensor<T>? target, long draw)
        {
            if (input.Rank != 2)
            {
                throw new ArgumentException("DeepONet expects a 2D input tensor.", nameof(input));
            }

            int trunkInputSize = _trunkNet.Architecture.CalculatedInputSize;
            if (trunkInputSize <= 0)
            {
                throw new InvalidOperationException("Trunk network input size must be positive.");
            }

            int expectedWidth = _numSensors + trunkInputSize;
            if (input.Shape[1] != expectedWidth)
            {
                throw new ArgumentException($"Expected input width {expectedWidth} (sensors + query), got {input.Shape[1]}.", nameof(input));
            }

            if (target is null) return (input, null);
            int batchSize = input.Shape[0];
            if (target.Rank == 1)
            {
                return (input, target.Reshape(batchSize, 1));
            }

            if (target.Rank != 2 || target.Shape[0] != batchSize || target.Shape[1] != 1)
            {
                throw new ArgumentException("Expected output shape [batch, 1].", nameof(target));
            }

            return (input, target);
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The operator PredictCore evaluates. The default walked Layers as one chain.
        /// </remarks>
        public override Tensor<T> ForwardForTraining(Tensor<T> input)
            => ForwardOperator(input);

        /// <summary>
        /// Gets metadata about the DeepONet model.
        /// </summary>
        /// <returns>Model metadata.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "LatentDimension", _latentDimension },
                    { "NumSensors", _numSensors },
                    { "BranchModel", _branchNet.GetType().Name },
                    { "TrunkModel", _trunkNet.GetType().Name },
                    { "ParameterCount", GetParameterCount() }
                },
                ModelDataProvider = () => Serialize()
            };
        }

        /// <summary>
        /// Serializes DeepONet-specific data.
        /// </summary>
        /// <param name="writer">Binary writer.</param>


        /// <summary>
        /// Deserializes DeepONet-specific data.
        /// </summary>
        /// <param name="reader">Binary reader.</param>


        private static void ClearNetworkGradients(NeuralNetworkBase<T> network)
        {
            foreach (var layer in network.Layers)
            {
                layer.ClearGradients();
            }
        }

        /// <inheritdoc />
        public override bool SupportsTraining => true;
    }
}

