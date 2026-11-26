using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.PhysicsInformed.PINNs
{
    /// <summary>
    /// Implements a Multi-Scale Physics-Informed Neural Network for solving PDEs with multiple scales.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Multi-scale problems are challenging because features span vastly different sizes.
    /// A single neural network struggles to capture both large-scale trends and fine details.
    ///
    /// Solution: Multi-Scale PINN
    /// Uses multiple sub-networks, each specialized for a different scale:
    /// - Coarse network: Captures large-scale, smooth variations
    /// - Fine network(s): Capture small-scale details and fluctuations
    ///
    /// Architecture:
    /// Input (x,t) → [Coarse Net] → u_coarse(x,t)
    ///            → [Fine Net 1] → u_fine1(x,t)
    ///            → [Fine Net 2] → u_fine2(x,t)
    ///            → ...
    ///
    /// Total solution: u(x,t) = u_coarse + u_fine1 + u_fine2 + ...
    ///
    /// Training Strategy:
    /// 1. Progressive Training: Train coarse first, then add finer scales
    /// 2. Simultaneous Training: Train all scales together with adaptive weights
    /// 3. Alternating Training: Alternate between scales during training
    ///
    /// Key Features:
    /// - Fourier feature encoding at different frequencies for each scale
    /// - Adaptive loss weighting to balance scale contributions
    /// - Scale coupling terms to ensure consistency
    /// - Progressive activation of finer scales during training
    ///
    /// Applications:
    /// - Turbulence modeling (large eddies + small vortices)
    /// - Composite materials (macroscopic + fiber-scale behavior)
    /// - Multi-physics problems (thermal + mechanical + chemical)
    /// - Climate modeling (global + regional + local scales)
    /// </remarks>
    /// <example>
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     InputType.OneDimensional, NeuralNetworkTaskType.Regression,
    ///     inputSize: 2, outputSize: 1);
    ///
    /// // The same physics at a coarse and a fine length scale, coupled where they overlap.
    /// var pde = new MultiScaleReactionDiffusion&lt;double&gt;(
    ///     coarseDiffusivity: 1.0,
    ///     fineDiffusivity: 0.01);
    ///
    /// // Held at zero on both edges of the domain.
    /// var boundaryConditions = new IBoundaryCondition&lt;double&gt;[]
    /// {
    ///     new DirichletBoundaryCondition&lt;double&gt;(boundaryValue: 0.0, lowerBound: 0.0, upperBound: 1.0)
    /// };
    ///
    /// var msPinn = new MultiScalePINN&lt;double&gt;(architecture, pde, boundaryConditions);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Science)]
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.PhysicsInformed)]
    [ModelTask(ModelTask.Regression)]
    [ModelComplexity(ModelComplexity.VeryHigh)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Multi-scale Physics-Informed Neural Networks for Stiff Chemical Kinetics", "https://doi.org/10.1021/acs.jpca.1c05102", Year = 2022, Authors = "Weiqi Ji, Wai Tong Chung, Sili Deng")]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    public partial class MultiScalePINN<T> : NeuralNetworkBase<T>
    {
        private readonly MultiScalePINNOptions _options;

        private readonly IMultiScalePDE<T> _multiScalePDE;
        private readonly List<NeuralNetworkBase<T>> _scaleNetworks;
        private readonly IBoundaryCondition<T>[] _boundaryConditions;
        private readonly IInitialCondition<T>? _initialCondition;
        private readonly MultiScaleTrainingOptions<T> _trainingOptions;

        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
        private readonly bool _usesDefaultOptimizer;
        private double _defaultOptimizerLearningRate;

        internal IReadOnlyList<NeuralNetworkBase<T>> ScaleNetworks => _scaleNetworks;

        private Tensor<T> _collocationPoints;
        private readonly int _numCollocationPointsPerScale;
        private T[] _scaleWeights;
        private int _currentActiveScales;

        /// <summary>
        /// Gets the number of scales in this multi-scale PINN.
        /// </summary>
        public int NumberOfScales => _multiScalePDE.NumberOfScales;

        /// <summary>
        /// Gets the characteristic length scales.
        /// </summary>
        public T[] ScaleCharacteristicLengths => _multiScalePDE.ScaleCharacteristicLengths;

        /// <summary>
        /// Initializes a new instance of the Multi-Scale PINN.
        /// </summary>
        /// <param name="architecture">The base neural network architecture.</param>
        /// <param name="multiScalePDE">The multi-scale PDE specification.</param>
        /// <param name="boundaryConditions">Boundary conditions for the problem.</param>
        /// <param name="initialCondition">Initial condition for time-dependent problems (optional).</param>
        /// <param name="trainingOptions">Options for multi-scale training.</param>
        /// <param name="optimizer">Optimization algorithm.</param>
        /// <remarks>
        /// For Beginners:
        ///
        /// This creates separate neural networks for each scale:
        /// - Coarse network: Wider/shallower, larger learning rate
        /// - Fine networks: Deeper, smaller learning rate, Fourier features
        ///
        /// The collocation points are generated at different densities:
        /// - Fewer points for coarse scale (smooth variations)
        /// - More points for fine scales (detailed features)
        /// </remarks>
        public MultiScalePINN(NeuralNetworkArchitecture<T> architecture,
            IMultiScalePDE<T> multiScalePDE,
            IBoundaryCondition<T>[] boundaryConditions,
            IInitialCondition<T>? initialCondition = null,
            MultiScaleTrainingOptions<T>? trainingOptions = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            MultiScalePINNOptions? options = null)
            : base(architecture, NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), 1.0)
        {
            options ??= new MultiScalePINNOptions();
            _options = options;
            Options = _options;

            Guard.NotNull(multiScalePDE);
            _multiScalePDE = multiScalePDE;
            Guard.NotNull(boundaryConditions);
            _boundaryConditions = boundaryConditions;
            _initialCondition = initialCondition;
            _numCollocationPointsPerScale = options.NumCollocationPointsPerScale;
            _trainingOptions = trainingOptions ?? new MultiScaleTrainingOptions<T>();

            _usesDefaultOptimizer = optimizer == null;
            _defaultOptimizerLearningRate = 0.001;
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _defaultOptimizerLearningRate,
                    UseAMSGrad = false,
                });
            AdoptConfiguredOptimizer(_optimizer);

            // Initialize scale weights
            _scaleWeights = new T[_multiScalePDE.NumberOfScales];
            for (int i = 0; i < _scaleWeights.Length; i++)
            {
                _scaleWeights[i] = _multiScalePDE.GetScaleLossWeight(i);
            }

            // Create sub-networks for each scale
            _scaleNetworks = new List<NeuralNetworkBase<T>>();
            CreateScaleNetworks(architecture);

            // Start with just coarse scale if sequential training
            _currentActiveScales = _trainingOptions.UseSequentialScaleTraining ? 1 : _multiScalePDE.NumberOfScales;

            InitializeLayers();
            _collocationPoints = new Tensor<T>(new[] { 0, _multiScalePDE.InputDimension });
            GenerateCollocationPoints();
        }

        /// <summary>
        /// Creates the neural networks for each scale.
        /// </summary>
        private void CreateScaleNetworks(NeuralNetworkArchitecture<T> baseArchitecture)
        {
            for (int scale = 0; scale < _multiScalePDE.NumberOfScales; scale++)
            {
                int outputDim = _multiScalePDE.GetScaleOutputDimension(scale);
                int? scaleSeed = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextSeedOrNull();
                var outerScope = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.CaptureScope();
                try
                {
                    if (scaleSeed.HasValue)
                    {
                        AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.ResetForModelConstruction(scaleSeed);
                    }

                    // Adjust network size based on scale
                    // Finer scales typically need deeper networks
                    var scaleArchitecture = CreateScaleArchitecture(baseArchitecture, scale, outputDim, scaleSeed);
                    var scaleNetwork = new FeedForwardNeuralNetwork<T>(scaleArchitecture);

                    _scaleNetworks.Add(scaleNetwork);
                }
                finally
                {
                    AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.RestoreScope(outerScope);
                }
            }
        }

        /// <summary>
        /// Creates architecture for a specific scale.
        /// </summary>
        private NeuralNetworkArchitecture<T> CreateScaleArchitecture(
            NeuralNetworkArchitecture<T> baseArchitecture,
            int scaleIndex,
            int outputDimension,
            int? scaleSeed = null)
        {
            // Finer scales use deeper smooth (Tanh) PINN sub-networks so second- and higher-order PDE derivatives
            // do not vanish (FeedForwardNeuralNetwork's default ReLU layers have zero second derivative everywhere).
            int depthMultiplier = 1 + scaleIndex;
            var templateArchitecture = new NeuralNetworkArchitecture<T>(
                baseArchitecture.InputType,
                baseArchitecture.TaskType,
                baseArchitecture.Complexity,
                baseArchitecture.InputSize,
                baseArchitecture.InputHeight,
                baseArchitecture.InputWidth,
                baseArchitecture.InputDepth,
                outputDimension,
                null,
                baseArchitecture.ShouldReturnFullSequence);
            int hiddenLayerCount = 2 * depthMultiplier;
            const int hiddenWidth = 32;
            IActivationFunction<T> hiddenActivation = new AiDotNet.ActivationFunctions.TanhActivation<T>();
            IActivationFunction<T> outputActivation = new AiDotNet.ActivationFunctions.IdentityActivation<T>();
            var scaleLayers = new List<ILayer<T>>(hiddenLayerCount + 1);
            for (int h = 0; h < hiddenLayerCount; h++)
            {
                scaleLayers.Add(new AiDotNet.NeuralNetworks.Layers.DenseLayer<T>(hiddenWidth, hiddenActivation));
            }

            scaleLayers.Add(new AiDotNet.NeuralNetworks.Layers.DenseLayer<T>(outputDimension, outputActivation));
            var inputShape = templateArchitecture.GetInputShape();
            if (inputShape is not null && inputShape.Length > 0 && inputShape.All(d => d > 0))
            {
                int[] running = inputShape;
                foreach (var layer in scaleLayers)
                {
                    if (layer is AiDotNet.NeuralNetworks.Layers.LayerBase<T> lb && !lb.IsShapeResolved)
                    {
                        lb.ResolveFromShape(running);
                    }

                    running = layer.GetOutputShape();
                }
            }

            var scaleArchitecture = new NeuralNetworkArchitecture<T>(
                baseArchitecture.InputType,
                baseArchitecture.TaskType,
                baseArchitecture.Complexity,
                baseArchitecture.InputSize,
                baseArchitecture.InputHeight,
                baseArchitecture.InputWidth,
                baseArchitecture.InputDepth,
                outputDimension,
                scaleLayers,
                baseArchitecture.ShouldReturnFullSequence);
            if (scaleSeed.HasValue)
            {
                scaleArchitecture.RandomSeed = scaleSeed.Value;
            }

            return scaleArchitecture;
        }

        /// <inheritdoc/>
        protected override void InitializeLayers()
        {
            ClearLayers();
            // Register all scale network layers for proper parameter management
            foreach (var scaleNet in _scaleNetworks)
            {
                Layers.AddRange(scaleNet.Layers);
            }
        }

        /// <summary>
        /// Generates collocation points for each scale.
        /// </summary>
        private void GenerateCollocationPoints()
        {
            using var _ = TensorArena.Suspend();
            int inputDim = _multiScalePDE.InputDimension;
            int totalPoints = _numCollocationPointsPerScale * _multiScalePDE.NumberOfScales;

            // Points in [0, 1]^d.
            _collocationPoints = Engine.TensorRandomUniformRange<T>(
                new[] { totalPoints, inputDim }, NumOps.Zero, NumOps.One, RandomHelper.CreateSeededRandom(42).Next());
        }

        /// <summary>
        /// Forward pass through all scale networks: the sum of the active scales' outputs.
        /// </summary>
        /// <param name="input">Input coordinates, one point [inputDim] or a batch [batch, inputDim].</param>
        /// <returns>Combined output from all active scales, [outputDim] or [batch, outputDim].</returns>
        /// <remarks>
        /// Prediction and training both evaluate this. Training used to walk Layers - every scale
        /// network's layers back to back - as one chain, feeding one scale's output into the next
        /// scale's input; and this sum was built with scalar copies the gradient tape cannot see
        /// through. It is now an engine sum over the scale networks' own layers.
        /// </remarks>
        public Tensor<T> Forward(Tensor<T> input)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            bool singlePoint = input.Rank == 1;
            var rows = singlePoint ? Engine.Reshape(input, new[] { 1, input.Shape[0] }) : input;
            if (rows.Rank != 2)
            {
                throw new ArgumentException("Input must be [inputDim] or [batch, inputDim].", nameof(input));
            }

            int batchSize = rows.Shape[0];
            int outputDim = _multiScalePDE.OutputDimension;
            Tensor<T>? combined = null;

            for (int scale = 0; scale < _currentActiveScales; scale++)
            {
                var scaleOutput = RunLayers(_scaleNetworks[scale], rows);
                int width = scaleOutput.Shape[scaleOutput.Shape.Length - 1];
                if (width > outputDim)
                {
                    scaleOutput = Engine.TensorSlice(scaleOutput, new[] { 0, 0 }, new[] { batchSize, outputDim });
                }
                else if (width < outputDim)
                {
                    var padding = new Tensor<T>(new[] { batchSize, outputDim - width });
                    scaleOutput = Engine.TensorConcatenate(new[] { scaleOutput, padding }, axis: 1);
                }

                combined = combined is null ? scaleOutput : Engine.TensorAdd(combined, scaleOutput);
            }

            if (combined is null)
            {
                throw new InvalidOperationException("No scale network is active.");
            }

            return singlePoint ? Engine.Reshape(combined, new[] { outputDim }) : combined;
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

        /// <inheritdoc/>
        /// <remarks>
        /// The sum Forward evaluates. The default walked Layers as one chain across the scales.
        /// </remarks>
        public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

        /// <inheritdoc/>
        public override Tensor<T> ForwardWithMemory(Tensor<T> input)
        {
            // Set all scale networks to training mode
            foreach (var network in _scaleNetworks.Take(_currentActiveScales))
            {
                network.SetTrainingMode(true);
            }

            return Forward(input);
        }

        /// <summary>
        /// Solves the multi-scale PDE using physics-informed training.
        /// </summary>
        /// <param name="epochs">Number of training epochs.</param>
        /// <param name="learningRate">Learning rate.</param>
        /// <param name="verbose">Whether to print progress.</param>
        /// <returns>Training history.</returns>
        public TrainingHistory<T> Solve(int epochs = 1000, double learningRate = 0.001, bool verbose = true)
        {
            ThrowIfCannotTrain();
            using var reentrancyGuard = AcquireTrainSentinel();
            var history = new TrainingHistory<T>();

            if (_usesDefaultOptimizer
                && ReferenceEquals(GetOrCreateBaseOptimizer(), _optimizer)
                && !_defaultOptimizerLearningRate.Equals(learningRate))
            {
                var options = new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = learningRate,
                    UseAMSGrad = false,
                };
                _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options);
                _defaultOptimizerLearningRate = learningRate;
                SetBaseTrainOptimizer(_optimizer);
            }

            SetTrainingMode(true);

            try
            {
                if (_trainingOptions.UseSequentialScaleTraining)
                {
                    // Progressive training: coarse to fine
                    SolveSequential(epochs, history, verbose);
                }
                else
                {
                    // Simultaneous training of all scales
                    SolveSimultaneous(epochs, history, verbose);
                }
            }
            finally
            {
                SetTrainingMode(false);
            }

            return history;
        }

        /// <summary>
        /// Sequential training: train coarse scale first, then progressively add finer scales.
        /// </summary>
        private void SolveSequential(int epochs, TrainingHistory<T> history, bool verbose)
        {
            int pretrainingEpochs = _trainingOptions.ScalePretrainingEpochs;
            int epochsPerScale = epochs / _multiScalePDE.NumberOfScales;

            for (int scale = 0; scale < _multiScalePDE.NumberOfScales; scale++)
            {
                _currentActiveScales = scale + 1;

                if (verbose)
                {
                    Console.WriteLine($"\n=== Training Scale {scale + 1}/{_multiScalePDE.NumberOfScales} ===");
                }

                int scaleEpochs = (scale == _multiScalePDE.NumberOfScales - 1)
                    ? epochs - scale * epochsPerScale
                    : epochsPerScale;

                for (int epoch = 0; epoch < scaleEpochs; epoch++)
                {
                    T loss = TrainEpoch();
                    history.AddEpoch(loss);

                    if (verbose && epoch % 50 == 0)
                    {
                        Console.WriteLine($"Scale {scale + 1}, Epoch {epoch}/{scaleEpochs}, Loss: {loss}");
                    }
                }
            }
        }

        /// <summary>
        /// Simultaneous training: train all scales together with adaptive weighting.
        /// </summary>
        private void SolveSimultaneous(int epochs, TrainingHistory<T> history, bool verbose)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                T loss = TrainEpoch();
                history.AddEpoch(loss);

                // Update scale weights adaptively
                if (_trainingOptions.UseAdaptiveScaleWeighting && epoch > 0 && epoch % 100 == 0)
                {
                    UpdateAdaptiveScaleWeights();
                }

                if (verbose && epoch % 50 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}/{epochs}, Loss: {loss}");
                }
            }
        }

        /// <summary>
        /// Performs one training epoch.
        /// </summary>
        private T TrainEpoch()
        {
            int numPoints = _collocationPoints.Shape[0];
            int inputDim = _collocationPoints.Shape[1];
            int batchSize = Math.Min(256, numPoints);
            var losses = new List<T>();

            for (int batchStart = 0; batchStart < numPoints; batchStart += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, numPoints - batchStart);
                Tensor<T> batch;
                using (TensorArena.Suspend())
                {
                    batch = Engine.TensorSlice(_collocationPoints, new[] { batchStart, 0 }, new[] { actualBatchSize, inputDim });
                }

                losses.Add(StepMultiScaleBatch(batch));
            }

            var lossTensor = Tensor<T>.FromVector(Vector<T>.FromList(losses));
            return Engine.ReduceMean(lossTensor, new[] { 0 }, keepDims: false).AsSpan()[0];
        }

        private T StepMultiScaleBatch(Tensor<T> batch)
        {
            ThrowIfCannotTrain();
            var stepOptimizer = ResolveCustomStepOptimizer(_usesDefaultOptimizer ? null : _optimizer);
            var layerGraphBreak = LayerGraphBreakReason();
            string? graphBreakReason = !SupportsFusedCompiledTraining
                ? "model opts out of fused compiled training (per-step host state)"
                : layerGraphBreak;

            if (!SupportsFusedCompiledTraining && FusedSession.IsCommitted)
            {
                throw Training.TapeTrainingStepper<T>.CommittedPlanCannotContinue(
                    new InvalidOperationException("MultiScalePINN.Solve cannot continue a committed fused plan after opting out of fused compiled training."));
            }

            double valueClip = GradientValueClip;
            if (valueClip > 0.0 && FusedSession.IsCommitted)
            {
                throw Training.TapeTrainingStepper<T>.CommittedPlanCannotContinue(
                    new InvalidOperationException("MultiScalePINN.Solve cannot switch a committed fused plan to model value clipping."));
            }

            EnsureLayerRandomSeedsWired();
            var parameters = CollectModelTrainableTensors();
            Tensor<T> placeholderTarget;
            using (TensorArena.Suspend())
            {
                placeholderTarget = new Tensor<T>(new[] { 1, 1, 1, 1, 1, Math.Max(1, _currentActiveScales) });
            }

            Tensor<T> Objective()
            {
                EnsureLayerRandomSeedsWired();
                var loss = ComputeMultiScaleLoss(batch);
                if (loss is null || loss.Length != 1)
                {
                    throw new InvalidOperationException("MultiScalePINN objective must return a scalar loss.");
                }

                return loss;
            }

            bool canTryFused = SupportsFusedCompiledTraining
                && valueClip <= 0.0
                && (layerGraphBreak is null || FusedSession.IsCommitted);
            var request = new Training.FusedTrainingStepRequest<T>
            {
                Layers = Training.TapeTrainingStep<T>.CollectTrainableLayers(Layers),
                Input = batch,
                Target = placeholderTarget,
                Forward = _ => Objective(),
                ComputeLoss = static (loss, _) => loss,
                Optimizer = stepOptimizer,
                ExtraParameters = null,
                Selection = null,
                ModelGradientClip = MaxGradNormValue,
                ExcludesParametersFromWeightDecay = () => DecayExcludedParameters().Count > 0,
                RegularizeGradients = GetDistributedOptimizerRegularizer(stepOptimizer),
                GraphBreakReason = graphBreakReason,
                ClosureGraphBreakPolicy = layerGraphBreak is not null && FusedSession.IsCommitted
                    ? Training.ClosureGraphBreakPolicy.UseConfiguredOptimizer
                    : Training.ClosureGraphBreakPolicy.Disallow,
                OnGradients = ScatterFusedGradients,
                OnEagerGradients = PublishEagerFallbackGradients,
                ClosureObjective = canTryFused ? Objective : null,
            };

            MarkTrainMutationStarted();
            T lossValue = NumOps.Zero;
            var outcome = !canTryFused || parameters.Count == 0
                ? Training.FusedStepOutcome.Declined
                : FusedSession.TryStep(request, out lossValue);
            if (outcome == Training.FusedStepOutcome.Stepped)
            {
                InvalidateWeightCachesAfterSuccessfulWeightUpdate(updatedOnDevice: true);
            }
            else
            {
                if (outcome == Training.FusedStepOutcome.CommittedFailure)
                {
                    FusedSession.DropAfterCommittedFailure();
                }

                using (TensorArena.Create())
                {
                    lossValue = valueClip > 0.0
                        ? StepMultiScaleBatchValueClippedEager(batch, placeholderTarget, Objective, stepOptimizer, valueClip)
                        : Training.TapeTrainingStepper<T>.EagerStep(request, this);
                }

                InvalidateWeightCachesAfterSuccessfulWeightUpdate();
            }

            double numericLoss = NumOps.ToDouble(lossValue);
            if (double.IsNaN(numericLoss) || double.IsInfinity(numericLoss))
            {
                throw new InvalidOperationException("MultiScalePINN objective must return a finite scalar loss.");
            }

            LastLoss = lossValue;
            return lossValue;
        }

        private T StepMultiScaleBatchValueClippedEager(
            Tensor<T> batch,
            Tensor<T> placeholderTarget,
            Func<Tensor<T>> objective,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> stepOptimizer,
            double valueClip)
        {
            foreach (var layer in Training.TapeTrainingStep<T>.CollectTrainableLayers(Layers))
            {
                layer.ZeroGrad();
            }

            Dictionary<Tensor<T>, Tensor<T>>? stepGradients = null;
            T lower = NumOps.FromDouble(-valueClip);
            T upper = NumOps.FromDouble(valueClip);
            return Training.TapeTrainingStepper<T>.EagerTapeStep(
                objective,
                CollectModelTrainableTensors,
                context =>
                {
                    stepOptimizer.Step(context);
                    StepSchedulerIfSupported(stepOptimizer);
                },
                batch,
                placeholderTarget,
                (_, _) => objective(),
                static (loss, _) => loss,
                new Training.TapeTrainingStepper<T>.EagerStepHooks
                {
                    Owner = this,
                    OnGradients = grads =>
                    {
                        stepGradients = grads as Dictionary<Tensor<T>, Tensor<T>>;
                        PublishEagerFallbackGradients(grads);
                    },
                    RegularizeGradients = GetDistributedOptimizerRegularizer(stepOptimizer),
                    ModelGradientClip = MaxGradNormValue,
                    Optimizer = stepOptimizer,
                    ReportsRetainedOutputOutcome = Training.TapeTrainingStepper<T>.ReportsRetainedOutputOutcome(stepOptimizer),
                    BeforeUpdate = () =>
                    {
                        if (stepGradients is null || stepGradients.Count == 0)
                        {
                            return;
                        }

                        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
                        {
                            foreach (var g in stepGradients.Values)
                            {
                                if (g is not null && g.Length > 0)
                                {
                                    Engine.TensorClampInPlace(g, lower, upper);
                                }
                            }
                        }

                        PublishEagerFallbackGradients(stepGradients);
                    },
                });
        }

        /// <summary>
        /// The multi-scale physics-informed loss at a batch of collocation points: sum_s w_s mean(R_s^2) plus
        /// the coupling weight times mean(C^2) for each adjacent pair of active scales.
        /// </summary>
        /// <remarks>
        /// Scale outputs and spatial/temporal derivatives up to order 3 are propagated analytically with pure
        /// engine tensor operations over each scale MLP's live weights, so both compiled closure plans and the
        /// eager gradient tape differentiate through the identical graph while leaving genuinely unreferenced
        /// output biases structurally absent from the gradient domain.
        /// </remarks>
        private Tensor<T> ComputeMultiScaleLoss(Tensor<T> points)
        {
            int order = _multiScalePDE is PDEs.PDESpecificationBase<T> spec ? Math.Max(1, spec.DerivativeOrder) : 2;
            var outputs = new Tensor<T>[_currentActiveScales];
            var derivatives = new PDEDerivatives<T>[_currentActiveScales];
            for (int scale = 0; scale < _currentActiveScales; scale++)
            {
                var network = _scaleNetworks[scale];
                outputs[scale] = EvaluateScaleNetwork(network, points, order, out derivatives[scale]);
            }

            Tensor<T>? total = null;
            for (int scale = 0; scale < _currentActiveScales; scale++)
            {
                var residual = _multiScalePDE.ComputeScaleResidual(scale, points, outputs[scale], derivatives[scale]);
                var meanSq = MeanSquare(residual);
                Tensor<T> weightLeaf;
                using (TensorArena.Suspend())
                {
                    weightLeaf = Tensor<T>.CreateDefault(meanSq._shape, _scaleWeights[scale]);
                }

                var term = Engine.TensorMultiply(meanSq, weightLeaf);
                total = total is null ? term : Engine.TensorAdd(total, term);
            }

            T couplingWeight = _trainingOptions.CouplingWeight ?? NumOps.One;
            for (int coarse = 0; coarse < _currentActiveScales - 1; coarse++)
            {
                int fine = coarse + 1;
                var coupling = _multiScalePDE.ComputeScaleCoupling(
                    coarse, fine, points, outputs[coarse], outputs[fine], derivatives[coarse], derivatives[fine]);
                var meanSq = MeanSquare(coupling);
                Tensor<T> couplingLeaf;
                using (TensorArena.Suspend())
                {
                    couplingLeaf = Tensor<T>.CreateDefault(meanSq._shape, couplingWeight);
                }

                var term = Engine.TensorMultiply(meanSq, couplingLeaf);
                total = total is null ? term : Engine.TensorAdd(total, term);
            }

            if (total is null)
            {
                throw new InvalidOperationException("No scale network is active.");
            }

            return total;
        }

        internal Tensor<T> EvaluateScaleNetwork(
            NeuralNetworkBase<T> network,
            Tensor<T> points,
            int order,
            out PDEDerivatives<T> derivatives)
        {
            if (order == 0)
            {
                derivatives = new PDEDerivatives<T>();
                return RunLayers(network, points);
            }

            if (!CanEvaluateScaleJet(network) || order > 3)
            {
                return PinnAutodiff<T>.Evaluate(x => RunLayers(network, x), points, order, out derivatives);
            }

            int n = points.Shape[0];
            int d = points.Shape[1];
            Tensor<T> h = points;

            Tensor<T> eyeBase;
            using (TensorArena.Suspend())
            {
                var eyeData = new T[d * d];
                for (int i = 0; i < d; i++)
                {
                    eyeData[i * d + i] = NumOps.One;
                }

                eyeBase = new Tensor<T>(eyeData, new[] { 1, d, d });
            }

            Tensor<T> jacobian = n == 1 ? eyeBase : Engine.TensorTile(eyeBase, new[] { n, 1, 1 });
            Tensor<T>? hessian = null;
            Tensor<T>? third = null;
            T two = NumOps.FromDouble(2.0);
            T minusTwo = NumOps.FromDouble(-2.0);
            T six = NumOps.FromDouble(6.0);

            foreach (var layer in network.Layers)
            {
                var dense = (AiDotNet.NeuralNetworks.Layers.DenseLayer<T>)layer;
                int inWidth = h.Shape[1];
                if (!dense.IsShapeResolved)
                {
                    dense.ResolveFromShape(new[] { inWidth });
                }

                var weights = dense.GetWeights();
                var biases = dense.GetBiases();
                int outWidth = weights.Shape[1];

                var pre = Engine.TensorMatMul(h, weights);
                if (biases is { Length: > 0 })
                {
                    var biasRow = Engine.TensorTile(Engine.Reshape(biases, new[] { 1, outWidth }), new[] { n, 1 });
                    pre = Engine.TensorAdd(pre, biasRow);
                }

                // First-order linear pushforward: G[n, p, i] = sum_q W[q, p] * J[n, q, i].
                var jFlat = Engine.Reshape(Engine.TensorPermute(jacobian, new[] { 0, 2, 1 }), new[] { n * d, inWidth });
                var g = Engine.TensorPermute(
                    Engine.Reshape(Engine.TensorMatMul(jFlat, weights), new[] { n, d, outWidth }),
                    new[] { 0, 2, 1 });

                Tensor<T>? s = null;
                if (order >= 2 && hessian is not null)
                {
                    var hFlat = Engine.Reshape(Engine.TensorPermute(hessian, new[] { 0, 2, 3, 1 }), new[] { n * d * d, inWidth });
                    s = Engine.TensorPermute(
                        Engine.Reshape(Engine.TensorMatMul(hFlat, weights), new[] { n, d, d, outWidth }),
                        new[] { 0, 3, 1, 2 });
                }

                Tensor<T>? u = null;
                if (order >= 3 && third is not null)
                {
                    var kFlat = Engine.Reshape(Engine.TensorPermute(third, new[] { 0, 2, 3, 4, 1 }), new[] { n * d * d * d, inWidth });
                    u = Engine.TensorPermute(
                        Engine.Reshape(Engine.TensorMatMul(kFlat, weights), new[] { n, d, d, d, outWidth }),
                        new[] { 0, 4, 1, 2, 3 });
                }

                if (dense.ScalarActivation is AiDotNet.ActivationFunctions.TanhActivation<T>)
                {
                    var act = Engine.Tanh(pre);
                    var actSq = Engine.TensorMultiply(act, act);
                    Tensor<T> ones;
                    using (TensorArena.Suspend())
                    {
                        ones = Tensor<T>.CreateDefault(actSq._shape, NumOps.One);
                    }

                    var sigma1 = Engine.TensorSubtract(ones, actSq);
                    var sigma1J = Engine.TensorTile(Engine.Reshape(sigma1, new[] { n, outWidth, 1 }), new[] { 1, 1, d });
                    var nextJacobian = Engine.TensorMultiply(g, sigma1J);

                    Tensor<T>? nextHessian = null;
                    Tensor<T>? sigma2 = null;
                    Tensor<T>? gg = null;
                    if (order >= 2)
                    {
                        sigma2 = Engine.TensorMultiplyScalar(Engine.TensorMultiply(act, sigma1), minusTwo);
                        var gLeft = Engine.TensorTile(Engine.Reshape(g, new[] { n, outWidth, d, 1 }), new[] { 1, 1, 1, d });
                        var gRight = Engine.TensorTile(Engine.Reshape(g, new[] { n, outWidth, 1, d }), new[] { 1, 1, d, 1 });
                        gg = Engine.TensorMultiply(gLeft, gRight);
                        var sigma2H = Engine.TensorTile(Engine.Reshape(sigma2, new[] { n, outWidth, 1, 1 }), new[] { 1, 1, d, d });
                        nextHessian = Engine.TensorMultiply(gg, sigma2H);
                        if (s is not null)
                        {
                            var sigma1H = Engine.TensorTile(Engine.Reshape(sigma1, new[] { n, outWidth, 1, 1 }), new[] { 1, 1, d, d });
                            nextHessian = Engine.TensorAdd(nextHessian, Engine.TensorMultiply(s, sigma1H));
                        }
                    }

                    Tensor<T>? nextThird = null;
                    if (order >= 3 && sigma2 is not null && gg is not null)
                    {
                        Tensor<T> twos;
                        using (TensorArena.Suspend())
                        {
                            twos = Tensor<T>.CreateDefault(actSq._shape, two);
                        }

                        var sixActSq = Engine.TensorMultiplyScalar(actSq, six);
                        var sigma3 = Engine.TensorMultiply(sigma1, Engine.TensorSubtract(sixActSq, twos));
                        var ggLeft = Engine.TensorTile(Engine.Reshape(gg, new[] { n, outWidth, d, d, 1 }), new[] { 1, 1, 1, 1, d });
                        var gThird = Engine.TensorTile(Engine.Reshape(g, new[] { n, outWidth, 1, 1, d }), new[] { 1, 1, d, d, 1 });
                        var ggg = Engine.TensorMultiply(ggLeft, gThird);
                        var sigma3K = Engine.TensorTile(Engine.Reshape(sigma3, new[] { n, outWidth, 1, 1, 1 }), new[] { 1, 1, d, d, d });
                        nextThird = Engine.TensorMultiply(ggg, sigma3K);

                        if (s is not null)
                        {
                            var sIJ = Engine.TensorTile(Engine.Reshape(s, new[] { n, outWidth, d, d, 1 }), new[] { 1, 1, 1, 1, d });
                            var sIK = Engine.TensorTile(Engine.Reshape(s, new[] { n, outWidth, d, 1, d }), new[] { 1, 1, 1, d, 1 });
                            var sJK = Engine.TensorTile(Engine.Reshape(s, new[] { n, outWidth, 1, d, d }), new[] { 1, 1, d, 1, 1 });
                            var gK = gThird;
                            var gJ = Engine.TensorTile(Engine.Reshape(g, new[] { n, outWidth, 1, d, 1 }), new[] { 1, 1, d, 1, d });
                            var gI = Engine.TensorTile(Engine.Reshape(g, new[] { n, outWidth, d, 1, 1 }), new[] { 1, 1, 1, d, d });
                            var sgSum = Engine.TensorAdd(
                                Engine.TensorAdd(Engine.TensorMultiply(sIJ, gK), Engine.TensorMultiply(sIK, gJ)),
                                Engine.TensorMultiply(sJK, gI));
                            var sigma2K = Engine.TensorTile(Engine.Reshape(sigma2, new[] { n, outWidth, 1, 1, 1 }), new[] { 1, 1, d, d, d });
                            nextThird = Engine.TensorAdd(nextThird, Engine.TensorMultiply(sgSum, sigma2K));
                        }

                        if (u is not null)
                        {
                            var sigma1K = Engine.TensorTile(Engine.Reshape(sigma1, new[] { n, outWidth, 1, 1, 1 }), new[] { 1, 1, d, d, d });
                            nextThird = Engine.TensorAdd(nextThird, Engine.TensorMultiply(u, sigma1K));
                        }
                    }

                    h = act;
                    jacobian = nextJacobian;
                    hessian = nextHessian;
                    third = nextThird;
                }
                else
                {
                    h = pre;
                    jacobian = g;
                    if (order >= 2)
                    {
                        if (s is not null)
                        {
                            hessian = s;
                        }
                        else
                        {
                            using (TensorArena.Suspend())
                            {
                                hessian = new Tensor<T>(new[] { n, outWidth, d, d });
                            }
                        }
                    }
                    else
                    {
                        hessian = null;
                    }

                    if (order >= 3)
                    {
                        if (u is not null)
                        {
                            third = u;
                        }
                        else
                        {
                            using (TensorArena.Suspend())
                            {
                                third = new Tensor<T>(new[] { n, outWidth, d, d, d });
                            }
                        }
                    }
                    else
                    {
                        third = null;
                    }
                }
            }

            derivatives = new PDEDerivatives<T>
            {
                FirstDerivatives = jacobian,
                SecondDerivatives = order >= 2 ? hessian : null,
                ThirdDerivatives = order >= 3 ? third : null,
            };
            return h;
        }

        private static bool CanEvaluateScaleJet(NeuralNetworkBase<T> network)
        {
            foreach (var layer in network.Layers)
            {
                if (layer is not AiDotNet.NeuralNetworks.Layers.DenseLayer<T> dense || dense.VectorActivation is not null)
                {
                    return false;
                }

                if (dense.ScalarActivation is not (null
                    or AiDotNet.ActivationFunctions.IdentityActivation<T>
                    or AiDotNet.ActivationFunctions.TanhActivation<T>))
                {
                    return false;
                }
            }

            return network.Layers.Count > 0;
        }

        private Tensor<T> MeanSquare(Tensor<T> values)
        {
            return Engine.ReduceMean(Engine.TensorMultiply(values, values), Enumerable.Range(0, values.Rank).ToArray(), keepDims: false);
        }

        /// <summary>
        /// Updates scale weights adaptively based on gradient magnitudes.
        /// </summary>
        /// <remarks>
        /// Inverse magnitude weighting: w_s = sum_r |g_r| / |g_s|, so scales with larger gradients get smaller
        /// weights. A scale with a zero gradient keeps weight sum_r |g_r| instead of dividing by a fixed epsilon,
        /// which underflows to zero in low precision.
        /// </remarks>
        private void UpdateAdaptiveScaleWeights()
        {
            var squaredNorms = new List<T>(_currentActiveScales);
            for (int scale = 0; scale < _currentActiveScales; scale++)
            {
                var gradients = _scaleNetworks[scale].GetGradients();
                squaredNorms.Add(Engine.DotProduct(gradients, gradients));
            }

            var magnitudes = Engine.TensorSqrt(Tensor<T>.FromVector(Vector<T>.FromList(squaredNorms)));
            T totalMagnitude = Engine.ReduceSum(magnitudes, new[] { 0 }, keepDims: false).AsSpan()[0];
            if (NumOps.Equals(totalMagnitude, NumOps.Zero))
            {
                return;
            }

            var ones = Tensor<T>.CreateDefault(magnitudes.Shape.ToArray(), NumOps.One);
            var safe = Engine.TensorWhere(Engine.TensorEquals(magnitudes, NumOps.Zero), ones, magnitudes);
            var weights = Engine.TensorMultiplyScalar(Engine.TensorReciprocal(safe), totalMagnitude).ToArray();
            Array.Copy(weights, _scaleWeights, weights.Length);
        }

        /// <inheritdoc/>
        protected override Tensor<T> PredictCore(Tensor<T> input)
        {
            SetTrainingMode(false);
            return Forward(input);
        }

    // UpdateParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
        // ParameterCount and GetParameters were overridden here to fold the scale sub-networks
        // directly. InitializeLayers already does Layers.AddRange(scaleNet.Layers) for every scale,
        // and each scale network is a FeedForwardNeuralNetwork whose parameters ARE its layers, so
        // the base walk over Layers covers exactly the same tensors in exactly the same order.
        // GetGradients below still folds the sub-networks, and stays consistent with that order.

        /// <inheritdoc/>
        public override Vector<T> GetGradients()
        {
            var allGradients = new List<Vector<T>>();

            foreach (var network in _scaleNetworks)
            {
                allGradients.Add(network.GetGradients());
            }

            return Vector<T>.Concatenate(allGradients.ToArray());
        }

        // ParameterCount summed _scaleNetworks.GetParameterCount() -- the SHADOWING accessor, whose
        // result depends on the static type of the reference. Deleted with GetParameters above.

        /// <inheritdoc/>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "NetworkType", "MultiScalePINN" },
                    { "NumberOfScales", _multiScalePDE.NumberOfScales },
                    { "ActiveScales", _currentActiveScales },
                    { "PDEName", _multiScalePDE.Name },
                    { "ParameterCount", ParameterCount }
                },
                ModelDataProvider = () => Serialize()
            };
        }

        /// <inheritdoc/>


        /// <inheritdoc/>


        /// <inheritdoc/>
        public override bool SupportsTraining => true;
    }
}
