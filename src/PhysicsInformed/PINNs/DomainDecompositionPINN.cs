using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PhysicsInformed.PINNs;

/// <summary>
/// Domain Decomposition Physics-Informed Neural Network for large-scale problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// For Beginners:
/// Domain decomposition is a strategy for solving large problems by breaking them
/// into smaller, manageable pieces. Think of it like solving a jigsaw puzzle:
/// - Each piece (subdomain) is solved separately
/// - The pieces must fit together at the edges (interface conditions)
///
/// Why Use Domain Decomposition?
/// 1. Memory: Large domains require too much memory for one network
/// 2. Parallelism: Subdomains can be trained independently (partially)
/// 3. Accuracy: Local networks can specialize for local behavior
/// 4. Geometry: Complex domains can be split into simpler shapes
///
/// Types of Decomposition:
///
/// 1. Non-overlapping (used here):
///    |-------|-------|
///    |  D1   |  D2   |  Interface at boundary
///    |-------|-------|
///
/// 2. Overlapping:
///    |----------|
///    |    D1    |---|
///    |----------|   |  D2  |
///              |----|------|
///              Overlap region
///
/// Interface Conditions:
/// At subdomain interfaces, we enforce:
/// 1. Continuity: u_1 = u_2 (solutions match)
/// 2. Flux continuity: du_1/dn = du_2/dn (derivatives match)
///
/// Training Strategy:
/// 1. Train each subdomain network on its local domain
/// 2. Enforce interface conditions at boundaries
/// 3. Iterate until global convergence
///
/// References:
/// - Jagtap, A.D., and Karniadakis, G.E. "Extended Physics-Informed Neural Networks
///   (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework"
///   Communications in Computational Physics, 2020.
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 2, outputSize: 1);
/// var pde = new HeatEquation&lt;float&gt;();
/// // No IBoundaryCondition implementation ships with the library: write your own with
/// // IsOnBoundary, ComputeBoundaryResidual and Name, then put instances in this array.
/// var bc = Array.Empty&lt;IBoundaryCondition&lt;float&gt;&gt;();
/// var subdomains = new List&lt;SubdomainDefinition&lt;float&gt;&gt;
/// {
///     new SubdomainDefinition&lt;float&gt;(new float[] { 0 }, new float[] { 1 }, "left"),
///     new SubdomainDefinition&lt;float&gt;(new float[] { 1 }, new float[] { 2 }, "right")
/// };
/// var ddPinn = new DomainDecompositionPINN&lt;float&gt;(architecture, pde, bc, subdomains);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Science)]
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.PhysicsInformed)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework", "https://doi.org/10.4208/cicp.OA-2020-0164", Year = 2020, Authors = "Ameya D. Jagtap, George Em Karniadakis")]
public partial class DomainDecompositionPINN<T> : PhysicsInformedNeuralNetwork<T>
{
    private readonly DomainDecompositionPINNOptions _options;

    /// <summary>
    /// Builds nothing: a domain-decomposition model's solution IS its subdomain networks.
    /// </summary>
    /// <remarks>
    /// The base constructor used to build a full global PINN here, and Predict and Train ran it while
    /// SolveWithDecomposition trained the subdomain networks - so the public entry points answered from
    /// an untrained network unrelated to the decomposition. The subdomain networks' parameters are
    /// surfaced by the generated component hook, once each.
    /// </remarks>
    protected override void InitializeLayers()
    {
    }

    private readonly List<PhysicsInformedNeuralNetwork<T>> _subdomainNetworks;
    private readonly List<SubdomainDefinition<T>> _subdomains;
    private readonly List<InterfaceDefinition<T>> _interfaces;

    // Training configuration
    private readonly double _pdeWeight;
    private readonly double _interfaceWeight;
    private readonly double _interfaceGradientWeight;
    private readonly int _schwarzIterations;

    // Caller-supplied boundary/initial training data for SolveWithDecomposition; not model state.
    [Scratch]
    private Tensor<T>? _trainingInputs;
    [Scratch]
    private Tensor<T>? _trainingOutputs;

    // The loss terms of the most recent training step, for the training history.
    private T _lastInterfaceLoss;
    private T _lastPhysicsLoss;
    private readonly List<T> _lastSubdomainLosses = new List<T>();

    /// <summary>
    /// Creates a Domain Decomposition PINN with specified subdomains.
    /// </summary>
    /// <param name="architecture">Base network architecture (used for creating subdomain networks if not provided).</param>
    /// <param name="pdeSpecification">The PDE specification.</param>
    /// <param name="boundaryConditions">Boundary conditions for the global domain.</param>
    /// <param name="subdomains">Subdomain definitions specifying bounds for each region.</param>
    /// <param name="subdomainNetworks">Custom networks for each subdomain (null = create defaults).</param>
    /// <param name="initialCondition">Initial condition (optional).</param>
    /// <param name="optimizer">The optimizer for the joint XPINN objective.</param>
    /// <param name="options">The loss weights and collocation settings.</param>
    /// <remarks>
    /// For Beginners:
    /// Parameters to tune:
    ///
    /// - InterfaceWeight: the weight W_I on the average-solution continuity term at the interfaces
    /// - InterfaceGradientWeight: the weight W_IF on the residual-continuity term at the interfaces
    /// - SchwarzIterations: optimizer steps per SolveWithDecomposition epoch
    /// </remarks>
    public DomainDecompositionPINN(
        NeuralNetworkArchitecture<T> architecture,
        IPDESpecification<T> pdeSpecification,
        IBoundaryCondition<T>[] boundaryConditions,
        List<SubdomainDefinition<T>> subdomains,
        List<PhysicsInformedNeuralNetwork<T>>? subdomainNetworks = null,
        IInitialCondition<T>? initialCondition = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        DomainDecompositionPINNOptions? options = null)
        : this(options ?? new DomainDecompositionPINNOptions(), architecture, pdeSpecification,
               boundaryConditions, subdomains, subdomainNetworks, initialCondition, optimizer)
    {
    }

    /// <summary>
    /// Initializes the model from an already-resolved options instance.
    /// </summary>
    private DomainDecompositionPINN(
        DomainDecompositionPINNOptions options,
        NeuralNetworkArchitecture<T> architecture,
        IPDESpecification<T> pdeSpecification,
        IBoundaryCondition<T>[] boundaryConditions,
        List<SubdomainDefinition<T>> subdomains,
        List<PhysicsInformedNeuralNetwork<T>>? subdomainNetworks,
        IInitialCondition<T>? initialCondition,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
        : base(architecture, pdeSpecification, boundaryConditions, initialCondition, optimizer,
               new PhysicsInformedNeuralNetworkOptions
               {
                   NumCollocationPoints = options.NumCollocationPointsPerSubdomain,
                   PdeWeight = options.PdeWeight,
                   BoundaryWeight = options.BoundaryWeight,
               })
    {
        _options = options;
        Options = _options;
        _lastInterfaceLoss = NumOps.Zero;
        _lastPhysicsLoss = NumOps.Zero;

        if (subdomains == null || subdomains.Count == 0)
        {
            throw new ArgumentException("At least one subdomain must be specified.", nameof(subdomains));
        }

        options.Validate();
        _subdomains = subdomains;
        _pdeWeight = options.PdeWeight;
        _interfaceWeight = options.InterfaceWeight;
        _interfaceGradientWeight = options.InterfaceGradientWeight;
        _schwarzIterations = options.SchwarzIterations;
        _subdomainNetworks = new List<PhysicsInformedNeuralNetwork<T>>();

        // Create or use provided subdomain networks
        if (subdomainNetworks != null && subdomainNetworks.Count != subdomains.Count)
        {
            // This used to fall through to default networks, silently discarding the caller's.
            throw new ArgumentException(
                $"{subdomainNetworks.Count} subdomain networks were given for {subdomains.Count} subdomains; " +
                "pass one network per subdomain, or none to create defaults.",
                nameof(subdomainNetworks));
        }

        if (subdomainNetworks != null)
        {
            _subdomainNetworks.AddRange(subdomainNetworks);
        }
        else
        {
            // A custom layer list is a set of layer INSTANCES. Handing it to several default subdomain
            // networks would make them all share one set of weights - one network, not several.
            if (architecture.Layers is not null && architecture.Layers.Count > 0 && subdomains.Count > 1)
            {
                throw new ArgumentException(
                    "An architecture with custom layers cannot seed several subdomain networks: they would " +
                    "share the same layer instances. Pass one network per subdomain through subdomainNetworks.",
                    nameof(subdomainNetworks));
            }

            // Create default subdomain networks
            for (int i = 0; i < subdomains.Count; i++)
            {
                var subNetwork = new PhysicsInformedNeuralNetwork<T>(
                    architecture,
                    pdeSpecification,
                    boundaryConditions,
                    initialCondition,
                    options: new PhysicsInformedNeuralNetworkOptions
                    {
                        NumCollocationPoints = options.NumCollocationPointsPerSubdomain,
                        PdeWeight = options.PdeWeight,
                        BoundaryWeight = options.BoundaryWeight,
                    });
                _subdomainNetworks.Add(subNetwork);
            }
        }

        // XPINN enforces each subdomain's residual on points inside that subdomain. The sub-networks sample
        // the unit cube; map each one's points onto its subdomain's box.
        for (int i = 0; i < _subdomainNetworks.Count; i++)
        {
            var network = _subdomainNetworks[i];
            network.SetCollocationPoints(MapUnitPointsToBox(network.CollocationPoints, _subdomains[i]));
        }

        // Automatically detect interfaces between subdomains
        _interfaces = DetectInterfaces(subdomains);
    }

    /// <summary>
    /// Maps points of the unit cube [N, d] onto the subdomain's box: lower + u * (upper - lower).
    /// </summary>
    private Tensor<T> MapUnitPointsToBox(Tensor<T> unitPoints, SubdomainDefinition<T> subdomain)
    {
        int dims = subdomain.LowerBounds.Length;
        if (unitPoints.Shape[1] != dims)
        {
            throw new ArgumentException(
                $"Subdomain bounds have {dims} dimensions; the PDE has {unitPoints.Shape[1]}.");
        }

        var lower = new Tensor<T>(subdomain.LowerBounds, new[] { 1, dims });
        var width = Engine.TensorSubtract(new Tensor<T>(subdomain.UpperBounds, new[] { 1, dims }), lower);
        return Engine.TensorAdd(Engine.TensorMultiply(unitPoints, width), lower);
    }

    /// <summary>
    /// Detects interfaces between adjacent subdomains.
    /// </summary>
    private List<InterfaceDefinition<T>> DetectInterfaces(List<SubdomainDefinition<T>> subdomains)
    {
        var interfaces = new List<InterfaceDefinition<T>>();

        for (int i = 0; i < subdomains.Count; i++)
        {
            for (int j = i + 1; j < subdomains.Count; j++)
            {
                var shared = FindSharedBoundary(subdomains[i], subdomains[j]);
                if (shared != null)
                {
                    interfaces.Add(new InterfaceDefinition<T>
                    {
                        Subdomain1Index = i,
                        Subdomain2Index = j,
                        SharedBoundary = shared
                    });
                }
            }
        }

        return interfaces;
    }

    /// <summary>
    /// Finds the shared boundary between two subdomains: the first axis along which one box's upper face is the
    /// other's lower face.
    /// </summary>
    /// <remarks>
    /// The faces are compared exactly in T. Subdomain bounds are the caller's own values, so a shared face has
    /// equal coordinates; the old comparison through double with a 1e-10 tolerance was not meaningful for every T.
    /// </remarks>
    private Tensor<T>? FindSharedBoundary(SubdomainDefinition<T> sub1, SubdomainDefinition<T> sub2)
    {
        int dims = sub1.LowerBounds.Length;
        var upper1 = new Tensor<T>(sub1.UpperBounds, new[] { dims });
        var lower1 = new Tensor<T>(sub1.LowerBounds, new[] { dims });
        var upper2 = new Tensor<T>(sub2.UpperBounds, new[] { dims });
        var lower2 = new Tensor<T>(sub2.LowerBounds, new[] { dims });

        var forward = Engine.TensorEquals(upper1, lower2).ToArray();
        var backward = Engine.TensorEquals(upper2, lower1).ToArray();
        T zero = NumOps.Zero;
        int forwardAxis = Array.FindIndex(forward, flag => !flag.Equals(zero));
        int backwardAxis = Array.FindIndex(backward, flag => !flag.Equals(zero));

        // The original scan checked both directions axis by axis and took the first axis that matched either.
        if (forwardAxis >= 0 && (backwardAxis < 0 || forwardAxis <= backwardAxis))
        {
            return GenerateBoundaryPoints(sub1.UpperBounds[forwardAxis], forwardAxis, sub1, 100);
        }

        if (backwardAxis >= 0)
        {
            return GenerateBoundaryPoints(sub1.LowerBounds[backwardAxis], backwardAxis, sub1, 100);
        }

        return null;
    }

    /// <summary>
    /// Generates points on a face of a subdomain: uniform in the box, with one coordinate fixed.
    /// </summary>
    private Tensor<T> GenerateBoundaryPoints(T fixedValue, int fixedDim, SubdomainDefinition<T> subdomain, int numPoints)
    {
        int inputDim = subdomain.LowerBounds.Length;
        var random = RandomHelper.CreateSeededRandom(42);
        var unit = Engine.TensorRandomUniformRange<T>(new[] { numPoints, inputDim }, NumOps.Zero, NumOps.One, random.Next());
        var inBox = MapUnitPointsToBox(unit, subdomain);

        // Replace column fixedDim by the face coordinate.
        var faceAxis = Engine.TensorSlice(Engine.TensorEye<T>(inputDim), new[] { fixedDim, 0 }, new[] { 1, inputDim });
        var keep = Engine.TensorSubtract(Tensor<T>.CreateDefault(new[] { 1, inputDim }, NumOps.One), faceAxis);
        return Engine.TensorAdd(
            Engine.TensorMultiply(inBox, keep),
            Engine.TensorMultiplyScalar(Engine.TensorTile(faceAxis, new[] { numPoints, 1 }), fixedValue));
    }

    /// <summary>
    /// Sets the training data for <see cref="SolveWithDecomposition"/>: boundary, initial and measured values.
    /// </summary>
    /// <param name="inputs">Input coordinates [numSamples, inputDim].</param>
    /// <param name="outputs">Solution values [numSamples, outputDim].</param>
    public void SetTrainingData(Tensor<T> inputs, Tensor<T> outputs)
    {
        Guard.NotNull(inputs);
        Guard.NotNull(outputs);
        if (inputs.Shape[0] != outputs.Shape[0])
        {
            throw new ArgumentException("Input and output sample counts must match.");
        }

        _trainingInputs = inputs;
        _trainingOutputs = outputs;
    }

    /// <summary>
    /// Solves the PDE using domain decomposition.
    /// </summary>
    /// <param name="epochs">Total number of training epochs.</param>
    /// <param name="learningRate">Learning rate for optimization.</param>
    /// <param name="verbose">Whether to print progress.</param>
    /// <param name="batchSize">Unused; every step evaluates all collocation and interface points.</param>
    /// <returns>Domain decomposition training history with per-subdomain metrics.</returns>
    /// <remarks>
    /// Each epoch takes <c>SchwarzIterations</c> steps of the joint XPINN objective (see
    /// <see cref="ComputeTrainingLoss"/>) on the data set with <see cref="SetTrainingData"/>. The sub-networks used
    /// to be trained one by one on a batch of zero inputs with zero targets, with no interface terms at all.
    /// </remarks>
    public DomainDecompositionTrainingHistory<T> SolveWithDecomposition(
        int epochs = 10000,
        double learningRate = 0.001,
        bool verbose = true,
        int batchSize = 256)
    {
        if (_trainingInputs is null || _trainingOutputs is null)
        {
            throw new InvalidOperationException(
                "Training data must be set before training. Call SetTrainingData with the boundary/initial data first.");
        }

        var history = new DomainDecompositionTrainingHistory<T>(_subdomains.Count);
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int step = 0; step < _schwarzIterations; step++)
            {
                Train(_trainingInputs, _trainingOutputs);
            }

            history.AddEpoch(
                GetLastLoss(),
                new List<T>(_lastSubdomainLosses),
                _lastInterfaceLoss,
                _lastPhysicsLoss);

            if (verbose && epoch % 100 == 0)
            {
                Console.WriteLine(
                    $"Epoch {epoch}/{epochs}, " +
                    $"Total: {GetLastLoss()}, " +
                    $"Interface: {_lastInterfaceLoss}");
            }
        }

        return history;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// The XPINN objective of Jagtap and Karniadakis (2020), Eq. 3.3, summed over the subdomains q: the data misfit
    /// of the decomposed solution, the residual MSE_F of each sub-network on its own collocation points, and for
    /// each interface between q and q+ the average-solution continuity
    /// W_I * (mean |u_q - {{u}}|^2 + mean |u_q+ - {{u}}|^2), {{u}} = (u_q + u_q+) / 2, and the residual continuity
    /// W_IF * mean |F_q - F_q+|^2. W_I is InterfaceWeight and W_IF is InterfaceGradientWeight.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ComputeTrainingLoss(Tensor<T> dataInputs, Tensor<T> dataTargets, Tensor<T> collocationPoints)
    {
        var diff = Engine.TensorSubtract(Forward(dataInputs), dataTargets);
        var loss = MeanOf(Engine.TensorMultiply(diff, diff));

        _lastSubdomainLosses.Clear();
        Tensor<T>? physics = null;
        foreach (var network in _subdomainNetworks)
        {
            var residualLoss = network.ComputePdeLossAt(network.CollocationPoints);
            _lastSubdomainLosses.Add(residualLoss.AsSpan()[0]);
            physics = physics is null ? residualLoss : Engine.TensorAdd(physics, residualLoss);
        }

        if (physics is not null)
        {
            _lastPhysicsLoss = physics.AsSpan()[0];
            loss = Engine.TensorAdd(loss, physics);
        }

        Tensor<T>? interfaceLoss = null;
        T half = NumOps.FromDouble(0.5);
        foreach (var iface in _interfaces)
        {
            if (iface.SharedBoundary is not { } points)
            {
                continue;
            }

            var first = _subdomainNetworks[iface.Subdomain1Index];
            var second = _subdomainNetworks[iface.Subdomain2Index];
            var firstResidual = first.ComputeResidualAt(points, out var firstOutputs);
            var secondResidual = second.ComputeResidualAt(points, out var secondOutputs);

            var average = Engine.TensorMultiplyScalar(Engine.TensorAdd(firstOutputs, secondOutputs), half);
            var firstGap = Engine.TensorSubtract(firstOutputs, average);
            var secondGap = Engine.TensorSubtract(secondOutputs, average);
            var continuity = Engine.TensorAdd(
                MeanOf(Engine.TensorMultiply(firstGap, firstGap)),
                MeanOf(Engine.TensorMultiply(secondGap, secondGap)));
            var residualGap = Engine.TensorSubtract(firstResidual, secondResidual);
            var term = Engine.TensorAdd(
                Engine.TensorMultiplyScalar(continuity, NumOps.FromDouble(_interfaceWeight)),
                Engine.TensorMultiplyScalar(MeanOf(Engine.TensorMultiply(residualGap, residualGap)), NumOps.FromDouble(_interfaceGradientWeight)));
            interfaceLoss = interfaceLoss is null ? term : Engine.TensorAdd(interfaceLoss, term);
        }

        if (interfaceLoss is not null)
        {
            _lastInterfaceLoss = interfaceLoss.AsSpan()[0];
            loss = Engine.TensorAdd(loss, interfaceLoss);
        }

        return loss;
    }

    private Tensor<T> MeanOf(Tensor<T> values)
        => Engine.ReduceMean(values, Enumerable.Range(0, values.Rank).ToArray(), keepDims: false);

    /// <summary>
    /// Evaluates the decomposed solution: each row is answered by the network of the subdomain that
    /// contains it (the first match, bounds inclusive), exactly as GetGlobalSolution chooses.
    /// </summary>
    /// <remarks>
    /// Ownership is decided with engine comparisons against each subdomain's box and applied with a select, so the
    /// evaluation stays on the gradient tape and training reaches exactly the network that owns each row. Predict,
    /// GetSolution and the PDE residual all come through here.
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        bool singlePoint = input.Shape.Length == 1;
        var rows = singlePoint ? Engine.Reshape(input, new[] { 1, input.Shape[0] }) : input;
        if (rows.Shape.Length != 2)
        {
            throw new ArgumentException("Expected points as [count, dimensions] or a single point.", nameof(input));
        }

        int count = rows.Shape[0];
        if (count == 0)
        {
            throw new ArgumentException("There are no points to evaluate.", nameof(input));
        }

        // Later subdomains first, so the first containing subdomain is the last select applied and wins.
        Tensor<T>? solution = null;
        Tensor<T>? covered = null;
        for (int s = _subdomainNetworks.Count - 1; s >= 0; s--)
        {
            var inside = ContainmentMask(rows, _subdomains[s]);
            var output = _subdomainNetworks[s].Forward(rows);
            var owner = Engine.TensorTile(inside, new[] { 1, output.Shape[1] });
            solution = Engine.TensorWhere(owner, output, solution ?? new Tensor<T>(output.Shape.ToArray()));
            covered = covered is null ? inside : Engine.TensorMax(covered, inside);
        }

        if (solution is null || covered is null)
        {
            throw new ArgumentException("There are no points to evaluate.", nameof(input));
        }

        if (NumOps.LessThan(Engine.TensorSum(covered), NumOps.FromDouble(count)))
        {
            throw new ArgumentException("A point lies outside every subdomain.", nameof(input));
        }

        return singlePoint ? Engine.Reshape(solution, new[] { solution.Length }) : solution;
    }

    /// <summary>
    /// 1 for each row inside the subdomain's box (bounds inclusive), 0 otherwise, as [count, 1].
    /// </summary>
    private Tensor<T> ContainmentMask(Tensor<T> rows, SubdomainDefinition<T> subdomain)
    {
        int count = rows.Shape[0];
        int dims = rows.Shape[1];
        if (subdomain.LowerBounds.Length != dims)
        {
            throw new ArgumentException($"Points have {dims} dimensions; the subdomain has {subdomain.LowerBounds.Length}.");
        }

        var lower = Engine.TensorTile(new Tensor<T>(subdomain.LowerBounds, new[] { 1, dims }), new[] { count, 1 });
        var upper = Engine.TensorTile(new Tensor<T>(subdomain.UpperBounds, new[] { 1, dims }), new[] { count, 1 });
        var withinAxis = Engine.TensorMultiply(
            Engine.TensorGreaterEqual(rows, lower),
            Engine.TensorLessEqual(rows, upper));
        var axesInside = Engine.ReduceSum(withinAxis, new[] { 1 }, keepDims: true);
        return Engine.TensorEquals(axesInside, NumOps.FromDouble(dims));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The decomposed solution - the function Predict and GetSolution evaluate. The default walked
    /// Layers, which this model does not use.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    /// <summary>
    /// Gets the solution at a point by finding the appropriate subdomain.
    /// </summary>
    /// <param name="point">Input coordinates.</param>
    /// <returns>Solution value from the containing subdomain network.</returns>
    public Vector<T> GetGlobalSolution(Vector<T> point)
    {
        Guard.NotNull(point);
        return GetSolution(point);
    }

    /// <summary>
    /// Gets the number of subdomains.
    /// </summary>
    public int SubdomainCount => _subdomains.Count;

    /// <summary>
    /// Gets a specific subdomain network.
    /// </summary>
    /// <param name="index">Subdomain index.</param>
    /// <returns>The subdomain PINN.</returns>
    public PhysicsInformedNeuralNetwork<T> GetSubdomainNetwork(int index)
    {
        if (index < 0 || index >= _subdomainNetworks.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        return _subdomainNetworks[index];
    }

    /// <summary>
    /// Gets all subdomain definitions.
    /// </summary>
    public IReadOnlyList<SubdomainDefinition<T>> Subdomains => _subdomains.AsReadOnly();

    /// <summary>
    /// Gets all interface definitions.
    /// </summary>
    public IReadOnlyList<InterfaceDefinition<T>> Interfaces => _interfaces.AsReadOnly();
}

/// <summary>
/// Defines a subdomain for domain decomposition.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class SubdomainDefinition<T>
{
    /// <summary>
    /// Lower bounds for each dimension.
    /// </summary>
    public T[] LowerBounds { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Upper bounds for each dimension.
    /// </summary>
    public T[] UpperBounds { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Optional name for the subdomain.
    /// </summary>
    public string? Name { get; set; }

    /// <summary>
    /// Creates a subdomain definition.
    /// </summary>
    /// <param name="lowerBounds">Lower bounds for each dimension.</param>
    /// <param name="upperBounds">Upper bounds for each dimension.</param>
    /// <param name="name">Optional subdomain name.</param>
    public SubdomainDefinition(T[] lowerBounds, T[] upperBounds, string? name = null)
    {
        if (lowerBounds.Length != upperBounds.Length)
        {
            throw new ArgumentException("Lower and upper bounds must have the same dimension.");
        }

        LowerBounds = lowerBounds;
        UpperBounds = upperBounds;
        Name = name;
    }

    /// <summary>
    /// Default constructor for serialization.
    /// </summary>
    public SubdomainDefinition()
    {
    }
}

/// <summary>
/// Defines an interface between two subdomains.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class InterfaceDefinition<T>
{
    /// <summary>
    /// Index of the first subdomain.
    /// </summary>
    public int Subdomain1Index { get; set; }

    /// <summary>
    /// Index of the second subdomain.
    /// </summary>
    public int Subdomain2Index { get; set; }

    /// <summary>
    /// Points on the shared boundary [numPoints, inputDim].
    /// </summary>
    public Tensor<T>? SharedBoundary { get; set; }
}
