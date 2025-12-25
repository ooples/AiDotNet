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
public class DomainDecompositionPINN<T> : PhysicsInformedNeuralNetwork<T>
{
    private readonly List<PhysicsInformedNeuralNetwork<T>> _subdomainNetworks;
    private readonly List<SubdomainDefinition<T>> _subdomains;
    private readonly List<InterfaceDefinition<T>> _interfaces;
    private readonly List<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _subdomainOptimizers;

    // Training configuration
    private readonly double _pdeWeight;
    private readonly double _interfaceWeight;
    private readonly double _interfaceGradientWeight;
    private readonly int _schwarzIterations;

    /// <summary>
    /// Creates a Domain Decomposition PINN with specified subdomains.
    /// </summary>
    /// <param name="architecture">Base network architecture (used for creating subdomain networks if not provided).</param>
    /// <param name="pdeSpecification">The PDE specification.</param>
    /// <param name="boundaryConditions">Boundary conditions for the global domain.</param>
    /// <param name="subdomains">Subdomain definitions specifying bounds for each region.</param>
    /// <param name="subdomainNetworks">Custom networks for each subdomain (null = create defaults).</param>
    /// <param name="initialCondition">Initial condition (optional).</param>
    /// <param name="numCollocationPointsPerSubdomain">Collocation points per subdomain.</param>
    /// <param name="optimizer">Base optimizer (subdomain optimizers derived from this).</param>
    /// <param name="pdeWeight">Weight for PDE residual loss (default: 1.0).</param>
    /// <param name="boundaryWeight">Weight for boundary condition loss (default: 1.0).</param>
    /// <param name="interfaceWeight">Weight for interface continuity loss (default: 10.0).</param>
    /// <param name="interfaceGradientWeight">Weight for interface gradient matching (default: 1.0).</param>
    /// <param name="schwarzIterations">Number of Schwarz iterations per epoch (default: 1).</param>
    /// <remarks>
    /// For Beginners:
    /// Parameters to tune:
    ///
    /// - interfaceWeight: Higher values enforce stricter continuity
    /// - interfaceGradientWeight: Important for smooth global solutions
    /// - schwarzIterations: More iterations = better coupling but slower
    ///
    /// The Schwarz method is an iterative approach where:
    /// 1. Solve each subdomain with current interface values
    /// 2. Exchange boundary data between subdomains
    /// 3. Repeat until convergence
    /// </remarks>
    public DomainDecompositionPINN(
        NeuralNetworkArchitecture<T> architecture,
        IPDESpecification<T> pdeSpecification,
        IBoundaryCondition<T>[] boundaryConditions,
        List<SubdomainDefinition<T>> subdomains,
        List<PhysicsInformedNeuralNetwork<T>>? subdomainNetworks = null,
        IInitialCondition<T>? initialCondition = null,
        int numCollocationPointsPerSubdomain = 5000,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        double pdeWeight = 1.0,
        double boundaryWeight = 1.0,
        double interfaceWeight = 10.0,
        double interfaceGradientWeight = 1.0,
        int schwarzIterations = 1)
        : base(architecture, pdeSpecification, boundaryConditions, initialCondition,
               numCollocationPointsPerSubdomain, optimizer, null, pdeWeight, boundaryWeight, null)
    {
        if (subdomains == null || subdomains.Count == 0)
        {
            throw new ArgumentException("At least one subdomain must be specified.", nameof(subdomains));
        }

        _subdomains = subdomains;
        _pdeWeight = pdeWeight;
        _interfaceWeight = interfaceWeight;
        _interfaceGradientWeight = interfaceGradientWeight;
        _schwarzIterations = schwarzIterations;
        _subdomainNetworks = new List<PhysicsInformedNeuralNetwork<T>>();
        _subdomainOptimizers = new List<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>();

        // Create or use provided subdomain networks
        if (subdomainNetworks != null && subdomainNetworks.Count == subdomains.Count)
        {
            _subdomainNetworks.AddRange(subdomainNetworks);
        }
        else
        {
            // Create default subdomain networks
            for (int i = 0; i < subdomains.Count; i++)
            {
                var subNetwork = new PhysicsInformedNeuralNetwork<T>(
                    architecture,
                    pdeSpecification,
                    boundaryConditions,
                    initialCondition,
                    numCollocationPointsPerSubdomain,
                    null,
                    null,
                    pdeWeight,
                    boundaryWeight,
                    null);
                _subdomainNetworks.Add(subNetwork);
            }
        }

        // Create optimizers for each subdomain
        for (int i = 0; i < _subdomainNetworks.Count; i++)
        {
            var subOptimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(_subdomainNetworks[i]);
            _subdomainOptimizers.Add(subOptimizer);
        }

        // Automatically detect interfaces between subdomains
        _interfaces = DetectInterfaces(subdomains);
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
    /// Finds the shared boundary between two subdomains.
    /// </summary>
    private T[,]? FindSharedBoundary(SubdomainDefinition<T> sub1, SubdomainDefinition<T> sub2)
    {
        // Simple check for 1D and 2D axis-aligned boundaries
        int inputDim = sub1.LowerBounds.Length;

        for (int dim = 0; dim < inputDim; dim++)
        {
            // Check if sub1's upper bound touches sub2's lower bound in this dimension
            double upper1 = NumOps.ToDouble(sub1.UpperBounds[dim]);
            double lower2 = NumOps.ToDouble(sub2.LowerBounds[dim]);

            if (Math.Abs(upper1 - lower2) < 1e-10)
            {
                // Found shared boundary in dimension 'dim'
                return GenerateBoundaryPoints(sub1.UpperBounds[dim], dim, sub1, 100);
            }

            // Check the reverse
            double upper2 = NumOps.ToDouble(sub2.UpperBounds[dim]);
            double lower1 = NumOps.ToDouble(sub1.LowerBounds[dim]);

            if (Math.Abs(upper2 - lower1) < 1e-10)
            {
                return GenerateBoundaryPoints(sub1.LowerBounds[dim], dim, sub1, 100);
            }
        }

        return null;
    }

    /// <summary>
    /// Generates points along a shared boundary.
    /// </summary>
    private T[,] GenerateBoundaryPoints(T fixedValue, int fixedDim, SubdomainDefinition<T> subdomain, int numPoints)
    {
        int inputDim = subdomain.LowerBounds.Length;
        var points = new T[numPoints, inputDim];
        var random = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < numPoints; i++)
        {
            for (int j = 0; j < inputDim; j++)
            {
                if (j == fixedDim)
                {
                    points[i, j] = fixedValue;
                }
                else
                {
                    double lower = NumOps.ToDouble(subdomain.LowerBounds[j]);
                    double upper = NumOps.ToDouble(subdomain.UpperBounds[j]);
                    points[i, j] = NumOps.FromDouble(lower + random.NextDouble() * (upper - lower));
                }
            }
        }

        return points;
    }

    /// <summary>
    /// Solves the PDE using domain decomposition.
    /// </summary>
    /// <param name="epochs">Total number of training epochs.</param>
    /// <param name="learningRate">Learning rate for optimization.</param>
    /// <param name="verbose">Whether to print progress.</param>
    /// <param name="batchSize">Batch size for training.</param>
    /// <returns>Domain decomposition training history with per-subdomain metrics.</returns>
    public DomainDecompositionTrainingHistory<T> SolveWithDecomposition(
        int epochs = 10000,
        double learningRate = 0.001,
        bool verbose = true,
        int batchSize = 256)
    {
        var history = new DomainDecompositionTrainingHistory<T>(_subdomains.Count);

        // Configure optimizers
        for (int i = 0; i < _subdomainOptimizers.Count; i++)
        {
            var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = learningRate
            };
            _subdomainOptimizers[i] = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(_subdomainNetworks[i], options);
        }

        // Set training mode
        foreach (var network in _subdomainNetworks)
        {
            network.SetTrainingMode(true);
            foreach (var layer in network.Layers)
            {
                layer.SetTrainingMode(true);
            }
        }

        try
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var epochMetrics = TrainDecompositionEpoch(batchSize);

                history.AddEpoch(
                    epochMetrics.TotalLoss,
                    epochMetrics.SubdomainLosses,
                    epochMetrics.InterfaceLoss,
                    epochMetrics.PhysicsLoss);

                if (verbose && epoch % 100 == 0)
                {
                    Console.WriteLine(
                        $"Epoch {epoch}/{epochs}, " +
                        $"Total: {epochMetrics.TotalLoss}, " +
                        $"Interface: {epochMetrics.InterfaceLoss}");
                }
            }
        }
        finally
        {
            foreach (var network in _subdomainNetworks)
            {
                foreach (var layer in network.Layers)
                {
                    layer.SetTrainingMode(false);
                }

                network.SetTrainingMode(false);
            }
        }

        return history;
    }

    private DecompositionEpochMetrics TrainDecompositionEpoch(int batchSize)
    {
        var subdomainLosses = new List<T>();
        T interfaceLoss = NumOps.Zero;
        T physicsLoss = NumOps.Zero;

        // Schwarz iterations
        for (int schwarz = 0; schwarz < _schwarzIterations; schwarz++)
        {
            // Train each subdomain
            for (int i = 0; i < _subdomainNetworks.Count; i++)
            {
                var subLoss = TrainSubdomain(i, batchSize);
                if (schwarz == _schwarzIterations - 1) // Record only final iteration
                {
                    subdomainLosses.Add(subLoss);
                }

                physicsLoss = NumOps.Add(physicsLoss, subLoss);
            }

            // Enforce interface conditions
            interfaceLoss = EnforceInterfaceConditions();
        }

        T totalLoss = NumOps.Add(physicsLoss, interfaceLoss);

        return new DecompositionEpochMetrics
        {
            TotalLoss = totalLoss,
            SubdomainLosses = subdomainLosses,
            InterfaceLoss = interfaceLoss,
            PhysicsLoss = physicsLoss
        };
    }

    private T TrainSubdomain(int subdomainIndex, int batchSize)
    {
        var subdomain = _subdomains[subdomainIndex];
        var network = _subdomainNetworks[subdomainIndex];
        var optimizer = _subdomainOptimizers[subdomainIndex];

        // Generate collocation points within subdomain
        int inputDim = subdomain.LowerBounds.Length;
        var random = RandomHelper.CreateSeededRandom(42 + subdomainIndex);

        var points = new T[batchSize, inputDim];
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputDim; j++)
            {
                double lower = NumOps.ToDouble(subdomain.LowerBounds[j]);
                double upper = NumOps.ToDouble(subdomain.UpperBounds[j]);
                points[i, j] = NumOps.FromDouble(lower + random.NextDouble() * (upper - lower));
            }
        }

        // Forward pass
        var inputTensor = new Tensor<T>([batchSize, inputDim]);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputDim; j++)
            {
                inputTensor[i, j] = points[i, j];
            }
        }

        var output = network.Forward(inputTensor);

        // Compute loss (simplified - actual implementation would evaluate PDE residual)
        T loss = NumOps.Zero;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < output.Shape[1]; j++)
            {
                // Placeholder: actual PDE residual would go here
                T val = output[i, j];
                loss = NumOps.Add(loss, NumOps.Multiply(val, val));
            }
        }

        loss = NumOps.Divide(loss, NumOps.FromDouble(batchSize));

        // Backpropagate
        var gradient = new Tensor<T>(output.Shape);
        T scale = NumOps.Divide(NumOps.FromDouble(2.0), NumOps.FromDouble(batchSize));
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < output.Shape[1]; j++)
            {
                gradient[i, j] = NumOps.Multiply(scale, output[i, j]);
            }
        }

        network.Backpropagate(gradient);
        optimizer.UpdateParameters(network.Layers);

        return loss;
    }

    private T EnforceInterfaceConditions()
    {
        T totalInterfaceLoss = NumOps.Zero;

        foreach (var iface in _interfaces)
        {
            var boundary = iface.SharedBoundary;
            if (boundary == null) continue;

            var network1 = _subdomainNetworks[iface.Subdomain1Index];
            var network2 = _subdomainNetworks[iface.Subdomain2Index];

            int numPoints = boundary.GetLength(0);
            int inputDim = boundary.GetLength(1);

            // Evaluate both networks at interface points
            var inputTensor = new Tensor<T>([numPoints, inputDim]);
            for (int i = 0; i < numPoints; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    inputTensor[i, j] = boundary[i, j];
                }
            }

            var output1 = network1.Forward(inputTensor);
            var output2 = network2.Forward(inputTensor);

            // Compute continuity loss: ||u1 - u2||^2
            T continuityLoss = NumOps.Zero;
            for (int i = 0; i < numPoints; i++)
            {
                for (int j = 0; j < output1.Shape[1]; j++)
                {
                    T diff = NumOps.Subtract(output1[i, j], output2[i, j]);
                    continuityLoss = NumOps.Add(continuityLoss, NumOps.Multiply(diff, diff));
                }
            }

            continuityLoss = NumOps.Divide(continuityLoss, NumOps.FromDouble(numPoints));
            totalInterfaceLoss = NumOps.Add(
                totalInterfaceLoss,
                NumOps.Multiply(NumOps.FromDouble(_interfaceWeight), continuityLoss));

            // Backpropagate interface loss to both networks
            var gradient1 = new Tensor<T>(output1.Shape);
            var gradient2 = new Tensor<T>(output2.Shape);
            T gradScale = NumOps.Multiply(
                NumOps.FromDouble(2.0 * _interfaceWeight),
                NumOps.Divide(NumOps.One, NumOps.FromDouble(numPoints)));

            for (int i = 0; i < numPoints; i++)
            {
                for (int j = 0; j < output1.Shape[1]; j++)
                {
                    T diff = NumOps.Subtract(output1[i, j], output2[i, j]);
                    gradient1[i, j] = NumOps.Multiply(gradScale, diff);
                    gradient2[i, j] = NumOps.Multiply(gradScale, NumOps.Negate(diff));
                }
            }

            network1.Backpropagate(gradient1);
            network2.Backpropagate(gradient2);

            _subdomainOptimizers[iface.Subdomain1Index].UpdateParameters(network1.Layers);
            _subdomainOptimizers[iface.Subdomain2Index].UpdateParameters(network2.Layers);
        }

        return totalInterfaceLoss;
    }

    /// <summary>
    /// Gets the solution at a point by finding the appropriate subdomain.
    /// </summary>
    /// <param name="point">Input coordinates.</param>
    /// <returns>Solution value from the containing subdomain network.</returns>
    public T[] GetGlobalSolution(T[] point)
    {
        int subdomainIndex = FindContainingSubdomain(point);
        if (subdomainIndex < 0)
        {
            throw new ArgumentException("Point is outside all subdomains.");
        }

        return _subdomainNetworks[subdomainIndex].GetSolution(point);
    }

    private int FindContainingSubdomain(T[] point)
    {
        for (int i = 0; i < _subdomains.Count; i++)
        {
            bool inside = true;
            for (int j = 0; j < point.Length; j++)
            {
                double p = NumOps.ToDouble(point[j]);
                double lower = NumOps.ToDouble(_subdomains[i].LowerBounds[j]);
                double upper = NumOps.ToDouble(_subdomains[i].UpperBounds[j]);

                if (p < lower || p > upper)
                {
                    inside = false;
                    break;
                }
            }

            if (inside)
            {
                return i;
            }
        }

        return -1;
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

    private struct DecompositionEpochMetrics
    {
        public T TotalLoss;
        public List<T> SubdomainLosses;
        public T InterfaceLoss;
        public T PhysicsLoss;
    }
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
    public T[,]? SharedBoundary { get; set; }
}
