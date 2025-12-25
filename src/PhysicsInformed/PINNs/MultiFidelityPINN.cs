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
/// Multi-fidelity Physics-Informed Neural Network for combining data of different accuracy levels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// For Beginners:
/// Multi-fidelity learning combines data from multiple sources with different accuracy levels:
///
/// Low-Fidelity Data (Cheap, Abundant):
/// - Coarse simulations
/// - Simplified physical models
/// - Fast but approximate calculations
/// - Example: 2D simulation of a 3D problem
///
/// High-Fidelity Data (Expensive, Scarce):
/// - Fine-mesh simulations
/// - Physical experiments
/// - High-accuracy calculations
/// - Example: Wind tunnel measurements
///
/// The Multi-Fidelity Approach:
/// 1. Train on abundant low-fidelity data to learn general trends
/// 2. Use scarce high-fidelity data to correct errors
/// 3. Learn the correlation between fidelity levels
/// 4. Enforce physics constraints at all fidelity levels
///
/// Mathematical Model:
/// u_HF(x) = rho(x) * u_LF(x) + delta(x)
///
/// Where:
/// - u_LF(x): Low-fidelity prediction
/// - u_HF(x): High-fidelity prediction
/// - rho(x): Scaling factor (learned)
/// - delta(x): Correction/bias term (learned)
///
/// This implementation uses a nonlinear correlation model where a neural network
/// learns the relationship between fidelity levels.
///
/// References:
/// - Meng, X., and Karniadakis, G.E. "A composite neural network that learns from
///   multi-fidelity data: Application to function approximation and inverse PDE problems"
///   Journal of Computational Physics, 2020.
/// </remarks>
public class MultiFidelityPINN<T> : PhysicsInformedNeuralNetwork<T>
{
    private readonly PhysicsInformedNeuralNetwork<T> _lowFidelityNetwork;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _multiFidelityOptimizer;

    // Multi-fidelity configuration
    private readonly double _lowFidelityWeight;
    private readonly double _highFidelityWeight;
    private readonly double _correlationWeight;
    private readonly bool _freezeLowFidelityAfterPretraining;
    private bool _lowFidelityFrozen;

    // Training data
    private T[,]? _lowFidelityInputs;
    private T[,]? _lowFidelityOutputs;
    private T[,]? _highFidelityInputs;
    private T[,]? _highFidelityOutputs;

    /// <summary>
    /// Creates a Multi-Fidelity PINN with optional custom low-fidelity network.
    /// </summary>
    /// <param name="architecture">Network architecture for the high-fidelity/correlation network.</param>
    /// <param name="pdeSpecification">The PDE specification.</param>
    /// <param name="boundaryConditions">Boundary conditions.</param>
    /// <param name="initialCondition">Initial condition (optional).</param>
    /// <param name="lowFidelityNetwork">Custom low-fidelity network (null = create default).</param>
    /// <param name="numCollocationPoints">Number of collocation points for PDE residual.</param>
    /// <param name="optimizer">Optimizer (null = use Adam with default settings).</param>
    /// <param name="lowFidelityWeight">Weight for low-fidelity data loss (default: 1.0).</param>
    /// <param name="highFidelityWeight">Weight for high-fidelity data loss (default: 10.0 - higher because scarcer).</param>
    /// <param name="correlationWeight">Weight for fidelity correlation loss (default: 1.0).</param>
    /// <param name="pdeWeight">Weight for PDE residual loss (default: 1.0).</param>
    /// <param name="boundaryWeight">Weight for boundary condition loss (default: 1.0).</param>
    /// <param name="freezeLowFidelityAfterPretraining">Whether to freeze low-fidelity network after pretraining (default: true).</param>
    /// <remarks>
    /// For Beginners:
    /// The loss weights control the relative importance of each objective:
    ///
    /// - lowFidelityWeight: How much to fit the cheap/abundant data
    /// - highFidelityWeight: How much to fit the expensive/accurate data (usually higher)
    /// - correlationWeight: How strongly to enforce the fidelity relationship
    /// - pdeWeight: How much to enforce the physics equations
    ///
    /// Typical values:
    /// - More high-fidelity data: Lower highFidelityWeight
    /// - Very noisy low-fidelity data: Lower lowFidelityWeight
    /// - Strong physics constraints: Higher pdeWeight
    /// </remarks>
    public MultiFidelityPINN(
        NeuralNetworkArchitecture<T> architecture,
        IPDESpecification<T> pdeSpecification,
        IBoundaryCondition<T>[] boundaryConditions,
        IInitialCondition<T>? initialCondition = null,
        PhysicsInformedNeuralNetwork<T>? lowFidelityNetwork = null,
        int numCollocationPoints = 10000,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        double lowFidelityWeight = 1.0,
        double highFidelityWeight = 10.0,
        double correlationWeight = 1.0,
        double pdeWeight = 1.0,
        double boundaryWeight = 1.0,
        bool freezeLowFidelityAfterPretraining = true)
        : base(architecture, pdeSpecification, boundaryConditions, initialCondition,
               numCollocationPoints, optimizer, null, pdeWeight, boundaryWeight, null)
    {
        _lowFidelityWeight = lowFidelityWeight;
        _highFidelityWeight = highFidelityWeight;
        _correlationWeight = correlationWeight;
        _freezeLowFidelityAfterPretraining = freezeLowFidelityAfterPretraining;
        _lowFidelityFrozen = false;

        // Create or use provided low-fidelity network
        if (lowFidelityNetwork != null)
        {
            _lowFidelityNetwork = lowFidelityNetwork;
        }
        else
        {
            // Create default low-fidelity network with smaller architecture
            var lfArchitecture = CreateLowFidelityArchitecture(architecture, pdeSpecification);
            _lowFidelityNetwork = new PhysicsInformedNeuralNetwork<T>(
                lfArchitecture,
                pdeSpecification,
                boundaryConditions,
                initialCondition,
                numCollocationPoints / 2, // Fewer collocation points for LF
                null,
                null,
                pdeWeight * 0.5, // Lower physics weight for LF
                boundaryWeight);
        }

        // Create optimizer for this network
        _multiFidelityOptimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
    }

    /// <summary>
    /// Creates a smaller architecture for the low-fidelity network.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateLowFidelityArchitecture(
        NeuralNetworkArchitecture<T> hfArchitecture,
        IPDESpecification<T> pdeSpec)
    {
        // Low-fidelity network typically has fewer layers/neurons
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: pdeSpec.InputDimension,
            outputSize: pdeSpec.OutputDimension);
    }

    /// <summary>
    /// Sets the low-fidelity training data.
    /// </summary>
    /// <param name="inputs">Input coordinates [numSamples, inputDim].</param>
    /// <param name="outputs">Solution values [numSamples, outputDim].</param>
    public void SetLowFidelityData(T[,] inputs, T[,] outputs)
    {
        if (inputs.GetLength(0) != outputs.GetLength(0))
        {
            throw new ArgumentException("Input and output sample counts must match.");
        }

        _lowFidelityInputs = inputs;
        _lowFidelityOutputs = outputs;
    }

    /// <summary>
    /// Sets the high-fidelity training data.
    /// </summary>
    /// <param name="inputs">Input coordinates [numSamples, inputDim].</param>
    /// <param name="outputs">Solution values [numSamples, outputDim].</param>
    public void SetHighFidelityData(T[,] inputs, T[,] outputs)
    {
        if (inputs.GetLength(0) != outputs.GetLength(0))
        {
            throw new ArgumentException("Input and output sample counts must match.");
        }

        _highFidelityInputs = inputs;
        _highFidelityOutputs = outputs;
    }

    /// <summary>
    /// Solves the PDE using multi-fidelity training.
    /// </summary>
    /// <param name="epochs">Total number of training epochs.</param>
    /// <param name="pretrainingEpochs">Epochs to pretrain low-fidelity network (default: epochs/4).</param>
    /// <param name="learningRate">Learning rate for optimization.</param>
    /// <param name="verbose">Whether to print progress.</param>
    /// <param name="batchSize">Batch size for training.</param>
    /// <returns>Multi-fidelity training history with detailed metrics.</returns>
    /// <remarks>
    /// For Beginners:
    /// Multi-fidelity training proceeds in stages:
    ///
    /// Stage 1: Pretrain Low-Fidelity Network
    /// - Train only the low-fidelity network on low-fidelity data
    /// - Goal: Learn general trends from abundant cheap data
    ///
    /// Stage 2: Joint Training
    /// - Train both networks together
    /// - High-fidelity network learns to correct low-fidelity predictions
    /// - Correlation ensures consistency between fidelity levels
    ///
    /// Optional: Freeze Low-Fidelity
    /// - After pretraining, lock the low-fidelity network weights
    /// - Only train the correction/correlation part
    /// - Can improve stability
    /// </remarks>
    public MultiFidelityTrainingHistory<T> SolveMultiFidelity(
        int epochs = 10000,
        int? pretrainingEpochs = null,
        double learningRate = 0.001,
        bool verbose = true,
        int batchSize = 256)
    {
        int actualPretrainingEpochs = pretrainingEpochs ?? epochs / 4;
        var history = new MultiFidelityTrainingHistory<T>();

        if (_lowFidelityInputs == null || _lowFidelityOutputs == null)
        {
            throw new InvalidOperationException(
                "Low-fidelity data must be set before training. Call SetLowFidelityData first.");
        }

        // Stage 1: Pretrain low-fidelity network
        if (verbose)
        {
            Console.WriteLine("Stage 1: Pretraining low-fidelity network...");
        }

        var lfHistory = _lowFidelityNetwork.Solve(
            _lowFidelityInputs,
            _lowFidelityOutputs,
            actualPretrainingEpochs,
            learningRate,
            verbose,
            batchSize);

        // Record pretraining in history
        foreach (var loss in lfHistory.Losses)
        {
            history.AddEpoch(loss, loss, NumOps.Zero, NumOps.Zero, NumOps.Zero);
        }

        // Optionally freeze low-fidelity network
        if (_freezeLowFidelityAfterPretraining)
        {
            _lowFidelityFrozen = true;
            if (verbose)
            {
                Console.WriteLine("Low-fidelity network frozen.");
            }
        }

        // Stage 2: Joint training
        if (verbose)
        {
            Console.WriteLine("Stage 2: Joint multi-fidelity training...");
        }

        // Configure optimizer
        var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            LearningRate = learningRate
        };
        _multiFidelityOptimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this, options);

        SetTrainingMode(true);
        foreach (var layer in Layers)
        {
            layer.SetTrainingMode(true);
        }

        int remainingEpochs = epochs - actualPretrainingEpochs;

        try
        {
            for (int epoch = 0; epoch < remainingEpochs; epoch++)
            {
                var epochMetrics = TrainMultiFidelityEpoch(batchSize);

                history.AddEpoch(
                    epochMetrics.TotalLoss,
                    epochMetrics.LowFidelityLoss,
                    epochMetrics.HighFidelityLoss,
                    epochMetrics.CorrelationLoss,
                    epochMetrics.PhysicsLoss);

                if (verbose && epoch % 100 == 0)
                {
                    Console.WriteLine(
                        $"Epoch {actualPretrainingEpochs + epoch}/{epochs}, " +
                        $"Total: {epochMetrics.TotalLoss}, " +
                        $"LF: {epochMetrics.LowFidelityLoss}, " +
                        $"HF: {epochMetrics.HighFidelityLoss}");
                }
            }
        }
        finally
        {
            foreach (var layer in Layers)
            {
                layer.SetTrainingMode(false);
            }

            SetTrainingMode(false);
        }

        return history;
    }

    private MultiFidelityEpochMetrics TrainMultiFidelityEpoch(int batchSize)
    {
        T totalLoss = NumOps.Zero;
        T lfLoss = NumOps.Zero;
        T hfLoss = NumOps.Zero;
        T corrLoss = NumOps.Zero;
        T physLoss = NumOps.Zero;
        int sampleCount = 0;

        // Train on low-fidelity data
        if (_lowFidelityInputs != null && _lowFidelityOutputs != null)
        {
            int lfCount = _lowFidelityInputs.GetLength(0);
            int inputDim = _lowFidelityInputs.GetLength(1);
            int outputDim = _lowFidelityOutputs.GetLength(1);

            for (int batchStart = 0; batchStart < lfCount; batchStart += batchSize)
            {
                int batchEnd = Math.Min(batchStart + batchSize, lfCount);
                int batchCount = batchEnd - batchStart;

                var batchInput = new Tensor<T>([batchCount, inputDim]);
                var batchTarget = new Tensor<T>([batchCount, outputDim]);

                for (int i = 0; i < batchCount; i++)
                {
                    for (int j = 0; j < inputDim; j++)
                    {
                        batchInput[i, j] = _lowFidelityInputs[batchStart + i, j];
                    }

                    for (int j = 0; j < outputDim; j++)
                    {
                        batchTarget[i, j] = _lowFidelityOutputs[batchStart + i, j];
                    }
                }

                // Low-fidelity prediction
                var lfPrediction = _lowFidelityNetwork.Forward(batchInput);

                // Compute low-fidelity loss
                var lfMse = ComputeMSE(lfPrediction, batchTarget);
                T weightedLfLoss = NumOps.Multiply(NumOps.FromDouble(_lowFidelityWeight), lfMse);
                lfLoss = NumOps.Add(lfLoss, weightedLfLoss);
                sampleCount += batchCount;

                // If not frozen, update low-fidelity network
                if (!_lowFidelityFrozen)
                {
                    var lfGradient = ComputeMSEGradient(lfPrediction, batchTarget);
                    _lowFidelityNetwork.Backpropagate(lfGradient);
                }
            }
        }

        // Train on high-fidelity data
        if (_highFidelityInputs != null && _highFidelityOutputs != null)
        {
            int hfCount = _highFidelityInputs.GetLength(0);
            int inputDim = _highFidelityInputs.GetLength(1);
            int outputDim = _highFidelityOutputs.GetLength(1);

            for (int batchStart = 0; batchStart < hfCount; batchStart += batchSize)
            {
                int batchEnd = Math.Min(batchStart + batchSize, hfCount);
                int batchCount = batchEnd - batchStart;

                var batchInput = new Tensor<T>([batchCount, inputDim]);
                var batchTarget = new Tensor<T>([batchCount, outputDim]);

                for (int i = 0; i < batchCount; i++)
                {
                    for (int j = 0; j < inputDim; j++)
                    {
                        batchInput[i, j] = _highFidelityInputs[batchStart + i, j];
                    }

                    for (int j = 0; j < outputDim; j++)
                    {
                        batchTarget[i, j] = _highFidelityOutputs[batchStart + i, j];
                    }
                }

                // Get low-fidelity prediction at high-fidelity points
                var lfAtHf = _lowFidelityNetwork.Forward(batchInput);

                // High-fidelity prediction (this network learns the correction)
                var hfPrediction = Forward(batchInput);

                // Combined prediction: hf_corrected = lf + hf_correction
                var correctedPrediction = new Tensor<T>(hfPrediction.Shape);
                for (int i = 0; i < batchCount; i++)
                {
                    for (int j = 0; j < outputDim; j++)
                    {
                        correctedPrediction[i, j] = NumOps.Add(lfAtHf[i, j], hfPrediction[i, j]);
                    }
                }

                // High-fidelity loss
                var hfMse = ComputeMSE(correctedPrediction, batchTarget);
                T weightedHfLoss = NumOps.Multiply(NumOps.FromDouble(_highFidelityWeight), hfMse);
                hfLoss = NumOps.Add(hfLoss, weightedHfLoss);

                // Correlation loss: correction should be small where LF is good
                var corrMse = ComputeMSE(hfPrediction, new Tensor<T>(hfPrediction.Shape)); // Target is zero
                T weightedCorrLoss = NumOps.Multiply(NumOps.FromDouble(_correlationWeight * 0.1), corrMse);
                corrLoss = NumOps.Add(corrLoss, weightedCorrLoss);

                // Backpropagate through this network
                var hfGradient = ComputeMSEGradient(correctedPrediction, batchTarget);
                Backpropagate(hfGradient);

                sampleCount += batchCount;
            }
        }

        // Update parameters
        _multiFidelityOptimizer.UpdateParameters(Layers);

        // Compute physics loss using base class Solve approach (on collocation points)
        // This is simplified - full implementation would evaluate PDE residual
        physLoss = NumOps.Zero; // TODO: Add PDE residual evaluation

        totalLoss = NumOps.Add(NumOps.Add(lfLoss, hfLoss), NumOps.Add(corrLoss, physLoss));

        return new MultiFidelityEpochMetrics
        {
            TotalLoss = totalLoss,
            LowFidelityLoss = lfLoss,
            HighFidelityLoss = hfLoss,
            CorrelationLoss = corrLoss,
            PhysicsLoss = physLoss
        };
    }

    private T ComputeMSE(Tensor<T> prediction, Tensor<T> target)
    {
        T sum = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < prediction.Shape[0]; i++)
        {
            for (int j = 0; j < prediction.Shape[1]; j++)
            {
                T diff = NumOps.Subtract(prediction[i, j], target[i, j]);
                sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
                count++;
            }
        }

        return count > 0 ? NumOps.Divide(sum, NumOps.FromDouble(count)) : NumOps.Zero;
    }

    private Tensor<T> ComputeMSEGradient(Tensor<T> prediction, Tensor<T> target)
    {
        var gradient = new Tensor<T>(prediction.Shape);
        int count = prediction.Shape[0] * prediction.Shape[1];
        T scale = NumOps.Divide(NumOps.FromDouble(2.0), NumOps.FromDouble(count));

        for (int i = 0; i < prediction.Shape[0]; i++)
        {
            for (int j = 0; j < prediction.Shape[1]; j++)
            {
                T diff = NumOps.Subtract(prediction[i, j], target[i, j]);
                gradient[i, j] = NumOps.Multiply(scale, diff);
            }
        }

        return gradient;
    }

    /// <summary>
    /// Gets the high-fidelity prediction at a point.
    /// </summary>
    /// <param name="point">Input coordinates.</param>
    /// <returns>High-fidelity solution estimate.</returns>
    public T[] GetHighFidelitySolution(T[] point)
    {
        var inputTensor = new Tensor<T>(new int[] { 1, point.Length });
        for (int i = 0; i < point.Length; i++)
        {
            inputTensor[0, i] = point[i];
        }

        // LF + HF correction
        var lfOutput = _lowFidelityNetwork.Forward(inputTensor);
        var hfCorrection = Forward(inputTensor);

        T[] result = new T[lfOutput.Shape[1]];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = NumOps.Add(lfOutput[0, i], hfCorrection[0, i]);
        }

        return result;
    }

    /// <summary>
    /// Gets the low-fidelity prediction at a point.
    /// </summary>
    /// <param name="point">Input coordinates.</param>
    /// <returns>Low-fidelity solution estimate.</returns>
    public T[] GetLowFidelitySolution(T[] point)
    {
        return _lowFidelityNetwork.GetSolution(point);
    }

    /// <summary>
    /// Gets the correction (difference between fidelity levels) at a point.
    /// </summary>
    /// <param name="point">Input coordinates.</param>
    /// <returns>Fidelity correction values.</returns>
    public T[] GetFidelityCorrection(T[] point)
    {
        return GetSolution(point);
    }

    /// <summary>
    /// Gets the low-fidelity network for external access.
    /// </summary>
    public PhysicsInformedNeuralNetwork<T> LowFidelityNetwork => _lowFidelityNetwork;

    /// <summary>
    /// Gets whether the low-fidelity network is frozen.
    /// </summary>
    public bool IsLowFidelityFrozen => _lowFidelityFrozen;

    /// <summary>
    /// Freezes or unfreezes the low-fidelity network.
    /// </summary>
    /// <param name="frozen">Whether to freeze the network.</param>
    public void SetLowFidelityFrozen(bool frozen)
    {
        _lowFidelityFrozen = frozen;
    }

    private struct MultiFidelityEpochMetrics
    {
        public T TotalLoss;
        public T LowFidelityLoss;
        public T HighFidelityLoss;
        public T CorrelationLoss;
        public T PhysicsLoss;
    }
}
