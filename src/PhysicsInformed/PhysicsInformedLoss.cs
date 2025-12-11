using System;
using System.Numerics;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed
{
    /// <summary>
    /// Loss function for Physics-Informed Neural Networks (PINNs).
    /// Combines data loss, PDE residual loss, boundary condition loss, and initial condition loss.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Traditional neural networks learn from data alone. Physics-Informed Neural Networks (PINNs)
    /// additionally enforce that the solution satisfies physical laws (PDEs) and constraints.
    ///
    /// This loss function has multiple components:
    /// 1. Data Loss: Measures how well predictions match observed data points
    /// 2. PDE Residual Loss: Measures how much the PDE is violated at collocation points
    /// 3. Boundary Loss: Ensures boundary conditions are satisfied
    /// 4. Initial Condition Loss: Ensures initial conditions are satisfied (for time-dependent problems)
    ///
    /// The total loss is a weighted sum:
    /// L_total = λ_data * L_data + λ_pde * L_pde + λ_bc * L_bc + λ_ic * L_ic
    ///
    /// Why This Works:
    /// By minimizing this loss, the network learns to:
    /// - Fit the available data
    /// - Obey physical laws everywhere (not just at data points)
    /// - Satisfy boundary and initial conditions
    /// This often requires far less data than traditional deep learning!
    ///
    /// Key Innovation:
    /// PINNs can solve PDEs in regions where we have NO data, as long as the physics is known.
    /// </remarks>
    public class PhysicsInformedLoss<T> : ILossFunction<T> where T : struct, INumber<T>
    {
        private readonly IPDESpecification<T>? _pdeSpecification;
        private readonly IBoundaryCondition<T>[]? _boundaryConditions;
        private readonly IInitialCondition<T>? _initialCondition;

        // Loss weights
        private readonly T _dataWeight;
        private readonly T _pdeWeight;
        private readonly T _boundaryWeight;
        private readonly T _initialWeight;

        /// <summary>
        /// Initializes a new instance of the Physics-Informed loss function.
        /// </summary>
        /// <param name="pdeSpecification">The PDE that the solution must satisfy.</param>
        /// <param name="boundaryConditions">Boundary conditions for the problem.</param>
        /// <param name="initialCondition">Initial condition for time-dependent problems.</param>
        /// <param name="dataWeight">Weight for data loss component.</param>
        /// <param name="pdeWeight">Weight for PDE residual loss component.</param>
        /// <param name="boundaryWeight">Weight for boundary condition loss.</param>
        /// <param name="initialWeight">Weight for initial condition loss.</param>
        public PhysicsInformedLoss(
            IPDESpecification<T>? pdeSpecification = null,
            IBoundaryCondition<T>[]? boundaryConditions = null,
            IInitialCondition<T>? initialCondition = null,
            T? dataWeight = null,
            T? pdeWeight = null,
            T? boundaryWeight = null,
            T? initialWeight = null)
        {
            _pdeSpecification = pdeSpecification;
            _boundaryConditions = boundaryConditions;
            _initialCondition = initialCondition;

            // Default weights
            _dataWeight = dataWeight ?? T.One;
            _pdeWeight = pdeWeight ?? T.One;
            _boundaryWeight = boundaryWeight ?? T.One;
            _initialWeight = initialWeight ?? T.One;
        }

        /// <summary>
        /// Gets the name of the loss function.
        /// </summary>
        public string Name => "Physics-Informed Loss";

        /// <summary>
        /// Computes the total physics-informed loss.
        /// </summary>
        /// <param name="predictions">Network predictions.</param>
        /// <param name="targets">Target values (may be null if no data available).</param>
        /// <param name="derivatives">Derivatives needed for PDE computation.</param>
        /// <param name="inputs">Input points where predictions were made.</param>
        /// <returns>The total loss value.</returns>
        public T ComputeLoss(T[] predictions, T[]? targets, PDEDerivatives<T> derivatives, T[] inputs)
        {
            T totalLoss = T.Zero;

            // 1. Data Loss (if targets are provided)
            if (targets != null && targets.Length > 0)
            {
                T dataLoss = ComputeDataLoss(predictions, targets);
                totalLoss += _dataWeight * dataLoss;
            }

            // 2. PDE Residual Loss
            if (_pdeSpecification != null)
            {
                T pdeLoss = ComputePDELoss(inputs, predictions, derivatives);
                totalLoss += _pdeWeight * pdeLoss;
            }

            // 3. Boundary Condition Loss
            if (_boundaryConditions != null && _boundaryConditions.Length > 0)
            {
                T boundaryLoss = ComputeBoundaryLoss(inputs, predictions, derivatives);
                totalLoss += _boundaryWeight * boundaryLoss;
            }

            // 4. Initial Condition Loss
            if (_initialCondition != null)
            {
                T initialLoss = ComputeInitialLoss(inputs, predictions);
                totalLoss += _initialWeight * initialLoss;
            }

            return totalLoss;
        }

        /// <summary>
        /// Computes the data fitting loss (Mean Squared Error).
        /// </summary>
        private T ComputeDataLoss(T[] predictions, T[] targets)
        {
            if (predictions.Length != targets.Length)
            {
                throw new ArgumentException("Predictions and targets must have the same length.");
            }

            T sumSquaredError = T.Zero;
            for (int i = 0; i < predictions.Length; i++)
            {
                T error = predictions[i] - targets[i];
                sumSquaredError += error * error;
            }

            return sumSquaredError / T.CreateChecked(predictions.Length);
        }

        /// <summary>
        /// Computes the PDE residual loss.
        /// </summary>
        private T ComputePDELoss(T[] inputs, T[] predictions, PDEDerivatives<T> derivatives)
        {
            if (_pdeSpecification == null)
            {
                return T.Zero;
            }

            T residual = _pdeSpecification.ComputeResidual(inputs, predictions, derivatives);
            return residual * residual; // Squared residual
        }

        /// <summary>
        /// Computes the boundary condition loss.
        /// </summary>
        private T ComputeBoundaryLoss(T[] inputs, T[] predictions, PDEDerivatives<T> derivatives)
        {
            if (_boundaryConditions == null || _boundaryConditions.Length == 0)
            {
                return T.Zero;
            }

            T totalBoundaryLoss = T.Zero;
            int boundaryCount = 0;

            foreach (var bc in _boundaryConditions)
            {
                if (bc.IsOnBoundary(inputs))
                {
                    T residual = bc.ComputeBoundaryResidual(inputs, predictions, derivatives);
                    totalBoundaryLoss += residual * residual;
                    boundaryCount++;
                }
            }

            return boundaryCount > 0
                ? totalBoundaryLoss / T.CreateChecked(boundaryCount)
                : T.Zero;
        }

        /// <summary>
        /// Computes the initial condition loss.
        /// </summary>
        private T ComputeInitialLoss(T[] inputs, T[] predictions)
        {
            if (_initialCondition == null)
            {
                return T.Zero;
            }

            if (!_initialCondition.IsAtInitialTime(inputs))
            {
                return T.Zero;
            }

            // Extract spatial coordinates (all except the last which is time)
            T[] spatialInputs = new T[inputs.Length - 1];
            Array.Copy(inputs, spatialInputs, spatialInputs.Length);

            T[] expectedValues = _initialCondition.ComputeInitialValue(spatialInputs);

            T sumSquaredError = T.Zero;
            for (int i = 0; i < Math.Min(predictions.Length, expectedValues.Length); i++)
            {
                T error = predictions[i] - expectedValues[i];
                sumSquaredError += error * error;
            }

            return sumSquaredError / T.CreateChecked(predictions.Length);
        }

        /// <summary>
        /// Computes the derivative of the loss with respect to predictions.
        /// Required by the ILossFunction interface.
        /// </summary>
        public T[] ComputeDerivative(T[] predictions, T[] targets)
        {
            // For simplicity, return MSE derivative for data loss component
            T[] derivative = new T[predictions.Length];
            T scale = T.CreateChecked(2.0) / T.CreateChecked(predictions.Length);

            for (int i = 0; i < predictions.Length; i++)
            {
                derivative[i] = scale * (predictions[i] - targets[i]);
            }

            return derivative;
        }
    }
}
