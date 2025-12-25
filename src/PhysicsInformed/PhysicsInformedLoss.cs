using System;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
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
    public class PhysicsInformedLoss<T> : LossFunctionBase<T>
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
            double? dataWeight = null,
            double? pdeWeight = null,
            double? boundaryWeight = null,
            double? initialWeight = null)
        {
            _pdeSpecification = pdeSpecification;
            _boundaryConditions = boundaryConditions;
            _initialCondition = initialCondition;

            // Default weights
            _dataWeight = NumOps.FromDouble(dataWeight ?? 1.0);
            _pdeWeight = NumOps.FromDouble(pdeWeight ?? 1.0);
            _boundaryWeight = NumOps.FromDouble(boundaryWeight ?? 1.0);
            _initialWeight = NumOps.FromDouble(initialWeight ?? 1.0);
        }

        /// <summary>
        /// Gets the name of the loss function.
        /// </summary>
        public string Name => "Physics-Informed Loss";

        /// <summary>
        /// Computes the total physics-informed loss for PINN training.
        /// </summary>
        /// <param name="predictions">Network predictions.</param>
        /// <param name="targets">Target values (may be null if no data available).</param>
        /// <param name="derivatives">Derivatives needed for PDE computation.</param>
        /// <param name="inputs">Input points where predictions were made.</param>
        /// <returns>The total loss value.</returns>
        public T ComputePhysicsLoss(T[] predictions, T[]? targets, PDEDerivatives<T> derivatives, T[] inputs)
        {
            T totalLoss = NumOps.Zero;

            // 1. Data Loss (if targets are provided)
            if (targets != null && targets.Length > 0)
            {
                T dataLoss = ComputeDataLoss(predictions, targets);
                totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(_dataWeight, dataLoss));
            }

            // 2. PDE Residual Loss
            if (_pdeSpecification != null)
            {
                T pdeLoss = ComputePDELoss(inputs, predictions, derivatives);
                totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(_pdeWeight, pdeLoss));
            }

            // 3. Boundary Condition Loss
            if (_boundaryConditions != null && _boundaryConditions.Length > 0)
            {
                T boundaryLoss = ComputeBoundaryLoss(inputs, predictions, derivatives);
                totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(_boundaryWeight, boundaryLoss));
            }

            // 4. Initial Condition Loss
            if (_initialCondition != null)
            {
                T initialLoss = ComputeInitialLoss(inputs, predictions);
                totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(_initialWeight, initialLoss));
            }

            return totalLoss;
        }

        /// <summary>
        /// Computes the total physics-informed loss (compatibility wrapper).
        /// </summary>
        /// <remarks>
        /// This overload is domain-specific and intentionally separate from <see cref="ILossFunction{T}"/>.
        /// Prefer <see cref="ComputePhysicsLoss"/> for clarity.
        /// </remarks>
        public T ComputeLoss(T[] predictions, T[]? targets, PDEDerivatives<T> derivatives, T[] inputs)
        {
            return ComputePhysicsLoss(predictions, targets, derivatives, inputs);
        }

        /// <summary>
        /// Computes the physics-informed loss and its gradients with respect to outputs and derivatives.
        /// </summary>
        /// <param name="predictions">Network predictions.</param>
        /// <param name="targets">Target values (may be null if no data available).</param>
        /// <param name="derivatives">Derivatives needed for PDE computation.</param>
        /// <param name="inputs">Input points where predictions were made.</param>
        /// <returns>The loss and gradients for this sample.</returns>
        public PhysicsLossGradient<T> ComputePhysicsLossGradients(
            T[] predictions,
            T[]? targets,
            PDEDerivatives<T> derivatives,
            T[] inputs)
        {
            if (predictions == null)
            {
                throw new ArgumentNullException(nameof(predictions));
            }

            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            var gradients = new PhysicsLossGradient<T>(predictions.Length, inputs.Length, NumOps);
            T totalLoss = NumOps.Zero;

            if (targets != null && targets.Length > 0)
            {
                if (predictions.Length != targets.Length)
                {
                    throw new ArgumentException("Predictions and targets must have the same length.");
                }

                T sumSquaredError = NumOps.Zero;
                T scale = NumOps.Divide(NumOps.FromDouble(2.0), NumOps.FromDouble(predictions.Length));

                for (int i = 0; i < predictions.Length; i++)
                {
                    T error = NumOps.Subtract(predictions[i], targets[i]);
                    sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Multiply(error, error));
                    T grad = NumOps.Multiply(scale, error);
                    gradients.OutputGradients[i] = NumOps.Add(
                        gradients.OutputGradients[i],
                        NumOps.Multiply(_dataWeight, grad));
                }

                T dataLoss = NumOps.Divide(sumSquaredError, NumOps.FromDouble(predictions.Length));
                totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(_dataWeight, dataLoss));
            }

            if (_pdeSpecification != null)
            {
                T residual = _pdeSpecification.ComputeResidual(inputs, predictions, derivatives);
                T pdeLoss = NumOps.Multiply(residual, residual);
                totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(_pdeWeight, pdeLoss));

                var residualGradient = ComputeResidualGradient(inputs, predictions, derivatives);
                T scale = NumOps.Multiply(_pdeWeight, NumOps.Multiply(NumOps.FromDouble(2.0), residual));
                AccumulateResidualGradient(gradients, residualGradient, scale);
            }

            if (_boundaryConditions != null && _boundaryConditions.Length > 0)
            {
                T boundaryLossSum = NumOps.Zero;
                int boundaryCount = 0;
                var boundaryGradient = new PhysicsLossGradient<T>(predictions.Length, inputs.Length, NumOps);

                foreach (var bc in _boundaryConditions.Where(bc => bc.IsOnBoundary(inputs)))
                {
                    T residual = bc.ComputeBoundaryResidual(inputs, predictions, derivatives);
                    boundaryLossSum = NumOps.Add(boundaryLossSum, NumOps.Multiply(residual, residual));
                    boundaryCount++;

                    var residualGradient = ComputeBoundaryResidualGradient(bc, inputs, predictions, derivatives);
                    T scale = NumOps.Multiply(NumOps.FromDouble(2.0), residual);
                    AccumulateResidualGradient(boundaryGradient, residualGradient, scale);
                }

                if (boundaryCount > 0)
                {
                    T invCount = NumOps.Divide(NumOps.One, NumOps.FromDouble(boundaryCount));
                    T boundaryLoss = NumOps.Multiply(boundaryLossSum, invCount);
                    totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(_boundaryWeight, boundaryLoss));

                    T gradientScale = NumOps.Multiply(_boundaryWeight, invCount);
                    AccumulateScaledGradients(gradients, boundaryGradient, gradientScale);
                }
            }

            if (_initialCondition != null && _initialCondition.IsAtInitialTime(inputs))
            {
                T[] spatialInputs = new T[inputs.Length - 1];
                Array.Copy(inputs, spatialInputs, spatialInputs.Length);
                T[] expectedValues = _initialCondition.ComputeInitialValue(spatialInputs);
                int count = Math.Min(predictions.Length, expectedValues.Length);

                if (count > 0)
                {
                    T sumSquaredError = NumOps.Zero;
                    T scale = NumOps.Divide(NumOps.FromDouble(2.0), NumOps.FromDouble(count));

                    for (int i = 0; i < count; i++)
                    {
                        T error = NumOps.Subtract(predictions[i], expectedValues[i]);
                        sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Multiply(error, error));
                        T grad = NumOps.Multiply(scale, error);
                        gradients.OutputGradients[i] = NumOps.Add(
                            gradients.OutputGradients[i],
                            NumOps.Multiply(_initialWeight, grad));
                    }

                    T initialLoss = NumOps.Divide(sumSquaredError, NumOps.FromDouble(count));
                    totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(_initialWeight, initialLoss));
                }
            }

            gradients.Loss = totalLoss;
            return gradients;
        }

        private PDEResidualGradient<T> ComputeResidualGradient(
            T[] inputs,
            T[] predictions,
            PDEDerivatives<T> derivatives)
        {
            if (_pdeSpecification is IPDEResidualGradient<T> gradientProvider)
            {
                return gradientProvider.ComputeResidualGradient(inputs, predictions, derivatives);
            }

            return ComputeResidualGradientFallback(
                _pdeSpecification!.ComputeResidual,
                inputs,
                predictions,
                derivatives);
        }

        private PDEResidualGradient<T> ComputeBoundaryResidualGradient(
            IBoundaryCondition<T> boundaryCondition,
            T[] inputs,
            T[] predictions,
            PDEDerivatives<T> derivatives)
        {
            if (boundaryCondition is IBoundaryConditionGradient<T> gradientProvider)
            {
                return gradientProvider.ComputeBoundaryResidualGradient(inputs, predictions, derivatives);
            }

            return ComputeResidualGradientFallback(
                boundaryCondition.ComputeBoundaryResidual,
                inputs,
                predictions,
                derivatives);
        }

        private PDEResidualGradient<T> ComputeResidualGradientFallback(
            Func<T[], T[], PDEDerivatives<T>, T> residualFunction,
            T[] inputs,
            T[] predictions,
            PDEDerivatives<T> derivatives)
        {
            int outputDim = predictions.Length;
            int inputDim = inputs.Length;
            var gradient = new PDEResidualGradient<T>(outputDim, inputDim);

            T eps = NumOps.FromDouble(1e-4);
            T invTwoEps = NumOps.Divide(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2.0), eps));

            var outputCopy = new T[outputDim];
            Array.Copy(predictions, outputCopy, outputDim);

            for (int i = 0; i < outputDim; i++)
            {
                T original = outputCopy[i];
                outputCopy[i] = NumOps.Add(original, eps);
                T plus = residualFunction(inputs, outputCopy, derivatives);
                outputCopy[i] = NumOps.Subtract(original, eps);
                T minus = residualFunction(inputs, outputCopy, derivatives);
                outputCopy[i] = original;
                gradient.OutputGradients[i] = NumOps.Multiply(NumOps.Subtract(plus, minus), invTwoEps);
            }

            if (derivatives.FirstDerivatives != null)
            {
                var derivativesCopy = CloneDerivatives(derivatives);
                for (int outIdx = 0; outIdx < outputDim; outIdx++)
                {
                    for (int dim = 0; dim < inputDim; dim++)
                    {
                        T original = derivativesCopy.FirstDerivatives![outIdx, dim];
                        derivativesCopy.FirstDerivatives[outIdx, dim] = NumOps.Add(original, eps);
                        T plus = residualFunction(inputs, predictions, derivativesCopy);
                        derivativesCopy.FirstDerivatives[outIdx, dim] = NumOps.Subtract(original, eps);
                        T minus = residualFunction(inputs, predictions, derivativesCopy);
                        derivativesCopy.FirstDerivatives[outIdx, dim] = original;
                        gradient.FirstDerivatives[outIdx, dim] = NumOps.Multiply(NumOps.Subtract(plus, minus), invTwoEps);
                    }
                }
            }

            if (derivatives.SecondDerivatives != null)
            {
                var derivativesCopy = CloneDerivatives(derivatives);
                for (int outIdx = 0; outIdx < outputDim; outIdx++)
                {
                    for (int row = 0; row < inputDim; row++)
                    {
                        for (int col = 0; col < inputDim; col++)
                        {
                            T original = derivativesCopy.SecondDerivatives![outIdx, row, col];
                            derivativesCopy.SecondDerivatives[outIdx, row, col] = NumOps.Add(original, eps);
                            T plus = residualFunction(inputs, predictions, derivativesCopy);
                            derivativesCopy.SecondDerivatives[outIdx, row, col] = NumOps.Subtract(original, eps);
                            T minus = residualFunction(inputs, predictions, derivativesCopy);
                            derivativesCopy.SecondDerivatives[outIdx, row, col] = original;
                            gradient.SecondDerivatives[outIdx, row, col] = NumOps.Multiply(NumOps.Subtract(plus, minus), invTwoEps);
                        }
                    }
                }
            }

            return gradient;
        }

        private static PDEDerivatives<T> CloneDerivatives(PDEDerivatives<T> source)
        {
            var clone = new PDEDerivatives<T>();
            if (source.FirstDerivatives != null)
            {
                var first = new T[source.FirstDerivatives.GetLength(0), source.FirstDerivatives.GetLength(1)];
                Array.Copy(source.FirstDerivatives, first, source.FirstDerivatives.Length);
                clone.FirstDerivatives = first;
            }

            if (source.SecondDerivatives != null)
            {
                var second = new T[source.SecondDerivatives.GetLength(0), source.SecondDerivatives.GetLength(1), source.SecondDerivatives.GetLength(2)];
                Array.Copy(source.SecondDerivatives, second, source.SecondDerivatives.Length);
                clone.SecondDerivatives = second;
            }

            if (source.HigherDerivatives != null)
            {
                var higher = new T[source.HigherDerivatives.GetLength(0), source.HigherDerivatives.GetLength(1), source.HigherDerivatives.GetLength(2), source.HigherDerivatives.GetLength(3)];
                Array.Copy(source.HigherDerivatives, higher, source.HigherDerivatives.Length);
                clone.HigherDerivatives = higher;
            }

            return clone;
        }

        private void AccumulateResidualGradient(PhysicsLossGradient<T> target, PDEResidualGradient<T> source, T scale)
        {
            for (int i = 0; i < source.OutputGradients.Length; i++)
            {
                target.OutputGradients[i] = NumOps.Add(
                    target.OutputGradients[i],
                    NumOps.Multiply(scale, source.OutputGradients[i]));
            }

            for (int outIdx = 0; outIdx < source.FirstDerivatives.GetLength(0); outIdx++)
            {
                for (int dim = 0; dim < source.FirstDerivatives.GetLength(1); dim++)
                {
                    target.FirstDerivatives[outIdx, dim] = NumOps.Add(
                        target.FirstDerivatives[outIdx, dim],
                        NumOps.Multiply(scale, source.FirstDerivatives[outIdx, dim]));
                }
            }

            for (int outIdx = 0; outIdx < source.SecondDerivatives.GetLength(0); outIdx++)
            {
                for (int row = 0; row < source.SecondDerivatives.GetLength(1); row++)
                {
                    for (int col = 0; col < source.SecondDerivatives.GetLength(2); col++)
                    {
                        target.SecondDerivatives[outIdx, row, col] = NumOps.Add(
                            target.SecondDerivatives[outIdx, row, col],
                            NumOps.Multiply(scale, source.SecondDerivatives[outIdx, row, col]));
                    }
                }
            }
        }

        private void AccumulateScaledGradients(PhysicsLossGradient<T> target, PhysicsLossGradient<T> source, T scale)
        {
            for (int i = 0; i < source.OutputGradients.Length; i++)
            {
                target.OutputGradients[i] = NumOps.Add(
                    target.OutputGradients[i],
                    NumOps.Multiply(scale, source.OutputGradients[i]));
            }

            for (int outIdx = 0; outIdx < source.FirstDerivatives.GetLength(0); outIdx++)
            {
                for (int dim = 0; dim < source.FirstDerivatives.GetLength(1); dim++)
                {
                    target.FirstDerivatives[outIdx, dim] = NumOps.Add(
                        target.FirstDerivatives[outIdx, dim],
                        NumOps.Multiply(scale, source.FirstDerivatives[outIdx, dim]));
                }
            }

            for (int outIdx = 0; outIdx < source.SecondDerivatives.GetLength(0); outIdx++)
            {
                for (int row = 0; row < source.SecondDerivatives.GetLength(1); row++)
                {
                    for (int col = 0; col < source.SecondDerivatives.GetLength(2); col++)
                    {
                        target.SecondDerivatives[outIdx, row, col] = NumOps.Add(
                            target.SecondDerivatives[outIdx, row, col],
                            NumOps.Multiply(scale, source.SecondDerivatives[outIdx, row, col]));
                    }
                }
            }
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

            T sumSquaredError = NumOps.Zero;
            for (int i = 0; i < predictions.Length; i++)
            {
                T error = NumOps.Subtract(predictions[i], targets[i]);
                sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Multiply(error, error));
            }

            return NumOps.Divide(sumSquaredError, NumOps.FromDouble(predictions.Length));
        }

        /// <summary>
        /// Computes the PDE residual loss.
        /// </summary>
        private T ComputePDELoss(T[] inputs, T[] predictions, PDEDerivatives<T> derivatives)
        {
            if (_pdeSpecification == null)
            {
                return NumOps.Zero;
            }

            T residual = _pdeSpecification.ComputeResidual(inputs, predictions, derivatives);
            return NumOps.Multiply(residual, residual); // Squared residual
        }

        /// <summary>
        /// Computes the boundary condition loss.
        /// </summary>
        private T ComputeBoundaryLoss(T[] inputs, T[] predictions, PDEDerivatives<T> derivatives)
        {
            if (_boundaryConditions == null || _boundaryConditions.Length == 0)
            {
                return NumOps.Zero;
            }

            T totalBoundaryLoss = NumOps.Zero;
            int boundaryCount = 0;

            foreach (var bc in _boundaryConditions.Where(bc => bc.IsOnBoundary(inputs)))
            {
                T residual = bc.ComputeBoundaryResidual(inputs, predictions, derivatives);
                totalBoundaryLoss = NumOps.Add(totalBoundaryLoss, NumOps.Multiply(residual, residual));
                boundaryCount++;
            }

            return boundaryCount > 0
                ? NumOps.Divide(totalBoundaryLoss, NumOps.FromDouble(boundaryCount))
                : NumOps.Zero;
        }

        /// <summary>
        /// Computes the initial condition loss.
        /// </summary>
        private T ComputeInitialLoss(T[] inputs, T[] predictions)
        {
            if (_initialCondition == null)
            {
                return NumOps.Zero;
            }

            if (!_initialCondition.IsAtInitialTime(inputs))
            {
                return NumOps.Zero;
            }

            // Extract spatial coordinates (all except the last which is time)
            T[] spatialInputs = new T[inputs.Length - 1];
            Array.Copy(inputs, spatialInputs, spatialInputs.Length);

            T[] expectedValues = _initialCondition.ComputeInitialValue(spatialInputs);

            int count = Math.Min(predictions.Length, expectedValues.Length);
            if (count == 0)
            {
                return NumOps.Zero;
            }

            T sumSquaredError = NumOps.Zero;
            for (int i = 0; i < count; i++)
            {
                T error = NumOps.Subtract(predictions[i], expectedValues[i]);
                sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Multiply(error, error));
            }

            return NumOps.Divide(sumSquaredError, NumOps.FromDouble(count));
        }

        /// <summary>
        /// Computes the derivative of the loss with respect to predictions.
        /// Required by the ILossFunction interface.
        /// </summary>
        public T[] ComputeDerivative(T[] predictions, T[] targets)
        {
            // For simplicity, return MSE derivative for data loss component
            T[] derivative = new T[predictions.Length];
            T scale = NumOps.Divide(NumOps.FromDouble(2.0), NumOps.FromDouble(predictions.Length));

            for (int i = 0; i < predictions.Length; i++)
            {
                derivative[i] = NumOps.Multiply(scale, NumOps.Subtract(predictions[i], targets[i]));
            }

            return derivative;
        }

        /// <inheritdoc/>
        public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
        {
            ValidateVectorLengths(predicted, actual);

            T sumSquaredError = NumOps.Zero;
            for (int i = 0; i < predicted.Length; i++)
            {
                T error = NumOps.Subtract(predicted[i], actual[i]);
                sumSquaredError = NumOps.Add(sumSquaredError, NumOps.Multiply(error, error));
            }

            return NumOps.Divide(sumSquaredError, NumOps.FromDouble(predicted.Length));
        }

        /// <inheritdoc/>
        public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
        {
            ValidateVectorLengths(predicted, actual);

            var derivative = new Vector<T>(predicted.Length);
            T scale = NumOps.Divide(NumOps.FromDouble(2.0), NumOps.FromDouble(predicted.Length));

            for (int i = 0; i < predicted.Length; i++)
            {
                derivative[i] = NumOps.Multiply(scale, NumOps.Subtract(predicted[i], actual[i]));
            }

            return derivative;
        }
    }

    /// <summary>
    /// Holds loss and gradient information for physics-informed objectives.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public sealed class PhysicsLossGradient<T>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PhysicsLossGradient{T}"/> class.
        /// </summary>
        /// <param name="outputDimension">The output dimension.</param>
        /// <param name="inputDimension">The input dimension.</param>
        public PhysicsLossGradient(int outputDimension, int inputDimension, INumericOperations<T> numOps)
        {
            if (numOps == null)
            {
                throw new ArgumentNullException(nameof(numOps));
            }

            Loss = numOps.Zero;
            OutputGradients = new T[outputDimension];
            FirstDerivatives = new T[outputDimension, inputDimension];
            SecondDerivatives = new T[outputDimension, inputDimension, inputDimension];
        }

        /// <summary>
        /// Gets or sets the loss value for this sample.
        /// </summary>
        public T Loss { get; set; }

        /// <summary>
        /// Gradient of loss with respect to outputs.
        /// </summary>
        public T[] OutputGradients { get; }

        /// <summary>
        /// Gradient of loss with respect to first derivatives.
        /// </summary>
        public T[,] FirstDerivatives { get; }

        /// <summary>
        /// Gradient of loss with respect to second derivatives.
        /// </summary>
        public T[,,] SecondDerivatives { get; }
    }
}
