using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

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
        public T ComputePhysicsLoss(Vector<T> predictions, Vector<T>? targets, PDEDerivatives<T> derivatives, Vector<T> inputs)
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
        public T ComputeLoss(Vector<T> predictions, Vector<T>? targets, PDEDerivatives<T> derivatives, Vector<T> inputs)
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
            Vector<T> predictions,
            Vector<T>? targets,
            PDEDerivatives<T> derivatives,
            Vector<T> inputs)
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

                // Vectorized data loss: MSE + gradient accumulation
                var error = (Vector<T>)Engine.Subtract(predictions, targets);
                T sumSquaredError = VectorHelper.DotProduct(error, error);
                T scale = NumOps.Divide(NumOps.FromDouble(2.0), NumOps.FromDouble(predictions.Length));
                var grad = (Vector<T>)Engine.Multiply(error, scale);
                var weightedGrad = (Vector<T>)Engine.Multiply(grad, _dataWeight);
                gradients.OutputGradients = (Vector<T>)Engine.Add(gradients.OutputGradients, weightedGrad);

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
                var boundaryResiduals = new List<T>();
                var boundaryGradient = new PhysicsLossGradient<T>(predictions.Length, inputs.Length, NumOps);

                // Each condition contributes r^2 to the loss and 2 r dr to the gradient. The gradient is linear in
                // its scale, so the factor 2 joins the shared scale after the loop.
                foreach (var bc in _boundaryConditions.Where(bc => bc.IsOnBoundary(inputs)))
                {
                    T residual = bc.ComputeBoundaryResidual(inputs, predictions, derivatives);
                    boundaryResiduals.Add(residual);

                    var residualGradient = ComputeBoundaryResidualGradient(bc, inputs, predictions, derivatives);
                    AccumulateResidualGradient(boundaryGradient, residualGradient, residual);
                }

                int boundaryCount = boundaryResiduals.Count;
                if (boundaryCount > 0)
                {
                    var residualVector = new Vector<T>(boundaryResiduals.ToArray());
                    T boundaryLossSum = VectorHelper.DotProduct(residualVector, residualVector);
                    T invCount = NumOps.Divide(NumOps.One, NumOps.FromDouble(boundaryCount));
                    T boundaryLoss = NumOps.Multiply(boundaryLossSum, invCount);
                    totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(_boundaryWeight, boundaryLoss));

                    T gradientScale = NumOps.Multiply(
                        NumOps.FromDouble(2.0), NumOps.Multiply(_boundaryWeight, invCount));
                    AccumulateScaledGradients(gradients, boundaryGradient, gradientScale);
                }
            }

            if (_initialCondition != null && _initialCondition.IsAtInitialTime(inputs))
            {
                var spatialInputs = SpatialPart(inputs);
                var expectedValues = _initialCondition.ComputeInitialValue(spatialInputs);
                int count = Math.Min(predictions.Length, expectedValues.Length);

                if (count > 0)
                {
                    var predSlice = Leading(predictions, count);
                    var expSlice = Leading(expectedValues, count);

                    var error = (Vector<T>)Engine.Subtract(predSlice, expSlice);
                    T sumSquaredError = VectorHelper.DotProduct(error, error);
                    T scale = NumOps.Divide(NumOps.FromDouble(2.0), NumOps.FromDouble(count));
                    var grad = (Vector<T>)Engine.Multiply(error, scale);
                    var weightedGrad = (Vector<T>)Engine.Multiply(grad, _initialWeight);

                    // Accumulate into output gradients (may need to add only to first 'count' elements)
                    var padded = count == gradients.OutputGradients.Length
                        ? weightedGrad
                        : Vector<T>.Concatenate(weightedGrad, new Vector<T>(gradients.OutputGradients.Length - count));
                    gradients.OutputGradients = (Vector<T>)Engine.Add(gradients.OutputGradients, padded);

                    T initialLoss = NumOps.Divide(sumSquaredError, NumOps.FromDouble(count));
                    totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(_initialWeight, initialLoss));
                }
            }

            gradients.Loss = totalLoss;
            return gradients;
        }

        private PDEResidualGradient<T> ComputeResidualGradient(
            Vector<T> inputs,
            Vector<T> predictions,
            PDEDerivatives<T> derivatives)
        {
            if (_pdeSpecification is IPDEResidualGradient<T> gradientProvider)
            {
                return gradientProvider.ComputeResidualGradient(inputs, predictions, derivatives);
            }

            return ComputeResidualGradientFallback(
                (_pdeSpecification ?? throw new InvalidOperationException("_pdeSpecification has not been initialized.")).ComputeResidual,
                inputs,
                predictions,
                derivatives);
        }

        private PDEResidualGradient<T> ComputeBoundaryResidualGradient(
            IBoundaryCondition<T> boundaryCondition,
            Vector<T> inputs,
            Vector<T> predictions,
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

        /// <summary>
        /// The residual's gradient with respect to the outputs and derivatives, for a residual with no analytic
        /// gradient.
        /// </summary>
        /// <remarks>
        /// The residual is a black-box scalar callback, so its gradient is the central difference of the callback
        /// along each coordinate direction, one callback pair per entry. The directions are the rows of a scaled
        /// identity, so the perturbed points are built with engine operations rather than written entry by entry.
        /// </remarks>
        private PDEResidualGradient<T> ComputeResidualGradientFallback(
            Func<Vector<T>, Vector<T>, PDEDerivatives<T>, T> residualFunction,
            Vector<T> inputs,
            Vector<T> predictions,
            PDEDerivatives<T> derivatives)
        {
            int outputDim = predictions.Length;
            int inputDim = inputs.Length;
            var gradient = new PDEResidualGradient<T>(outputDim, inputDim);

            T eps = NumOps.FromDouble(1e-4);
            T invTwoEps = NumOps.Divide(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2.0), eps));

            var outputGradient = CentralDifference(
                Tensor<T>.FromVector(predictions),
                perturbed => residualFunction(inputs, perturbed.ToVector(), derivatives),
                eps,
                invTwoEps);
            CopyInto(outputGradient, gradient.OutputGradients);

            if (derivatives.FirstDerivatives is not null)
            {
                var result = CentralDifference(
                    derivatives.FirstDerivatives,
                    perturbed => residualFunction(inputs, predictions, WithFirst(derivatives, perturbed)),
                    eps,
                    invTwoEps);
                CopyInto(result, gradient.FirstDerivatives);
            }

            if (derivatives.SecondDerivatives is not null)
            {
                var result = CentralDifference(
                    derivatives.SecondDerivatives,
                    perturbed => residualFunction(inputs, predictions, WithSecond(derivatives, perturbed)),
                    eps,
                    invTwoEps);
                CopyInto(result, gradient.SecondDerivatives);
            }

            return gradient;
        }

        /// <summary>
        /// Central differences of <paramref name="function"/> along every coordinate direction of
        /// <paramref name="point"/>; the result has the point's shape.
        /// </summary>
        private Tensor<T> CentralDifference(Tensor<T> point, Func<Tensor<T>, T> function, T eps, T invTwoEps)
        {
            int count = point.Length;
            var shape = point.Shape.ToArray();
            var directions = Engine.TensorMultiplyScalar(Engine.TensorEye<T>(count), eps);
            var flat = Engine.Reshape(point, new[] { count });
            var plusValues = new List<T>(count);
            var minusValues = new List<T>(count);
            for (int i = 0; i < count; i++)
            {
                var direction = Engine.Reshape(
                    Engine.TensorSlice(directions, new[] { i, 0 }, new[] { 1, count }), new[] { count });
                plusValues.Add(function(Engine.Reshape(Engine.TensorAdd(flat, direction), shape)));
                minusValues.Add(function(Engine.Reshape(Engine.TensorSubtract(flat, direction), shape)));
            }

            var difference = Engine.TensorSubtract(
                Tensor<T>.FromVector(Vector<T>.FromList(plusValues), shape),
                Tensor<T>.FromVector(Vector<T>.FromList(minusValues), shape));
            return Engine.TensorMultiplyScalar(difference, invTwoEps);
        }

        private static PDEDerivatives<T> WithFirst(PDEDerivatives<T> source, Tensor<T> first) => new PDEDerivatives<T>
        {
            FirstDerivatives = first,
            SecondDerivatives = source.SecondDerivatives,
            ThirdDerivatives = source.ThirdDerivatives,
            HigherDerivatives = source.HigherDerivatives
        };

        private static PDEDerivatives<T> WithSecond(PDEDerivatives<T> source, Tensor<T> second) => new PDEDerivatives<T>
        {
            FirstDerivatives = source.FirstDerivatives,
            SecondDerivatives = second,
            ThirdDerivatives = source.ThirdDerivatives,
            HigherDerivatives = source.HigherDerivatives
        };

        private static void CopyInto(Tensor<T> source, Vector<T> destination)
        {
            if (source.Length != destination.Length)
            {
                throw new InvalidOperationException(
                    $"Output gradient has {source.Length} entries; expected {destination.Length}.");
            }

            source.AsSpan().CopyTo(destination.AsWritableSpan());
        }

        private static void CopyInto(Tensor<T> source, Tensor<T> destination)
        {
            if (source.Length != destination.Length)
            {
                throw new InvalidOperationException(
                    $"Derivative gradient has {source.Length} entries; expected {destination.Length}.");
            }

            source.AsSpan().CopyTo(destination.AsWritableSpan());
        }

        private void AccumulateResidualGradient(PhysicsLossGradient<T> target, PDEResidualGradient<T> source, T scale)
        {
            target.OutputGradients = (Vector<T>)Engine.Add(
                target.OutputGradients, (Vector<T>)Engine.Multiply(source.OutputGradients, scale));
            target.FirstDerivatives = Engine.TensorAdd(
                target.FirstDerivatives, Engine.TensorMultiplyScalar(source.FirstDerivatives, scale));
            target.SecondDerivatives = Engine.TensorAdd(
                target.SecondDerivatives, Engine.TensorMultiplyScalar(source.SecondDerivatives, scale));
        }

        private void AccumulateScaledGradients(PhysicsLossGradient<T> target, PhysicsLossGradient<T> source, T scale)
        {
            target.OutputGradients = (Vector<T>)Engine.Add(
                target.OutputGradients, (Vector<T>)Engine.Multiply(source.OutputGradients, scale));
            target.FirstDerivatives = Engine.TensorAdd(
                target.FirstDerivatives, Engine.TensorMultiplyScalar(source.FirstDerivatives, scale));
            target.SecondDerivatives = Engine.TensorAdd(
                target.SecondDerivatives, Engine.TensorMultiplyScalar(source.SecondDerivatives, scale));
        }

        /// <summary>The spatial coordinates of a space-time point: every input but the last (time).</summary>
        private static Vector<T> SpatialPart(Vector<T> inputs) => inputs.GetSubVector(0, inputs.Length - 1);

        private static Vector<T> Leading(Vector<T> values, int count)
            => count == values.Length ? values : values.GetSubVector(0, count);

        /// <summary>
        /// Computes the data fitting loss (Mean Squared Error).
        /// </summary>
        private T ComputeDataLoss(Vector<T> predictions, Vector<T> targets)
        {
            if (predictions.Length != targets.Length)
            {
                throw new ArgumentException("Predictions and targets must have the same length.");
            }

            // Vectorized MSE: sum((pred - target)^2) / N
            var error = (Vector<T>)Engine.Subtract(predictions, targets);
            T sumSquaredError = VectorHelper.DotProduct(error, error);
            return NumOps.Divide(sumSquaredError, NumOps.FromDouble(predictions.Length));
        }

        /// <summary>
        /// Computes the PDE residual loss.
        /// </summary>
        private T ComputePDELoss(Vector<T> inputs, Vector<T> predictions, PDEDerivatives<T> derivatives)
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
        private T ComputeBoundaryLoss(Vector<T> inputs, Vector<T> predictions, PDEDerivatives<T> derivatives)
        {
            if (_boundaryConditions == null || _boundaryConditions.Length == 0)
            {
                return NumOps.Zero;
            }

            var residuals = _boundaryConditions
                .Where(bc => bc.IsOnBoundary(inputs))
                .Select(bc => bc.ComputeBoundaryResidual(inputs, predictions, derivatives))
                .ToList();
            if (residuals.Count == 0)
            {
                return NumOps.Zero;
            }

            var residualVector = new Vector<T>(residuals.ToArray());
            return NumOps.Divide(
                VectorHelper.DotProduct(residualVector, residualVector), NumOps.FromDouble(residuals.Count));
        }

        /// <summary>
        /// Computes the initial condition loss.
        /// </summary>
        private T ComputeInitialLoss(Vector<T> inputs, Vector<T> predictions)
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
            var spatialInputs = SpatialPart(inputs);

            var expectedValues = _initialCondition.ComputeInitialValue(spatialInputs);

            int count = Math.Min(predictions.Length, expectedValues.Length);
            if (count == 0)
            {
                return NumOps.Zero;
            }

            var predSlice = Leading(predictions, count);
            var expSlice = Leading(expectedValues, count);

            var error = (Vector<T>)Engine.Subtract(predSlice, expSlice);
            T sumSquaredError = VectorHelper.DotProduct(error, error);
            return NumOps.Divide(sumSquaredError, NumOps.FromDouble(count));
        }

        /// <inheritdoc/>
        public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
        {
            ValidateVectorLengths(predicted, actual);

            // Vectorized MSE: sum((pred - actual)^2) / N
            var error = (Vector<T>)Engine.Subtract(predicted, actual);
            T sumSquaredError = VectorHelper.DotProduct(error, error);
            return NumOps.Divide(sumSquaredError, NumOps.FromDouble(predicted.Length));
        }

        /// <inheritdoc />
        /// <remarks>
        /// The weighted data term, data_weight * MSE(predicted, target). The PDE term needs the network's input
        /// derivatives, which a prediction alone does not carry; a PINN adds it with
        /// <see cref="ComputeTapePdeLoss"/>.
        /// </remarks>
        public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
        {
            var diff = Engine.TensorSubtract(predicted, target);
            var sq = Engine.TensorMultiply(diff, diff);
            var allAxes = Enumerable.Range(0, sq.Shape.Length).ToArray();
            var dataLoss = Engine.ReduceMean(sq, allAxes, keepDims: false);
            return Engine.TensorMultiplyScalar(dataLoss, _dataWeight);
        }

        /// <summary>
        /// The weighted PDE term pde_weight * mean(f^2) over a batch of collocation points, on the gradient tape.
        /// </summary>
        /// <param name="inputs">The collocation points [N, InputDimension].</param>
        /// <param name="outputs">The network outputs there [N, OutputDimension].</param>
        /// <param name="derivatives">The batched input derivatives of the outputs.</param>
        /// <returns>A scalar loss tensor.</returns>
        /// <remarks>This is MSE_f of Raissi et al. (2019), Eq. 4, weighted by the PDE weight.</remarks>
        public Tensor<T> ComputeTapePdeLoss(Tensor<T> inputs, Tensor<T> outputs, PDEDerivatives<T> derivatives)
        {
            if (_pdeSpecification is not PDEs.PDESpecificationBase<T> tapeSpec)
            {
                throw new NotSupportedException(
                    "The PDE term on the gradient tape needs a PDESpecificationBase residual (ComputeTapeResidual).");
            }

            var residual = tapeSpec.ComputeTapeResidual(inputs, outputs, derivatives);
            var squared = Engine.TensorMultiply(residual, residual);
            var axes = Enumerable.Range(0, squared.Shape.Length).ToArray();
            return Engine.TensorMultiplyScalar(Engine.ReduceMean(squared, axes, keepDims: false), _pdeWeight);
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
            OutputGradients = new Vector<T>(outputDimension);
            FirstDerivatives = new Tensor<T>(new[] { outputDimension, inputDimension });
            SecondDerivatives = new Tensor<T>(new[] { outputDimension, inputDimension, inputDimension });
        }

        /// <summary>
        /// Gets or sets the loss value for this sample.
        /// </summary>
        public T Loss { get; set; }

        /// <summary>
        /// Gradient of loss with respect to outputs.
        /// </summary>
        public Vector<T> OutputGradients { get; set; }

        /// <summary>
        /// Gradient of loss with respect to first derivatives, shape [output_dim, input_dim].
        /// </summary>
        public Tensor<T> FirstDerivatives { get; set; }

        /// <summary>
        /// Gradient of loss with respect to second derivatives, shape [output_dim, input_dim, input_dim].
        /// </summary>
        public Tensor<T> SecondDerivatives { get; set; }
    }
}
