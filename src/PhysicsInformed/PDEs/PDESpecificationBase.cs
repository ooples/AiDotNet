using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Base class for all Partial Differential Equation (PDE) specifications.
    /// Provides common functionality for PDE implementations used with Physics-Informed Neural Networks.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Partial Differential Equation (PDE) describes how a quantity changes
    /// with respect to multiple variables (like space and time). This base class provides the
    /// foundation for implementing any PDE that can be solved using Physics-Informed Neural Networks.
    /// </para>
    /// <para>
    /// All PDE implementations should inherit from this class rather than implementing
    /// <see cref="IPDESpecification{T}"/> directly. This ensures consistent behavior and
    /// provides access to common helper methods.
    /// </para>
    /// <para>
    /// To create a new PDE, override the abstract members:
    /// <list type="bullet">
    ///     <item><description><see cref="ComputeResidual"/> - Calculate how much the PDE is violated</description></item>
    ///     <item><description><see cref="InputDimension"/> - Number of independent variables (space + time)</description></item>
    ///     <item><description><see cref="OutputDimension"/> - Number of dependent variables (solution components)</description></item>
    ///     <item><description><see cref="Name"/> - Human-readable name of the PDE</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public abstract class PDESpecificationBase<T> : IPDESpecification<T>
    {
        /// <summary>
        /// Provides mathematical operations for the numeric type T.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This object knows how to do math operations (add, multiply, etc.)
        /// for whatever numeric type T is (float, double, etc.). Use it for all calculations.
        /// </remarks>
        protected readonly INumericOperations<T> NumOps;

        /// <summary>
        /// Initializes a new instance of the <see cref="PDESpecificationBase{T}"/> class.
        /// </summary>
        protected PDESpecificationBase()
        {
            NumOps = MathHelper.GetNumericOperations<T>();
        }

        /// <inheritdoc/>
        public abstract T ComputeResidual(Vector<T> inputs, Vector<T> outputs, PDEDerivatives<T> derivatives);

        /// <inheritdoc/>
        public abstract int InputDimension { get; }

        /// <inheritdoc/>
        public abstract int OutputDimension { get; }

        /// <inheritdoc/>
        public abstract string Name { get; }

        /// <summary>
        /// Validates that first-order derivatives are available.
        /// </summary>
        /// <param name="derivatives">The derivatives to validate.</param>
        /// <exception cref="ArgumentException">Thrown when first derivatives are null.</exception>
        protected void ValidateFirstDerivatives(PDEDerivatives<T> derivatives)
        {
            if (derivatives.FirstDerivatives == null)
            {
                throw new ArgumentException($"{Name} requires first-order derivatives.");
            }
        }

        /// <summary>
        /// Validates that both first and second-order derivatives are available.
        /// </summary>
        /// <param name="derivatives">The derivatives to validate.</param>
        /// <exception cref="ArgumentException">Thrown when required derivatives are null.</exception>
        protected void ValidateSecondDerivatives(PDEDerivatives<T> derivatives)
        {
            if (derivatives.FirstDerivatives == null || derivatives.SecondDerivatives == null)
            {
                throw new ArgumentException($"{Name} requires first and second-order derivatives.");
            }
        }

        /// <summary>
        /// Validates that first, second, and third-order derivatives are available.
        /// </summary>
        /// <param name="derivatives">The derivatives to validate.</param>
        /// <exception cref="ArgumentException">Thrown when required derivatives are null.</exception>
        protected void ValidateThirdDerivatives(PDEDerivatives<T> derivatives)
        {
            if (derivatives.FirstDerivatives == null || derivatives.ThirdDerivatives == null)
            {
                throw new ArgumentException($"{Name} requires first and third-order derivatives.");
            }
        }

        /// <summary>
        /// Validates that a parameter is positive (greater than zero).
        /// </summary>
        /// <param name="value">The value to validate.</param>
        /// <param name="parameterName">The name of the parameter for error messages.</param>
        /// <exception cref="ArgumentException">Thrown when the value is not positive.</exception>
        protected void ValidatePositive(T value, string parameterName)
        {
            if (NumOps.LessThanOrEquals(value, NumOps.Zero))
            {
                throw new ArgumentException($"{parameterName} must be positive.", parameterName);
            }
        }

        /// <summary>
        /// Validates that a parameter is non-negative (greater than or equal to zero).
        /// </summary>
        /// <param name="value">The value to validate.</param>
        /// <param name="parameterName">The name of the parameter for error messages.</param>
        /// <exception cref="ArgumentException">Thrown when the value is negative.</exception>
        protected void ValidateNonNegative(T value, string parameterName)
        {
            if (NumOps.LessThan(value, NumOps.Zero))
            {
                throw new ArgumentException($"{parameterName} must be non-negative.", parameterName);
            }
        }

        /// <summary>
        /// Creates a new PDEResidualGradient with the appropriate dimensions.
        /// </summary>
        /// <returns>A new gradient object initialized with zeros.</returns>
        protected PDEResidualGradient<T> CreateGradient()
        {
            return new PDEResidualGradient<T>(OutputDimension, InputDimension);
        }

        /// <summary>
        /// The highest input-derivative order <see cref="ComputeTapeResidual"/> reads (1 to 3).
        /// </summary>
        /// <remarks>
        /// A PINN computes this many orders of derivatives through the gradient tape at its collocation points.
        /// Second order (diffusion, wave and Poisson terms) is the default; first-order systems (Maxwell) override
        /// it with 1, third-order equations (Korteweg-de Vries) with 3.
        /// </remarks>
        public virtual int DerivativeOrder => 2;

        /// <summary>
        /// Computes the PDE residual at a batch of points with tape-differentiable engine operations.
        /// </summary>
        /// <param name="inputs">The coordinates [N, InputDimension].</param>
        /// <param name="outputs">The network outputs at those coordinates [N, OutputDimension].</param>
        /// <param name="derivatives">
        /// The batched input derivatives of the outputs: first [N, out, in], second [N, out, in, in] and third
        /// [N, out, in, in, in], up to <see cref="DerivativeOrder"/>.
        /// </param>
        /// <returns>The residual at every point, [N].</returns>
        /// <remarks>
        /// This is the residual f of Raissi et al. (2019) evaluated for a whole batch of collocation points, so a
        /// PINN can minimise mean(f^2) on the gradient tape. It must agree with <see cref="ComputeResidual"/> point by
        /// point. Implementations read the derivative entries with <see cref="Output"/>, <see cref="D1"/>,
        /// <see cref="D2"/> and <see cref="D3"/> and combine them with engine operations so the gradient reaches the
        /// network.
        /// </remarks>
        public virtual Tensor<T> ComputeTapeResidual(Tensor<T> inputs, Tensor<T> outputs, PDEDerivatives<T> derivatives)
        {
            throw new NotSupportedException(
                $"PDE '{Name}' does not implement a tape-differentiable residual. Override ComputeTapeResidual with " +
                "engine operations so a PINN can train on it.");
        }

        /// <summary>The engine the tape residual helpers use.</summary>
        protected static IEngine TapeEngine => AiDotNetEngine.Current;

        /// <summary>Output component <paramref name="k"/> of a batch, [N].</summary>
        protected static Tensor<T> Output(Tensor<T> outputs, int k)
            => PinnAutodiff<T>.Column(TapeEngine, outputs, k);

        /// <summary>Input coordinate <paramref name="i"/> of a batch, [N].</summary>
        protected static Tensor<T> Coordinate(Tensor<T> inputs, int i)
            => PinnAutodiff<T>.Column(TapeEngine, inputs, i);

        /// <summary>du_k/dx_i over a batch, [N].</summary>
        protected Tensor<T> D1(PDEDerivatives<T> derivatives, int k, int i)
            => Entry(derivatives.FirstDerivatives, "first", new[] { k, i });

        /// <summary>d2u_k/(dx_i dx_j) over a batch, [N].</summary>
        protected Tensor<T> D2(PDEDerivatives<T> derivatives, int k, int i, int j)
            => Entry(derivatives.SecondDerivatives, "second", new[] { k, i, j });

        /// <summary>d3u_k/(dx_i dx_j dx_l) over a batch, [N].</summary>
        protected Tensor<T> D3(PDEDerivatives<T> derivatives, int k, int i, int j, int l)
            => Entry(derivatives.ThirdDerivatives, "third", new[] { k, i, j, l });

        private Tensor<T> Entry(Tensor<T>? batched, string order, int[] index)
        {
            if (batched is null)
            {
                throw new ArgumentException($"{Name} requires batched {order}-order derivatives.");
            }

            if (batched.Rank != index.Length + 1)
            {
                throw new ArgumentException(
                    $"Batched {order}-order derivatives must have rank {index.Length + 1}; got {batched.Rank}.");
            }

            int n = batched.Shape[0];
            var start = new int[index.Length + 1];
            var length = new int[index.Length + 1];
            length[0] = n;
            for (int axis = 0; axis < index.Length; axis++)
            {
                start[axis + 1] = index[axis];
                length[axis + 1] = 1;
            }

            var engine = TapeEngine;
            return engine.Reshape(engine.TensorSlice(batched, start, length), new[] { n });
        }
    }
}
