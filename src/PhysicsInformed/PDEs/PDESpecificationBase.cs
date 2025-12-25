using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

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
        public abstract T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives);

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
    }
}
