using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents the Allen-Cahn equation: u_t - epsilon^2 * u_xx + u^3 - u = 0.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// The Allen-Cahn equation models phase separation and interface motion.
    /// It combines diffusion (smoothing) with a nonlinear reaction term.
    /// </remarks>
    public class AllenCahnEquation<T> : PDESpecificationBase<T>, IPDEResidualGradient<T>
    {
        private readonly T _epsilon;

        /// <summary>
        /// Initializes a new instance of the Allen-Cahn equation.
        /// </summary>
        /// <param name="epsilon">Interface width parameter (must be positive).</param>
        public AllenCahnEquation(T epsilon)
        {
            ValidatePositive(epsilon, nameof(epsilon));
            _epsilon = epsilon;
        }

        /// <summary>
        /// Initializes a new instance of the Allen-Cahn equation with double parameter.
        /// </summary>
        /// <param name="epsilon">Interface width parameter (default 0.01, must be positive).</param>
        public AllenCahnEquation(double epsilon = 0.01)
            : this(MathHelper.GetNumericOperations<T>().FromDouble(epsilon))
        {
        }

        /// <inheritdoc/>
        public override T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateSecondDerivatives(derivatives);

            var firstDerivs = derivatives.FirstDerivatives;
            var secondDerivs = derivatives.SecondDerivatives;

            if (firstDerivs is null || secondDerivs is null)
            {
                throw new InvalidOperationException("Derivatives were null after validation.");
            }

            T u = outputs[0];
            T dudt = firstDerivs[0, 1];
            T d2udx2 = secondDerivs[0, 0, 0];

            T uSquared = NumOps.Multiply(u, u);
            T uCubed = NumOps.Multiply(uSquared, u);
            T reaction = NumOps.Subtract(uCubed, u);
            T epsilonSquared = NumOps.Multiply(_epsilon, _epsilon);
            T diffusion = NumOps.Multiply(epsilonSquared, d2udx2);

            return NumOps.Subtract(NumOps.Add(dudt, reaction), diffusion);
        }

        /// <inheritdoc/>
        public PDEResidualGradient<T> ComputeResidualGradient(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateSecondDerivatives(derivatives);

            var gradient = CreateGradient();

            T u = outputs[0];
            T uSquared = NumOps.Multiply(u, u);
            T three = NumOps.FromDouble(3.0);
            gradient.OutputGradients[0] = NumOps.Subtract(NumOps.Multiply(three, uSquared), NumOps.One);
            gradient.FirstDerivatives[0, 1] = NumOps.One;

            T epsilonSquared = NumOps.Multiply(_epsilon, _epsilon);
            gradient.SecondDerivatives[0, 0, 0] = NumOps.Negate(epsilonSquared);

            return gradient;
        }

        /// <inheritdoc/>
        public override int InputDimension => 2;

        /// <inheritdoc/>
        public override int OutputDimension => 1;

        /// <inheritdoc/>
        public override string Name => "Allen-Cahn Equation";
    }
}
