using System;

namespace AiDotNet.PhysicsInformed.Interfaces
{
    /// <summary>
    /// Defines the interface for specifying Partial Differential Equations (PDEs) that can be used with Physics-Informed Neural Networks.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.) used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// A Partial Differential Equation (PDE) is an equation that involves rates of change with respect to multiple variables.
    /// For example, the heat equation describes how temperature changes over both space and time.
    /// This interface allows you to define any PDE in a way that neural networks can learn to solve it.
    /// </remarks>
    [AiDotNet.Configuration.YamlConfigurable("PDESpecification")]
    public interface IPDESpecification<T>
    {
        /// <summary>
        /// Computes the PDE residual at the given point.
        /// The residual is how much the PDE equation is violated at that point.
        /// For a true solution, the residual should be zero everywhere.
        /// </summary>
        /// <param name="inputs">The spatial and temporal coordinates where to evaluate the PDE.</param>
        /// <param name="outputs">The predicted solution values at those coordinates.</param>
        /// <param name="derivatives">The derivatives of the solution (gradient, Hessian, etc.).</param>
        /// <returns>The PDE residual value (should be zero for a perfect solution).</returns>
        T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives);

        /// <summary>
        /// Gets the dimension of the input space (e.g., 2 for 2D spatial problems, 3 for 2D space + time).
        /// </summary>
        int InputDimension { get; }

        /// <summary>
        /// Gets the dimension of the output space (e.g., 1 for scalar fields like temperature, 3 for vector fields like velocity).
        /// </summary>
        int OutputDimension { get; }

        /// <summary>
        /// Gets the name or description of the PDE (e.g., "Heat Equation", "Navier-Stokes").
        /// </summary>
        string Name { get; }
    }

    /// <summary>
    /// Provides gradients of the PDE residual with respect to outputs and derivatives.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    [AiDotNet.Configuration.YamlConfigurable("PDEResidualGradient")]
    public interface IPDEResidualGradient<T>
    {
        /// <summary>
        /// Computes gradients of the PDE residual with respect to outputs and derivatives.
        /// </summary>
        /// <param name="inputs">The spatial and temporal coordinates.</param>
        /// <param name="outputs">The predicted solution values.</param>
        /// <param name="derivatives">The derivatives of the solution.</param>
        /// <returns>Residual gradients for outputs and derivatives.</returns>
        PDEResidualGradient<T> ComputeResidualGradient(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives);
    }

    /// <summary>
    /// Provides gradients for boundary residuals with respect to outputs and derivatives.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    [AiDotNet.Configuration.YamlConfigurable("BoundaryConditionGradient")]
    public interface IBoundaryConditionGradient<T>
    {
        /// <summary>
        /// Computes gradients of the boundary residual with respect to outputs and derivatives.
        /// </summary>
        /// <param name="inputs">The boundary point coordinates.</param>
        /// <param name="outputs">The predicted solution values.</param>
        /// <param name="derivatives">The derivatives of the solution.</param>
        /// <returns>Residual gradients for outputs and derivatives.</returns>
        PDEResidualGradient<T> ComputeBoundaryResidualGradient(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives);
    }

    /// <summary>
    /// Holds gradients of a residual with respect to outputs and derivatives.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    public sealed class PDEResidualGradient<T>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PDEResidualGradient{T}"/> class.
        /// </summary>
        /// <param name="outputDimension">The output dimension.</param>
        /// <param name="inputDimension">The input dimension.</param>
        public PDEResidualGradient(int outputDimension, int inputDimension)
        {
            OutputGradients = new T[outputDimension];
            FirstDerivatives = new T[outputDimension, inputDimension];
            SecondDerivatives = new T[outputDimension, inputDimension, inputDimension];
            ThirdDerivatives = new T[outputDimension, inputDimension, inputDimension, inputDimension];
        }

        /// <summary>
        /// Gradient of residual with respect to outputs.
        /// </summary>
        public T[] OutputGradients { get; }

        /// <summary>
        /// Gradient of residual with respect to first derivatives.
        /// </summary>
        public T[,] FirstDerivatives { get; }

        /// <summary>
        /// Gradient of residual with respect to second derivatives.
        /// </summary>
        public T[,,] SecondDerivatives { get; }

        /// <summary>
        /// Gradient of residual with respect to third derivatives.
        /// Used for higher-order PDEs like Korteweg-de Vries equation.
        /// </summary>
        public T[,,,] ThirdDerivatives { get; }
    }

    /// <summary>
    /// Holds the derivatives needed for PDE computation.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Derivatives tell us how fast a function is changing. For PDEs, we need:
    /// - First derivatives (gradient): How fast the solution changes in each direction
    /// - Second derivatives (Hessian): How fast the rate of change itself is changing
    /// These are computed automatically using automatic differentiation.
    /// </remarks>
    public class PDEDerivatives<T>
    {
        /// <summary>
        /// First-order derivatives (gradient) of the output with respect to each input dimension.
        /// Shape: [output_dim, input_dim]
        /// </summary>
        public T[,]? FirstDerivatives { get; set; }

        /// <summary>
        /// Second-order derivatives (Hessian) of the output with respect to input dimensions.
        /// Shape: [output_dim, input_dim, input_dim]
        /// </summary>
        public T[,,]? SecondDerivatives { get; set; }

        /// <summary>
        /// Third-order derivatives of the output with respect to input dimensions.
        /// Shape: [output_dim, input_dim, input_dim, input_dim]
        /// Used for higher-order PDEs like Korteweg-de Vries equation.
        /// </summary>
        public T[,,,]? ThirdDerivatives { get; set; }

        /// <summary>
        /// Higher-order derivatives (4th order and above) if needed for the specific PDE.
        /// </summary>
        public T[,,,]? HigherDerivatives { get; set; }
    }

    /// <summary>
    /// Defines boundary conditions for a PDE problem.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Boundary conditions specify what happens at the edges of your problem domain.
    /// For example, in a heat equation, you might specify the temperature at the boundaries of a rod.
    /// Common types:
    /// - Dirichlet: The value is specified at the boundary (e.g., temperature = 100°C)
    /// - Neumann: The derivative is specified at the boundary (e.g., heat flux = 0, meaning insulated)
    /// - Robin: A combination of value and derivative
    /// </remarks>
    [AiDotNet.Configuration.YamlConfigurable("BoundaryCondition")]
    public interface IBoundaryCondition<T>
    {
        /// <summary>
        /// Determines if a point is on the boundary.
        /// </summary>
        /// <param name="inputs">The coordinates to check.</param>
        /// <returns>True if the point is on the boundary.</returns>
        bool IsOnBoundary(T[] inputs);

        /// <summary>
        /// Computes the boundary condition residual.
        /// For a perfect solution, this should be zero at all boundary points.
        /// </summary>
        /// <param name="inputs">The boundary point coordinates.</param>
        /// <param name="outputs">The predicted solution at the boundary.</param>
        /// <param name="derivatives">The derivatives at the boundary (needed for Neumann/Robin conditions).</param>
        /// <returns>The boundary residual (should be zero for a perfect solution).</returns>
        T ComputeBoundaryResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives);

        /// <summary>
        /// Gets the name of the boundary (e.g., "Left Wall", "Top Edge").
        /// </summary>
        string Name { get; }
    }

    /// <summary>
    /// Defines initial conditions for time-dependent PDEs.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Initial conditions specify the state of the system at the starting time (t=0).
    /// For example, in a heat equation, you might specify the initial temperature distribution.
    /// </remarks>
    [AiDotNet.Configuration.YamlConfigurable("InitialCondition")]
    public interface IInitialCondition<T>
    {
        /// <summary>
        /// Determines if a point is at the initial time.
        /// </summary>
        /// <param name="inputs">The coordinates to check (typically the last dimension is time).</param>
        /// <returns>True if the point is at t=0.</returns>
        bool IsAtInitialTime(T[] inputs);

        /// <summary>
        /// Computes the initial condition value at the given spatial location.
        /// </summary>
        /// <param name="spatialInputs">The spatial coordinates (excluding time).</param>
        /// <returns>The initial value at that location.</returns>
        T[] ComputeInitialValue(T[] spatialInputs);

        /// <summary>
        /// Gets the name of the initial condition.
        /// </summary>
        string Name { get; }
    }
}
