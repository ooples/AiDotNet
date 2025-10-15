using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.DiffusionModels.Solvers
{
    /// <summary>
    /// Implements the Runge-Kutta 4th/5th order method for numerical solution of Ordinary Differential Equations (ODEs).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The RK45 solver provides a high-accuracy method for solving deterministic differential equations.
    /// While it ignores the stochastic component (diffusion term), it's particularly useful for solving
    /// probability flow ODEs in score-based diffusion models, which provide deterministic sampling paths.
    /// </para>
    /// <para><b>For Beginners:</b> This is a very accurate method for solving equations without randomness.
    /// 
    /// Think of it like plotting a very precise trajectory:
    /// - Instead of taking one big step, it takes multiple small "test" steps
    /// - It averages these test steps to find the best path forward
    /// - This gives much more accurate results than simple methods
    /// 
    /// The "45" means it uses formulas of 4th and 5th order accuracy.
    /// 
    /// Best used for:
    /// - Deterministic sampling in diffusion models
    /// - When you need high accuracy
    /// - When the equation has no random components
    /// 
    /// Not suitable for:
    /// - Equations with randomness (use Euler-Maruyama instead)
    /// - When speed is more important than accuracy
    /// </para>
    /// </remarks>
    public class RK45Solver : ISolver
    {
        private readonly double _tolerance;
        private readonly double _minTimeStep;
        private readonly double _maxTimeStep;
        private readonly bool _adaptiveStepSize;
        
        /// <summary>
        /// Initializes a new instance of the RK45Solver class.
        /// </summary>
        /// <param name="tolerance">Error tolerance for adaptive step size (default: 1e-6)</param>
        /// <param name="minTimeStep">Minimum allowed time step (default: 1e-8)</param>
        /// <param name="maxTimeStep">Maximum allowed time step (default: 0.1)</param>
        /// <param name="adaptiveStepSize">Whether to use adaptive step size control (default: false)</param>
        public RK45Solver(
            double tolerance = 1e-6, 
            double minTimeStep = 1e-8, 
            double maxTimeStep = 0.1,
            bool adaptiveStepSize = false)
        {
            if (tolerance <= 0)
                throw new ArgumentException("Tolerance must be positive", nameof(tolerance));
            if (minTimeStep <= 0)
                throw new ArgumentException("Minimum time step must be positive", nameof(minTimeStep));
            if (maxTimeStep <= minTimeStep)
                throw new ArgumentException("Maximum time step must be greater than minimum time step", nameof(maxTimeStep));
            
            _tolerance = tolerance;
            _minTimeStep = minTimeStep;
            _maxTimeStep = maxTimeStep;
            _adaptiveStepSize = adaptiveStepSize;
        }
        
        /// <summary>
        /// Performs one step of the Runge-Kutta 4th order integration scheme.
        /// </summary>
        /// <param name="x">Current state vector</param>
        /// <param name="t">Current time</param>
        /// <param name="dt">Time step size</param>
        /// <param name="drift">Drift function (ODE right-hand side)</param>
        /// <param name="diffusion">Diffusion coefficient (ignored for deterministic ODE)</param>
        /// <param name="random">Random number generator (unused for deterministic ODE)</param>
        /// <returns>Updated state vector</returns>
        /// <exception cref="ArgumentNullException">Thrown when required parameters are null</exception>
        /// <exception cref="ArgumentException">Thrown when time step is invalid</exception>
        public Tensor<double> Step(
            Tensor<double> x, 
            double t, 
            double dt,
            Func<Tensor<double>, double, Tensor<double>> drift,
            Func<double, double> diffusion,
            Random random)
        {
            // Validate inputs
            if (x == null)
                throw new ArgumentNullException(nameof(x));
            if (drift == null)
                throw new ArgumentNullException(nameof(drift));
            
            if (dt < _minTimeStep || dt > _maxTimeStep)
                throw new ArgumentException($"Time step must be between {_minTimeStep} and {_maxTimeStep}", nameof(dt));
            
            if (double.IsNaN(t) || double.IsInfinity(t))
                throw new ArgumentException("Time must be a valid finite number", nameof(t));
            
            try
            {
                if (_adaptiveStepSize)
                {
                    return AdaptiveRK45Step(x, t, dt, drift);
                }
                else
                {
                    return FixedRK4Step(x, t, dt, drift);
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to perform RK45 step: {ex.Message}", ex);
            }
        }
        
        /// <summary>
        /// Performs a fixed-step RK4 integration.
        /// </summary>
        private Tensor<double> FixedRK4Step(
            Tensor<double> x, 
            double t, 
            double dt,
            Func<Tensor<double>, double, Tensor<double>> drift)
        {
            // Classical RK4 formula:
            // k1 = f(x_n, t_n)
            // k2 = f(x_n + dt/2 * k1, t_n + dt/2)
            // k3 = f(x_n + dt/2 * k2, t_n + dt/2)
            // k4 = f(x_n + dt * k3, t_n + dt)
            // x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            var k1 = drift(x, t);
            ValidateTensor(k1, "k1");
            
            var k2 = drift(x.Add(k1.Multiply(dt / 2)), t + dt / 2);
            ValidateTensor(k2, "k2");
            
            var k3 = drift(x.Add(k2.Multiply(dt / 2)), t + dt / 2);
            ValidateTensor(k3, "k3");
            
            var k4 = drift(x.Add(k3.Multiply(dt)), t + dt);
            ValidateTensor(k4, "k4");
            
            // Weighted average of slopes
            var increment = k1.Add(k2.Multiply(2))
                              .Add(k3.Multiply(2))
                              .Add(k4)
                              .Multiply(dt / 6);
            
            var result = x.Add(increment);
            ValidateTensor(result, "result");
            
            return result;
        }
        
        /// <summary>
        /// Performs an adaptive-step RK45 integration with error control.
        /// </summary>
        private Tensor<double> AdaptiveRK45Step(
            Tensor<double> x, 
            double t, 
            double dt,
            Func<Tensor<double>, double, Tensor<double>> drift)
        {
            // Dormand-Prince RK45 coefficients
            const double a21 = 1.0 / 5.0;
            const double a31 = 3.0 / 40.0, a32 = 9.0 / 40.0;
            const double a41 = 44.0 / 45.0, a42 = -56.0 / 15.0, a43 = 32.0 / 9.0;
            const double a51 = 19372.0 / 6561.0, a52 = -25360.0 / 2187.0, a53 = 64448.0 / 6561.0, a54 = -212.0 / 729.0;
            const double a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0, a63 = 46732.0 / 5247.0, a64 = 49.0 / 176.0, a65 = -5103.0 / 18656.0;
            
            // 5th order weights
            const double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0, b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;
            
            // 4th order weights for error estimation
            const double b1p = 5179.0 / 57600.0, b3p = 7571.0 / 16695.0, b4p = 393.0 / 640.0, b5p = -92097.0 / 339200.0, b6p = 187.0 / 2100.0, b7p = 1.0 / 40.0;
            
            // Compute intermediate stages
            var k1 = drift(x, t);
            var k2 = drift(x.Add(k1.Multiply(dt * a21)), t + dt * a21);
            var k3 = drift(x.Add(k1.Multiply(dt * a31)).Add(k2.Multiply(dt * a32)), t + dt * 0.3);
            var k4 = drift(x.Add(k1.Multiply(dt * a41)).Add(k2.Multiply(dt * a42)).Add(k3.Multiply(dt * a43)), t + dt * 0.8);
            var k5 = drift(x.Add(k1.Multiply(dt * a51)).Add(k2.Multiply(dt * a52)).Add(k3.Multiply(dt * a53)).Add(k4.Multiply(dt * a54)), t + dt * 8.0/9.0);
            var k6 = drift(x.Add(k1.Multiply(dt * a61)).Add(k2.Multiply(dt * a62)).Add(k3.Multiply(dt * a63)).Add(k4.Multiply(dt * a64)).Add(k5.Multiply(dt * a65)), t + dt);
            
            // 5th order solution
            var y5 = x.Add(k1.Multiply(dt * b1))
                      .Add(k3.Multiply(dt * b3))
                      .Add(k4.Multiply(dt * b4))
                      .Add(k5.Multiply(dt * b5))
                      .Add(k6.Multiply(dt * b6));
            
            // 4th order solution for error estimation
            var k7 = drift(y5, t + dt);
            var y4 = x.Add(k1.Multiply(dt * b1p))
                      .Add(k3.Multiply(dt * b3p))
                      .Add(k4.Multiply(dt * b4p))
                      .Add(k5.Multiply(dt * b5p))
                      .Add(k6.Multiply(dt * b6p))
                      .Add(k7.Multiply(dt * b7p));
            
            // Estimate error
            var error = EstimateError(y4, y5);
            
            // Accept step if error is within tolerance
            if (error <= _tolerance)
            {
                ValidateTensor(y5, "result");
                return y5;
            }
            else
            {
                // Reduce step size and retry
                var newDt = Math.Max(_minTimeStep, dt * 0.5);
                return AdaptiveRK45Step(x, t, newDt, drift);
            }
        }
        
        /// <summary>
        /// Estimates the error between two solutions.
        /// </summary>
        private double EstimateError(Tensor<double> y4, Tensor<double> y5)
        {
            var diff = y5.Subtract(y4);
            var error = 0.0;
            var data = diff.ToVector();
            
            for (int i = 0; i < data.Length; i++)
            {
                error = Math.Max(error, Math.Abs(data[i]));
            }
            
            return error;
        }
        
        /// <summary>
        /// Validates tensor for numerical stability.
        /// </summary>
        private void ValidateTensor(Tensor<double> tensor, string name)
        {
            var data = tensor.ToVector();
            for (int i = 0; i < data.Length; i++)
            {
                if (double.IsNaN(data[i]) || double.IsInfinity(data[i]))
                {
                    throw new InvalidOperationException($"Numerical instability in {name}: contains NaN or infinity values");
                }
            }
        }
    }
}