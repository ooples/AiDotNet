using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Defines the interface for numerical solvers used to solve Stochastic Differential Equations (SDEs) and Ordinary Differential Equations (ODEs).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This interface provides a contract for implementing various numerical integration methods for solving
    /// differential equations. These solvers are essential for diffusion models that use continuous-time
    /// formulations, such as Score-based SDE models.
    /// </para>
    /// <para><b>For Beginners:</b> Think of a solver as a numerical calculator for equations that describe change over time.
    /// 
    /// Imagine you're tracking the position of a car:
    /// - You know the car's current position and speed
    /// - You want to predict where it will be after some time
    /// - A solver helps calculate this by taking small time steps
    /// 
    /// In diffusion models:
    /// - We start with noisy data
    /// - We want to gradually remove noise over time
    /// - The solver calculates each small step of this process
    /// 
    /// Different solvers offer different trade-offs:
    /// - Simple solvers (like Euler) are fast but less accurate
    /// - Complex solvers (like RK45) are more accurate but slower
    /// - The choice depends on your accuracy and speed requirements
    /// </para>
    /// </remarks>
    public interface ISolver
    {
        /// <summary>
        /// Performs a single integration step for solving a differential equation.
        /// </summary>
        /// <param name="x">The current state of the system.</param>
        /// <param name="t">The current time.</param>
        /// <param name="dt">The time step size.</param>
        /// <param name="drift">The drift function that defines deterministic evolution.</param>
        /// <param name="diffusion">The diffusion coefficient function that scales random noise.</param>
        /// <param name="random">Random number generator for stochastic components.</param>
        /// <returns>The new state after taking one integration step.</returns>
        /// <remarks>
        /// <para>
        /// This method implements one step of numerical integration for equations of the form:
        /// dx = drift(x,t)dt + diffusion(t)dW
        /// where dW represents random noise (Brownian motion).
        /// </para>
        /// <para><b>For Beginners:</b> This method calculates the next position given the current position and velocities.
        /// 
        /// Breaking down the parameters:
        /// - x: Where we are now (current state)
        /// - t: What time it is now
        /// - dt: How big a time step to take
        /// - drift: How the system naturally moves (like gravity pulling things down)
        /// - diffusion: How much random noise affects the movement
        /// - random: Source of randomness for the noise
        /// 
        /// The method returns where we'll be after taking one small step forward in time.
        /// 
        /// For deterministic equations (ODEs), the diffusion is zero.
        /// For stochastic equations (SDEs), both drift and diffusion contribute.
        /// </para>
        /// </remarks>
        Tensor<double> Step(
            Tensor<double> x, 
            double t, 
            double dt,
            Func<Tensor<double>, double, Tensor<double>> drift,
            Func<double, double> diffusion,
            Random random);
    }
}