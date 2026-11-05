using System;

namespace AiDotNet.PhysicsInformed.Benchmarks
{
    /// <summary>
    /// The grid and time span shared by the 1D time-dependent PDE benchmarks of <see cref="PdeBenchmarkSuite"/>: a
    /// finite-difference reference solution is computed on this grid and a predictor u(x, t) is scored against it at
    /// <see cref="FinalTime"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These settings say where the line segment lies, how finely it is divided, and how far
    /// forward in time the reference simulation runs. The specific equations add their own coefficients.
    /// </para>
    /// </remarks>
    public class PdeBenchmarkOptions
    {
        /// <summary>Gets or sets the number of grid points across the domain, both ends included.</summary>
        /// <value>Defaults to 64; at least 3.</value>
        public int SpatialPoints { get; set; } = 64;

        /// <summary>Gets or sets the number of explicit time steps from t = 0 to <see cref="FinalTime"/>.</summary>
        /// <value>
        /// Defaults to 200; at least 1. The diffusion term must satisfy the CFL bound (coefficient × dt / dx² ≤ 0.5), and
        /// the reference solver throws if too few steps break it.
        /// </value>
        public int TimeSteps { get; set; } = 200;

        /// <summary>Gets or sets the left end of the spatial domain.</summary>
        /// <value>Defaults to -1; must be below <see cref="DomainEnd"/>.</value>
        public double DomainStart { get; set; } = -1.0;

        /// <summary>Gets or sets the right end of the spatial domain.</summary>
        /// <value>Defaults to 1; must be above <see cref="DomainStart"/>.</value>
        public double DomainEnd { get; set; } = 1.0;

        /// <summary>Gets or sets the time at which the predictor is compared with the reference solution.</summary>
        /// <value>Defaults to 1; must be positive.</value>
        public double FinalTime { get; set; } = 1.0;

        /// <summary>Throws <see cref="ArgumentOutOfRangeException"/> for any value outside its documented range.</summary>
        public virtual void Validate()
        {
            if (SpatialPoints < 3)
            {
                throw new ArgumentOutOfRangeException(nameof(SpatialPoints), "SpatialPoints must be at least 3.");
            }

            if (TimeSteps < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(TimeSteps), "TimeSteps must be at least 1.");
            }

            if (DomainEnd <= DomainStart)
            {
                throw new ArgumentOutOfRangeException(nameof(DomainEnd), "DomainEnd must be greater than DomainStart.");
            }

            if (FinalTime <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(FinalTime), "FinalTime must be positive.");
            }
        }
    }

    /// <summary>
    /// The viscous Burgers benchmark, u_t + u u_x = ν u_xx, the shock-forming equation of Raissi et al. (2019)'s PINN
    /// paper.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A simple model of a fluid where faster parts catch up with slower ones and form a
    /// steep front; the viscosity smooths that front.
    /// </para>
    /// </remarks>
    public sealed class BurgersBenchmarkOptions : PdeBenchmarkOptions
    {
        /// <summary>Gets or sets the viscosity ν of the diffusion term.</summary>
        /// <value>Defaults to 0.01; must be non-negative. Smaller values give sharper fronts.</value>
        public double Viscosity { get; set; } = 0.01;

        /// <summary>Gets or sets u(x, 0).</summary>
        /// <value>Defaults to -sin(πx), the PINN paper's initial condition.</value>
        public Func<double, double> InitialCondition { get; set; } = x => -Math.Sin(Math.PI * x);

        /// <summary>Validates the shared grid and the Burgers coefficients.</summary>
        public override void Validate()
        {
            base.Validate();

            if (Viscosity < 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(Viscosity), "Viscosity must be non-negative.");
            }

            if (InitialCondition == null)
            {
                throw new ArgumentNullException(nameof(InitialCondition));
            }
        }
    }

    /// <summary>
    /// The Allen-Cahn benchmark, u_t = ε² u_xx + u - u³, a reaction-diffusion equation whose solution separates into
    /// phases at ±1 joined by thin interfaces.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Models two materials separating, like oil and water: the value drifts towards -1 or 1,
    /// and ε sets how wide the boundary between the two regions is.
    /// </para>
    /// </remarks>
    public sealed class AllenCahnBenchmarkOptions : PdeBenchmarkOptions
    {
        /// <summary>Gets or sets the interface width ε (the diffusion coefficient is ε²).</summary>
        /// <value>Defaults to 0.01; must be positive.</value>
        public double Epsilon { get; set; } = 0.01;

        /// <summary>Gets or sets u(x, 0).</summary>
        /// <value>Defaults to x² cos(πx), the initial condition of the Allen-Cahn PINN benchmarks.</value>
        public Func<double, double> InitialCondition { get; set; } = x => x * x * Math.Cos(Math.PI * x);

        /// <summary>Validates the shared grid and the Allen-Cahn coefficient.</summary>
        public override void Validate()
        {
            base.Validate();

            if (Epsilon <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(Epsilon), "Epsilon must be positive.");
            }

            if (InitialCondition == null)
            {
                throw new ArgumentNullException(nameof(InitialCondition));
            }
        }
    }
}
