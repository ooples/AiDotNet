using System;

namespace AiDotNet.PhysicsInformed.Benchmarks
{
    /// <summary>
    /// Configures the 2D Darcy-flow operator-learning dataset that <see cref="OperatorBenchmarkSuite"/> generates: each
    /// sample pairs a random permeability field a(x, y) with the pressure u solving -∇·(a∇u) = f on the unit square.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The permeability is exp of a random sine series, and u is solved by
    /// iterating to <see cref="Tolerance"/> with zero boundary values. A neural operator learns the map a ↦ u.
    /// </para>
    /// <para><b>For Beginners:</b> Think of water seeping through rock of varying porosity. The input image is how easily
    /// water flows at each point; the output image is the resulting pressure. A model that learns this mapping can
    /// replace a slow physics solver.
    /// </para>
    /// </remarks>
    public sealed class DarcyOperatorBenchmarkOptions
    {
        /// <summary>Gets or sets the number of grid points per side of the unit square.</summary>
        /// <value>Defaults to 64; at least 4.</value>
        public int GridSize { get; set; } = 64;

        /// <summary>Gets or sets the number of (permeability, pressure) pairs generated.</summary>
        /// <value>Defaults to 16; at least 1.</value>
        public int SampleCount { get; set; } = 16;

        /// <summary>Gets or sets the highest sine mode per axis in the random log-permeability series.</summary>
        /// <value>Defaults to 4; at least 1. Higher values give rougher permeability fields.</value>
        public int MaxFrequency { get; set; } = 4;

        /// <summary>Gets or sets the most solver sweeps run for each sample.</summary>
        /// <value>Defaults to 3000; at least 1.</value>
        public int MaxIterations { get; set; } = 3000;

        /// <summary>Gets or sets the largest change of any grid value in a sweep at which the solver stops.</summary>
        /// <value>Defaults to 1e-6; must be positive.</value>
        public double Tolerance { get; set; } = 1e-6;

        /// <summary>Gets or sets the constant source term f on the right-hand side.</summary>
        /// <value>Defaults to 1.</value>
        public double ForcingValue { get; set; } = 1.0;

        /// <summary>Gets or sets the amplitude of the random log-permeability field before it is exponentiated.</summary>
        /// <value>Defaults to 0.5; must be positive. Larger values give higher permeability contrast.</value>
        public double LogPermeabilityScale { get; set; } = 0.5;

        /// <summary>Gets or sets the seed of the random fields, so a dataset can be regenerated exactly.</summary>
        /// <value>Defaults to 42.</value>
        public int Seed { get; set; } = 42;

        /// <summary>Throws <see cref="ArgumentOutOfRangeException"/> for any value outside its documented range.</summary>
        public void Validate()
        {
            if (GridSize < 4)
            {
                throw new ArgumentOutOfRangeException(nameof(GridSize), "GridSize must be at least 4.");
            }

            if (SampleCount < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(SampleCount), "SampleCount must be at least 1.");
            }

            if (MaxFrequency < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(MaxFrequency), "MaxFrequency must be at least 1.");
            }

            if (MaxIterations < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(MaxIterations), "MaxIterations must be at least 1.");
            }

            if (Tolerance <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(Tolerance), "Tolerance must be positive.");
            }

            if (LogPermeabilityScale <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(LogPermeabilityScale), "LogPermeabilityScale must be positive.");
            }
        }
    }
}
