using System;

namespace AiDotNet.PhysicsInformed.Benchmarks
{
    /// <summary>
    /// Configures the 2D Poisson operator-learning dataset that <see cref="OperatorBenchmarkSuite"/> generates: each
    /// sample pairs a random forcing field f(x, y) with the u solving -Δu = f on the unit square.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The forcing is a random sine series, and u is solved by iterating to <see cref="Tolerance"/> with zero boundary
    /// values. A neural operator learns the map f ↦ u, the standard first test for operator learning.
    /// </para>
    /// <para><b>For Beginners:</b> The input image is a heat source spread over a plate; the output is the steady
    /// temperature it produces. A model that learns this mapping can stand in for the solver.
    /// </para>
    /// </remarks>
    public sealed class PoissonOperatorBenchmarkOptions
    {
        /// <summary>Gets or sets the number of grid points per side of the unit square.</summary>
        /// <value>Defaults to 64; at least 4.</value>
        public int GridSize { get; set; } = 64;

        /// <summary>Gets or sets the number of (forcing, solution) pairs generated.</summary>
        /// <value>Defaults to 16; at least 1.</value>
        public int SampleCount { get; set; } = 16;

        /// <summary>Gets or sets the highest sine mode per axis in the random forcing series.</summary>
        /// <value>Defaults to 4; at least 1. Higher values give rougher forcing fields.</value>
        public int MaxFrequency { get; set; } = 4;

        /// <summary>Gets or sets the most solver sweeps run for each sample.</summary>
        /// <value>Defaults to 2000; at least 1.</value>
        public int MaxIterations { get; set; } = 2000;

        /// <summary>Gets or sets the largest change of any grid value in a sweep at which the solver stops.</summary>
        /// <value>Defaults to 1e-6; must be positive.</value>
        public double Tolerance { get; set; } = 1e-6;

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
        }
    }
}
