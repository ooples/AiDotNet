using System;

namespace AiDotNet.PhysicsInformed.Benchmarks
{
    public sealed class PoissonOperatorBenchmarkOptions
    {
        public int GridSize { get; set; } = 64;
        public int SampleCount { get; set; } = 16;
        public int MaxFrequency { get; set; } = 4;
        public int MaxIterations { get; set; } = 2000;
        public double Tolerance { get; set; } = 1e-6;
        public int Seed { get; set; } = 42;

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
