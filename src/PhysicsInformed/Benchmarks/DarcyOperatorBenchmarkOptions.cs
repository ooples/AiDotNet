using System;

namespace AiDotNet.PhysicsInformed.Benchmarks
{
    public sealed class DarcyOperatorBenchmarkOptions
    {
        public int GridSize { get; set; } = 64;
        public int SampleCount { get; set; } = 16;
        public int MaxFrequency { get; set; } = 4;
        public int MaxIterations { get; set; } = 3000;
        public double Tolerance { get; set; } = 1e-6;
        public double ForcingValue { get; set; } = 1.0;
        public double LogPermeabilityScale { get; set; } = 0.5;
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

            if (LogPermeabilityScale <= 0.0)
            {
                throw new ArgumentOutOfRangeException(nameof(LogPermeabilityScale), "LogPermeabilityScale must be positive.");
            }
        }
    }
}
