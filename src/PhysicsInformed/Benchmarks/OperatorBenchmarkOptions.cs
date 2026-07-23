using System;

namespace AiDotNet.PhysicsInformed.Benchmarks
{
    public sealed class OperatorBenchmarkOptions
    {
        public int SpatialPoints { get; set; } = 64;
        public int SampleCount { get; set; } = 32;
        public int MaxFrequency { get; set; } = 3;
        public int SmoothingWindow { get; set; } = 5;
        public int Seed { get; set; } = 42;

        public void Validate()
        {
            if (SpatialPoints < 4)
            {
                throw new ArgumentOutOfRangeException(nameof(SpatialPoints), "SpatialPoints must be at least 4.");
            }

            if (SampleCount < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(SampleCount), "SampleCount must be at least 1.");
            }

            if (MaxFrequency < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(MaxFrequency), "MaxFrequency must be at least 1.");
            }

            if (SmoothingWindow < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(SmoothingWindow), "SmoothingWindow must be at least 1.");
            }
        }
    }
}
