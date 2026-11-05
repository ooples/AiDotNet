using System;

namespace AiDotNet.PhysicsInformed.Benchmarks
{
    /// <summary>
    /// Configures the 1D smoothing-operator benchmark of <see cref="OperatorBenchmarkSuite"/>: each input is a random
    /// sine series on [0, 1], and the target is its moving average, which a predictor is scored against.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A quick sanity check for an operator-learning model: given a wiggly curve, can it
    /// output the smoothed curve? The error is reported relative to the size of the targets.
    /// </para>
    /// </remarks>
    public sealed class OperatorBenchmarkOptions
    {
        /// <summary>Gets or sets the number of points each input curve is sampled at.</summary>
        /// <value>Defaults to 64; at least 4.</value>
        public int SpatialPoints { get; set; } = 64;

        /// <summary>Gets or sets the number of random curves scored.</summary>
        /// <value>Defaults to 32; at least 1.</value>
        public int SampleCount { get; set; } = 32;

        /// <summary>Gets or sets the number of sine modes in each random input curve.</summary>
        /// <value>Defaults to 3; at least 1.</value>
        public int MaxFrequency { get; set; } = 3;

        /// <summary>Gets or sets the width, in points, of the moving average that defines the target.</summary>
        /// <value>Defaults to 5; at least 1 (1 makes the target the input itself).</value>
        public int SmoothingWindow { get; set; } = 5;

        /// <summary>Gets or sets the seed of the random curves, so a run can be repeated exactly.</summary>
        /// <value>Defaults to 42.</value>
        public int Seed { get; set; } = 42;

        /// <summary>Throws <see cref="ArgumentOutOfRangeException"/> for any value outside its documented range.</summary>
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
