using System;

namespace AiDotNet.PhysicsInformed.Benchmarks
{
    public class PdeBenchmarkOptions
    {
        public int SpatialPoints { get; set; } = 64;
        public int TimeSteps { get; set; } = 200;
        public double DomainStart { get; set; } = -1.0;
        public double DomainEnd { get; set; } = 1.0;
        public double FinalTime { get; set; } = 1.0;

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

    public sealed class BurgersBenchmarkOptions : PdeBenchmarkOptions
    {
        public double Viscosity { get; set; } = 0.01;
        public Func<double, double> InitialCondition { get; set; } = x => -Math.Sin(Math.PI * x);

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

    public sealed class AllenCahnBenchmarkOptions : PdeBenchmarkOptions
    {
        public double Epsilon { get; set; } = 0.01;
        public Func<double, double> InitialCondition { get; set; } = x => x * x * Math.Cos(Math.PI * x);

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
