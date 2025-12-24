using System;

namespace AiDotNet.PhysicsInformed.Benchmarks
{
    public static class PdeBenchmarkSuite
    {
        public static PdeBenchmarkResult RunBurgers(
            BurgersBenchmarkOptions options,
            Func<double, double, double> predictor)
        {
            if (options == null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            options.Validate();
            ValidatePredictor(predictor);

            var baseline = FiniteDifferenceBaseline.SolveBurgers(options, out var grid);
            return ComputeError("Burgers Equation", options, grid, baseline, predictor);
        }

        public static PdeBenchmarkResult RunAllenCahn(
            AllenCahnBenchmarkOptions options,
            Func<double, double, double> predictor)
        {
            if (options == null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            options.Validate();
            ValidatePredictor(predictor);

            var baseline = FiniteDifferenceBaseline.SolveAllenCahn(options, out var grid);
            return ComputeError("Allen-Cahn Equation", options, grid, baseline, predictor);
        }

        private static void ValidatePredictor(Func<double, double, double> predictor)
        {
            if (predictor == null)
            {
                throw new ArgumentNullException(nameof(predictor));
            }
        }

        private static PdeBenchmarkResult ComputeError(
            string equationName,
            PdeBenchmarkOptions options,
            double[] grid,
            double[] baseline,
            Func<double, double, double> predictor)
        {
            double sumSquared = 0.0;
            double maxError = 0.0;

            for (int i = 0; i < grid.Length; i++)
            {
                double prediction = predictor(grid[i], options.FinalTime);
                double error = prediction - baseline[i];
                sumSquared += error * error;
                double absError = Math.Abs(error);
                if (absError > maxError)
                {
                    maxError = absError;
                }
            }

            return new PdeBenchmarkResult
            {
                EquationName = equationName,
                SpatialPoints = options.SpatialPoints,
                TimeSteps = options.TimeSteps,
                FinalTime = options.FinalTime,
                L2Error = Math.Sqrt(sumSquared / grid.Length),
                MaxError = maxError
            };
        }
    }

    internal static class FiniteDifferenceBaseline
    {
        public static double[] SolveBurgers(BurgersBenchmarkOptions options, out double[] grid)
        {
            options.Validate();
            grid = CreateSpatialGrid(options, out double dx);

            int n = grid.Length;
            double dt = options.FinalTime / options.TimeSteps;
            double invDx = 1.0 / dx;
            double invDx2 = invDx * invDx;

            var current = new double[n];
            var next = new double[n];

            for (int i = 0; i < n; i++)
            {
                current[i] = options.InitialCondition(grid[i]);
            }

            for (int step = 0; step < options.TimeSteps; step++)
            {
                for (int i = 0; i < n; i++)
                {
                    int im1 = (i - 1 + n) % n;
                    int ip1 = (i + 1) % n;

                    double dudx = (current[ip1] - current[im1]) * 0.5 * invDx;
                    double d2udx2 = (current[ip1] - (2.0 * current[i]) + current[im1]) * invDx2;

                    next[i] = current[i]
                        - (dt * current[i] * dudx)
                        + (options.Viscosity * dt * d2udx2);
                }

                var temp = current;
                current = next;
                next = temp;
            }

            return current;
        }

        public static double[] SolveAllenCahn(AllenCahnBenchmarkOptions options, out double[] grid)
        {
            options.Validate();
            grid = CreateSpatialGrid(options, out double dx);

            int n = grid.Length;
            double dt = options.FinalTime / options.TimeSteps;
            double invDx2 = 1.0 / (dx * dx);
            double epsilonSquared = options.Epsilon * options.Epsilon;

            var current = new double[n];
            var next = new double[n];

            for (int i = 0; i < n; i++)
            {
                current[i] = options.InitialCondition(grid[i]);
            }

            for (int step = 0; step < options.TimeSteps; step++)
            {
                for (int i = 0; i < n; i++)
                {
                    int im1 = (i - 1 + n) % n;
                    int ip1 = (i + 1) % n;

                    double d2udx2 = (current[ip1] - (2.0 * current[i]) + current[im1]) * invDx2;
                    double reaction = (current[i] * current[i] * current[i]) - current[i];

                    next[i] = current[i] + (dt * ((epsilonSquared * d2udx2) - reaction));
                }

                var temp = current;
                current = next;
                next = temp;
            }

            return current;
        }

        private static double[] CreateSpatialGrid(PdeBenchmarkOptions options, out double dx)
        {
            dx = (options.DomainEnd - options.DomainStart) / (options.SpatialPoints - 1);
            var grid = new double[options.SpatialPoints];
            for (int i = 0; i < grid.Length; i++)
            {
                grid[i] = options.DomainStart + (i * dx);
            }

            return grid;
        }
    }
}
