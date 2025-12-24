using System;

namespace AiDotNet.PhysicsInformed.Benchmarks
{
    public static class OperatorBenchmarkSuite
    {
        public static OperatorBenchmarkResult RunSmoothingOperatorBenchmark(
            OperatorBenchmarkOptions options,
            Func<double[], double[]> predictor)
        {
            if (options == null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            if (predictor == null)
            {
                throw new ArgumentNullException(nameof(predictor));
            }

            options.Validate();

            var random = new Random(options.Seed);
            double sumSquared = 0.0;
            double sumTargetSquared = 0.0;
            double maxError = 0.0;
            int totalCount = 0;

            for (int sample = 0; sample < options.SampleCount; sample++)
            {
                var input = GenerateInput(random, options);
                var target = ApplyMovingAverage(input, options.SmoothingWindow);
                var prediction = predictor(input) ?? throw new InvalidOperationException("Predictor returned null.");

                if (prediction.Length != input.Length)
                {
                    throw new InvalidOperationException("Predictor output length does not match input length.");
                }

                for (int i = 0; i < prediction.Length; i++)
                {
                    double error = prediction[i] - target[i];
                    sumSquared += error * error;
                    sumTargetSquared += target[i] * target[i];
                    double absError = Math.Abs(error);
                    if (absError > maxError)
                    {
                        maxError = absError;
                    }
                }

                totalCount += prediction.Length;
            }

            return CreateOperatorResult(
                "MovingAverage",
                options.SpatialPoints,
                options.SampleCount,
                sumSquared,
                sumTargetSquared,
                maxError,
                totalCount);
        }

        public static OperatorBenchmarkResult RunPoissonOperatorBenchmark(
            PoissonOperatorBenchmarkOptions options,
            Func<double[,], double[,]> predictor)
        {
            if (options == null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            if (predictor == null)
            {
                throw new ArgumentNullException(nameof(predictor));
            }

            options.Validate();

            var dataset = GeneratePoissonDataset(options);
            return EvaluateOperatorDataset(dataset, predictor);
        }

        public static OperatorBenchmarkResult RunDarcyOperatorBenchmark(
            DarcyOperatorBenchmarkOptions options,
            Func<double[,], double[,]> predictor)
        {
            if (options == null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            if (predictor == null)
            {
                throw new ArgumentNullException(nameof(predictor));
            }

            options.Validate();

            var dataset = GenerateDarcyDataset(options);
            return EvaluateOperatorDataset(dataset, predictor);
        }

        public static OperatorDataset2D GeneratePoissonDataset(PoissonOperatorBenchmarkOptions options)
        {
            if (options == null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            options.Validate();

            int gridSize = options.GridSize;
            var inputs = new double[options.SampleCount, gridSize, gridSize];
            var outputs = new double[options.SampleCount, gridSize, gridSize];
            var random = new Random(options.Seed);

            for (int sample = 0; sample < options.SampleCount; sample++)
            {
                var forcing = GenerateRandomField(random, gridSize, options.MaxFrequency, scale: 1.0);
                var solution = SolvePoisson(forcing, options.MaxIterations, options.Tolerance);
                CopyToTensor(forcing, inputs, sample);
                CopyToTensor(solution, outputs, sample);
            }

            return new OperatorDataset2D
            {
                OperatorName = "Poisson",
                GridSize = gridSize,
                SampleCount = options.SampleCount,
                Inputs = inputs,
                Outputs = outputs
            };
        }

        public static OperatorDataset2D GenerateDarcyDataset(DarcyOperatorBenchmarkOptions options)
        {
            if (options == null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            options.Validate();

            int gridSize = options.GridSize;
            var inputs = new double[options.SampleCount, gridSize, gridSize];
            var outputs = new double[options.SampleCount, gridSize, gridSize];
            var random = new Random(options.Seed);

            for (int sample = 0; sample < options.SampleCount; sample++)
            {
                var logField = GenerateRandomField(random, gridSize, options.MaxFrequency, options.LogPermeabilityScale);
                var permeability = ApplyExponential(logField);
                var solution = SolveDarcy(permeability, options.ForcingValue, options.MaxIterations, options.Tolerance);
                CopyToTensor(permeability, inputs, sample);
                CopyToTensor(solution, outputs, sample);
            }

            return new OperatorDataset2D
            {
                OperatorName = "DarcyFlow",
                GridSize = gridSize,
                SampleCount = options.SampleCount,
                Inputs = inputs,
                Outputs = outputs
            };
        }

        public static double[] ApplyMovingAverage(double[] input, int window)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (input.Length == 0)
            {
                return Array.Empty<double>();
            }

            int n = input.Length;
            int adjustedWindow = Math.Min(Math.Max(1, window), n);
            int radius = adjustedWindow / 2;

            var output = new double[n];
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                int count = 0;

                for (int offset = -radius; offset <= radius; offset++)
                {
                    int index = (i + offset + n) % n;
                    sum += input[index];
                    count++;
                }

                output[i] = sum / count;
            }

            return output;
        }

        internal static double[,] SolvePoisson(double[,] forcing, int maxIterations, double tolerance)
        {
            if (forcing == null)
            {
                throw new ArgumentNullException(nameof(forcing));
            }

            int n = forcing.GetLength(0);
            if (n != forcing.GetLength(1))
            {
                throw new ArgumentException("Poisson solver expects a square grid.", nameof(forcing));
            }

            var current = new double[n, n];
            var next = new double[n, n];
            double dx = 1.0 / (n - 1);
            double dx2 = dx * dx;

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                double maxDelta = 0.0;

                for (int i = 1; i < n - 1; i++)
                {
                    for (int j = 1; j < n - 1; j++)
                    {
                        double value = 0.25 * (
                            current[i - 1, j] + current[i + 1, j] +
                            current[i, j - 1] + current[i, j + 1] -
                            dx2 * forcing[i, j]);

                        double delta = Math.Abs(value - current[i, j]);
                        if (delta > maxDelta)
                        {
                            maxDelta = delta;
                        }

                        next[i, j] = value;
                    }
                }

                var temp = current;
                current = next;
                next = temp;

                if (maxDelta < tolerance)
                {
                    break;
                }
            }

            return current;
        }

        internal static double[,] SolveDarcy(double[,] permeability, double forcingValue, int maxIterations, double tolerance)
        {
            if (permeability == null)
            {
                throw new ArgumentNullException(nameof(permeability));
            }

            int n = permeability.GetLength(0);
            if (n != permeability.GetLength(1))
            {
                throw new ArgumentException("Darcy solver expects a square grid.", nameof(permeability));
            }

            var current = new double[n, n];
            var next = new double[n, n];
            double dx = 1.0 / (n - 1);
            double dx2 = dx * dx;

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                double maxDelta = 0.0;

                for (int i = 1; i < n - 1; i++)
                {
                    for (int j = 1; j < n - 1; j++)
                    {
                        double axp = 0.5 * (permeability[i, j] + permeability[i + 1, j]);
                        double axm = 0.5 * (permeability[i, j] + permeability[i - 1, j]);
                        double ayp = 0.5 * (permeability[i, j] + permeability[i, j + 1]);
                        double aym = 0.5 * (permeability[i, j] + permeability[i, j - 1]);

                        double denom = axp + axm + ayp + aym;
                        double value = (axp * current[i + 1, j] +
                                        axm * current[i - 1, j] +
                                        ayp * current[i, j + 1] +
                                        aym * current[i, j - 1] +
                                        forcingValue * dx2) / denom;

                        double delta = Math.Abs(value - current[i, j]);
                        if (delta > maxDelta)
                        {
                            maxDelta = delta;
                        }

                        next[i, j] = value;
                    }
                }

                var temp = current;
                current = next;
                next = temp;

                if (maxDelta < tolerance)
                {
                    break;
                }
            }

            return current;
        }

        private static double[] GenerateInput(Random random, OperatorBenchmarkOptions options)
        {
            var grid = CreateUnitGrid(options.SpatialPoints);
            var coefficients = new double[options.MaxFrequency];
            for (int k = 0; k < options.MaxFrequency; k++)
            {
                coefficients[k] = (random.NextDouble() * 2.0) - 1.0;
            }

            var input = new double[options.SpatialPoints];
            for (int i = 0; i < input.Length; i++)
            {
                double value = 0.0;
                for (int k = 0; k < options.MaxFrequency; k++)
                {
                    double frequency = k + 1;
                    value += coefficients[k] * Math.Sin(2.0 * Math.PI * frequency * grid[i]);
                }

                input[i] = value / options.MaxFrequency;
            }

            return input;
        }

        private static double[] CreateUnitGrid(int points)
        {
            var grid = new double[points];
            double dx = 1.0 / (points - 1);
            for (int i = 0; i < points; i++)
            {
                grid[i] = i * dx;
            }

            return grid;
        }

        private static OperatorBenchmarkResult EvaluateOperatorDataset(
            OperatorDataset2D dataset,
            Func<double[,], double[,]> predictor)
        {
            double sumSquared = 0.0;
            double sumTargetSquared = 0.0;
            double maxError = 0.0;
            int totalCount = 0;

            for (int sample = 0; sample < dataset.SampleCount; sample++)
            {
                var input = ExtractSample(dataset.Inputs, sample);
                var target = ExtractSample(dataset.Outputs, sample);
                var prediction = predictor(input) ?? throw new InvalidOperationException("Predictor returned null.");

                if (prediction.GetLength(0) != dataset.GridSize || prediction.GetLength(1) != dataset.GridSize)
                {
                    throw new InvalidOperationException("Predictor output shape does not match input grid.");
                }

                for (int i = 0; i < dataset.GridSize; i++)
                {
                    for (int j = 0; j < dataset.GridSize; j++)
                    {
                        double error = prediction[i, j] - target[i, j];
                        sumSquared += error * error;
                        sumTargetSquared += target[i, j] * target[i, j];
                        double absError = Math.Abs(error);
                        if (absError > maxError)
                        {
                            maxError = absError;
                        }
                    }
                }

                totalCount += dataset.GridSize * dataset.GridSize;
            }

            return CreateOperatorResult(
                dataset.OperatorName,
                dataset.GridSize,
                dataset.SampleCount,
                sumSquared,
                sumTargetSquared,
                maxError,
                totalCount);
        }

        private static OperatorBenchmarkResult CreateOperatorResult(
            string operatorName,
            int spatialPoints,
            int sampleCount,
            double sumSquared,
            double sumTargetSquared,
            double maxError,
            int totalCount)
        {
            double safeCount = Math.Max(1, totalCount);
            double mse = sumSquared / safeCount;
            double relativeL2 = sumTargetSquared > 0.0
                ? Math.Sqrt(sumSquared / sumTargetSquared)
                : 0.0;

            return new OperatorBenchmarkResult
            {
                OperatorName = operatorName,
                SpatialPoints = spatialPoints,
                SampleCount = sampleCount,
                Mse = mse,
                L2Error = Math.Sqrt(mse),
                RelativeL2Error = relativeL2,
                MaxError = maxError
            };
        }

        private static double[,] GenerateRandomField(Random random, int gridSize, int maxFrequency, double scale)
        {
            var coefficients = new double[maxFrequency, maxFrequency];
            for (int k = 0; k < maxFrequency; k++)
            {
                for (int l = 0; l < maxFrequency; l++)
                {
                    coefficients[k, l] = (random.NextDouble() * 2.0 - 1.0);
                }
            }

            var field = new double[gridSize, gridSize];
            double norm = Math.Max(1.0, maxFrequency * maxFrequency);

            for (int i = 0; i < gridSize; i++)
            {
                double x = (double)i / (gridSize - 1);
                for (int j = 0; j < gridSize; j++)
                {
                    double y = (double)j / (gridSize - 1);
                    double sum = 0.0;
                    for (int k = 0; k < maxFrequency; k++)
                    {
                        double kx = 2.0 * Math.PI * (k + 1) * x;
                        for (int l = 0; l < maxFrequency; l++)
                        {
                            double ly = 2.0 * Math.PI * (l + 1) * y;
                            sum += coefficients[k, l] * Math.Sin(kx) * Math.Sin(ly);
                        }
                    }

                    field[i, j] = (sum / norm) * scale;
                }
            }

            return field;
        }

        private static double[,] ApplyExponential(double[,] field)
        {
            int n0 = field.GetLength(0);
            int n1 = field.GetLength(1);
            var output = new double[n0, n1];

            for (int i = 0; i < n0; i++)
            {
                for (int j = 0; j < n1; j++)
                {
                    output[i, j] = Math.Exp(field[i, j]);
                }
            }

            return output;
        }

        private static void CopyToTensor(double[,] source, double[,,] destination, int sampleIndex)
        {
            int n0 = source.GetLength(0);
            int n1 = source.GetLength(1);

            for (int i = 0; i < n0; i++)
            {
                for (int j = 0; j < n1; j++)
                {
                    destination[sampleIndex, i, j] = source[i, j];
                }
            }
        }

        private static double[,] ExtractSample(double[,,] source, int sampleIndex)
        {
            int n0 = source.GetLength(1);
            int n1 = source.GetLength(2);
            var output = new double[n0, n1];

            for (int i = 0; i < n0; i++)
            {
                for (int j = 0; j < n1; j++)
                {
                    output[i, j] = source[sampleIndex, i, j];
                }
            }

            return output;
        }
    }
}
