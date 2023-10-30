using System;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.Regression
{
    public sealed class MultipleRegression : IRegression<double[], double>
    {
        private double YIntercept { get; set; }
        private double[] Coefficients { get; set; } = Array.Empty<double>();
        private MultipleRegressionOptions RegressionOptions { get; }

        public double[] Predictions { get; private set; }
        public IMetrics Metrics { get; private set; }

        public MultipleRegression(double[][] inputs, double[] outputs, MultipleRegressionOptions? regressionOptions = null)
        {
            ValidationHelper.CheckForNullItems(inputs, outputs);
            var inputSize = inputs[0].Length;
            ValidationHelper.CheckForInvalidInputSize(inputSize, outputs.Length);

            RegressionOptions = regressionOptions ?? new MultipleRegressionOptions();

            var trainingPctSize = RegressionOptions.TrainingPctSize;
            ValidationHelper.CheckForInvalidTrainingPctSize(trainingPctSize);
            var trainingSize = (int)Math.Floor(inputSize * trainingPctSize / 100);
            ValidationHelper.CheckForInvalidTrainingSizes(trainingSize, inputSize - trainingSize, Math.Max(2, inputs.Length), trainingPctSize);

            var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) =
                PrepareData(inputs, outputs, trainingSize, RegressionOptions.Normalization);
            Fit(trainingInputs, trainingOutputs);
            Predictions = Transform(oosInputs);
            Metrics = new Metrics(Predictions, oosOutputs, inputs.Length);
        }

        internal override void Fit(double[][] inputs, double[] outputs)
        {
            var m = Matrix<double>.Build;
            var inputMatrix = RegressionOptions.MatrixLayout switch
            {
                MatrixLayout.ColumnArrays => m.DenseOfColumnArrays(inputs),
                MatrixLayout.RowArrays => m.DenseOfRowArrays(inputs),
                _ => m.DenseOfColumnArrays(inputs)
            };
            var outputVector = CreateVector.Dense(outputs);
            var result = CreateVector.Dense<double>(inputs.Length + (RegressionOptions.UseIntercept ? 1 : 0));

            if (RegressionOptions.UseIntercept)
            {
                inputMatrix = RegressionOptions.MatrixLayout == MatrixLayout.ColumnArrays ? 
                    inputMatrix.InsertColumn(0, CreateVector.Dense(outputs.Length, Vector<double>.One)) :
                    inputMatrix.InsertRow(0, CreateVector.Dense(outputs.Length, Vector<double>.One));
            }

            switch (RegressionOptions.MatrixDecomposition)
            {
                case MatrixDecomposition.Cholesky:
                    inputMatrix.Cholesky().Solve(outputVector, result);
                    break;
                case MatrixDecomposition.Evd:
                    inputMatrix.Evd().Solve(outputVector, result);
                    break;
                case MatrixDecomposition.GramSchmidt:
                    inputMatrix.GramSchmidt().Solve(outputVector, result);
                    break;
                case MatrixDecomposition.Lu:
                    inputMatrix.LU().Solve(outputVector, result);
                    break;
                case MatrixDecomposition.Qr:
                    inputMatrix.QR().Solve(outputVector, result);
                    break;
                case MatrixDecomposition.Svd:
                    inputMatrix.Svd().Solve(outputVector, result);
                    break;
                default:
                    throw new ArgumentException("Invalid MatrixDecomposition option");
            }

            Coefficients = result.ToArray();
            YIntercept = 0;
        }

        internal override (double[][] trainingInputs, double[] trainingOutputs, double[][] oosInputs, double[] oosOutputs) 
            PrepareData(double[][] inputs, double[] outputs, int trainingSize, INormalization? normalization)
        {
            return normalization?.PrepareData(inputs, outputs, trainingSize) ?? NormalizationHelper.SplitData(inputs, outputs, trainingSize);
        }

        internal override double[] Transform(double[][] inputs)
        {
            var predictions = new double[inputs[0].Length];
            var inputsLength = inputs.Length;
            var inputs0Length = inputs[0].Length;

            Parallel.For(0, inputsLength, i =>
            {
                for (var j = 0; j < inputs0Length; j++)
                {
                    predictions[j] += YIntercept + Coefficients[i] * inputs[i][j];
                }
            });

            return predictions;
        }
    }
}
