using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Grey Wolf Optimizer for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Inspired by grey wolf pack hunting behavior with alpha, beta, delta leaders.
/// Wolves encircle prey (optimal solution) guided by the three best solutions.
/// </para>
/// <para><b>For Beginners:</b> Like wolves hunting together, this algorithm
/// uses the three best current solutions (alpha, beta, delta) to guide the
/// pack toward better feature combinations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GreyWolfFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _populationSize;
    private readonly int _nIterations;
    private readonly int _nFeaturesToSelect;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, bool[], double>? _fitnessFunc;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private double _bestFitness;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public double BestFitness => _bestFitness;
    public override bool SupportsInverseTransform => false;

    public GreyWolfFS(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int nIterations = 100,
        Func<Matrix<T>, Vector<T>, bool[], double>? fitnessFunc = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 4)
            throw new ArgumentException("Population size must be at least 4.", nameof(populationSize));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nIterations = nIterations;
        _fitnessFunc = fitnessFunc;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GreyWolfFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize wolves (population)
        var wolves = new List<(double[] Position, double Fitness)>();
        for (int i = 0; i < _populationSize; i++)
        {
            var position = new double[p];
            for (int j = 0; j < p; j++)
                position[j] = random.NextDouble();

            var solution = PositionToSolution(position);
            double fitness = EvaluateFitness(data, target, solution);
            wolves.Add((position, fitness));
        }

        _featureImportances = new double[p];

        // Sort to get alpha, beta, delta
        wolves = wolves.OrderByDescending(w => w.Fitness).ToList();
        var alpha = wolves[0];
        var beta = wolves[1];
        var delta = wolves[2];
        _bestFitness = alpha.Fitness;

        // Main GWO loop
        for (int iter = 0; iter < _nIterations; iter++)
        {
            double a = 2 - iter * (2.0 / _nIterations);  // Linearly decreasing from 2 to 0

            for (int i = 3; i < _populationSize; i++)
            {
                var wolf = wolves[i];
                var newPosition = new double[p];

                for (int j = 0; j < p; j++)
                {
                    // Alpha contribution
                    double r1 = random.NextDouble();
                    double r2 = random.NextDouble();
                    double A1 = 2 * a * r1 - a;
                    double C1 = 2 * r2;
                    double D_alpha = Math.Abs(C1 * alpha.Position[j] - wolf.Position[j]);
                    double X1 = alpha.Position[j] - A1 * D_alpha;

                    // Beta contribution
                    r1 = random.NextDouble();
                    r2 = random.NextDouble();
                    double A2 = 2 * a * r1 - a;
                    double C2 = 2 * r2;
                    double D_beta = Math.Abs(C2 * beta.Position[j] - wolf.Position[j]);
                    double X2 = beta.Position[j] - A2 * D_beta;

                    // Delta contribution
                    r1 = random.NextDouble();
                    r2 = random.NextDouble();
                    double A3 = 2 * a * r1 - a;
                    double C3 = 2 * r2;
                    double D_delta = Math.Abs(C3 * delta.Position[j] - wolf.Position[j]);
                    double X3 = delta.Position[j] - A3 * D_delta;

                    // Average position
                    newPosition[j] = (X1 + X2 + X3) / 3;
                    newPosition[j] = Math.Max(0, Math.Min(1, newPosition[j]));  // Clamp to [0,1]
                }

                var newSolution = PositionToSolution(newPosition);
                double newFitness = EvaluateFitness(data, target, newSolution);

                // Update feature importances
                for (int j = 0; j < p; j++)
                    if (newSolution[j])
                        _featureImportances[j] += newFitness;

                wolves[i] = (newPosition, newFitness);
            }

            // Update alpha, beta, delta
            wolves = wolves.OrderByDescending(w => w.Fitness).ToList();
            if (wolves[0].Fitness > alpha.Fitness)
            {
                alpha = wolves[0];
                _bestFitness = alpha.Fitness;
            }
            if (wolves[1].Fitness > beta.Fitness)
                beta = wolves[1];
            if (wolves[2].Fitness > delta.Fitness)
                delta = wolves[2];
        }

        // Select from best (alpha) solution
        var bestSolution = PositionToSolution(alpha.Position);
        var selectedList = new List<int>();
        for (int j = 0; j < p; j++)
            if (bestSolution[j])
                selectedList.Add(j);

        if (selectedList.Count > _nFeaturesToSelect)
        {
            selectedList = selectedList
                .OrderByDescending(j => _featureImportances[j])
                .Take(_nFeaturesToSelect)
                .ToList();
        }
        else if (selectedList.Count == 0)
        {
            selectedList = _featureImportances
                .Select((imp, idx) => (imp, idx))
                .OrderByDescending(x => x.imp)
                .Take(_nFeaturesToSelect)
                .Select(x => x.idx)
                .ToList();
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private bool[] PositionToSolution(double[] position)
    {
        var solution = new bool[position.Length];
        double threshold = 0.5;

        // Ensure roughly nFeaturesToSelect features
        var sorted = position.Select((p, i) => (p, i)).OrderByDescending(x => x.p).ToList();
        int count = 0;
        foreach (var (prob, idx) in sorted)
        {
            if (prob > threshold || count < Math.Min(_nFeaturesToSelect, position.Length))
            {
                solution[idx] = true;
                count++;
            }
            if (count >= _nFeaturesToSelect + 3) break;
        }

        return solution;
    }

    private double EvaluateFitness(Matrix<T> data, Vector<T> target, bool[] mask)
    {
        if (_fitnessFunc is not null)
            return _fitnessFunc(data, target, mask);

        int n = data.Rows;
        int nSelected = mask.Count(b => b);
        if (nSelected == 0) return double.NegativeInfinity;

        double totalCorr = 0;
        for (int j = 0; j < mask.Length; j++)
        {
            if (!mask[j]) continue;

            double xMean = 0, yMean = 0;
            for (int i = 0; i < n; i++)
            {
                xMean += NumOps.ToDouble(data[i, j]);
                yMean += NumOps.ToDouble(target[i]);
            }
            xMean /= n;
            yMean /= n;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = NumOps.ToDouble(data[i, j]) - xMean;
                double dy = NumOps.ToDouble(target[i]) - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
                ssYY += dy * dy;
            }

            if (ssXX > 1e-10 && ssYY > 1e-10)
                totalCorr += Math.Abs(ssXY / Math.Sqrt(ssXX * ssYY));
        }

        double penalty = nSelected > _nFeaturesToSelect ? 0.1 * (nSelected - _nFeaturesToSelect) : 0;
        return totalCorr / nSelected - penalty;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GreyWolfFS has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GreyWolfFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GreyWolfFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
