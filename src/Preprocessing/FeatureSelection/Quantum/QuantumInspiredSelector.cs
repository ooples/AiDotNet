using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Quantum;

/// <summary>
/// Quantum-Inspired Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses quantum-inspired optimization with quantum bits (qubits) represented
/// as probability amplitudes to search the feature subset space.
/// </para>
/// <para><b>For Beginners:</b> This method is inspired by quantum computing
/// principles. Each feature is represented by a "qubit" that can be in a
/// superposition of selected and not-selected states. The algorithm evolves
/// these quantum states to find the best feature subset, exploring many
/// possibilities simultaneously.
/// </para>
/// </remarks>
public class QuantumInspiredSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly double _rotationAngle;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public QuantumInspiredSelector(
        int nFeaturesToSelect = 10,
        int populationSize = 20,
        int nGenerations = 100,
        double rotationAngle = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nGenerations = nGenerations;
        _rotationAngle = rotationAngle;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "QuantumInspiredSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = RandomHelper.CreateSecureRandom();
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Initialize quantum population (qubits as angles)
        // Each qubit j has angle theta[j], where cos^2(theta) = probability of being 0
        var qubits = new double[_populationSize, p];
        for (int i = 0; i < _populationSize; i++)
            for (int j = 0; j < p; j++)
                qubits[i, j] = Math.PI / 4; // Initialize to 50-50 superposition

        // Best solution tracking
        var bestSolution = new bool[p];
        double bestFitness = double.MinValue;
        var selectionCounts = new double[p];

        for (int gen = 0; gen < _nGenerations; gen++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                // Observe (collapse) qubits to classical solution
                var solution = new bool[p];
                for (int j = 0; j < p; j++)
                {
                    double probOne = Math.Sin(qubits[i, j]) * Math.Sin(qubits[i, j]);
                    solution[j] = rand.NextDouble() < probOne;
                }

                // Ensure at least one feature selected
                if (!solution.Any(s => s))
                    solution[rand.Next(p)] = true;

                // Evaluate fitness
                double fitness = EvaluateFitness(X, y, solution, n, p);

                // Update best
                if (fitness > bestFitness)
                {
                    bestFitness = fitness;
                    Array.Copy(solution, bestSolution, p);
                }

                // Update selection counts
                for (int j = 0; j < p; j++)
                    if (solution[j])
                        selectionCounts[j]++;

                // Quantum rotation gate update
                for (int j = 0; j < p; j++)
                {
                    // Direction based on whether best solution has this feature
                    double direction = bestSolution[j] ? 1 : -1;

                    // Adjust based on current solution
                    if (solution[j] != bestSolution[j])
                    {
                        double delta = _rotationAngle * direction;

                        // Bound the rotation
                        if (bestSolution[j] && qubits[i, j] < Math.PI / 2)
                            qubits[i, j] += delta;
                        else if (!bestSolution[j] && qubits[i, j] > 0)
                            qubits[i, j] -= delta;
                    }
                }
            }
        }

        // Compute feature scores from selection counts and best solution
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            _featureScores[j] = selectionCounts[j] / (_populationSize * _nGenerations);
            if (bestSolution[j])
                _featureScores[j] += 1.0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double EvaluateFitness(double[,] X, double[] y, bool[] solution, int n, int p)
    {
        var features = Enumerable.Range(0, p).Where(j => solution[j]).ToList();
        if (features.Count == 0) return double.MinValue;

        // Compute average correlation with target
        double yMean = y.Average();
        double totalScore = 0;

        foreach (int j in features)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++) xMean += X[i, j];
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xd = X[i, j] - xMean;
                double yd = y[i] - yMean;
                sxy += xd * yd;
                sxx += xd * xd;
                syy += yd * yd;
            }
            double corr = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
            totalScore += corr;
        }

        // Penalty for too many features
        double sizePenalty = Math.Abs(features.Count - _nFeaturesToSelect) * 0.01;

        return totalScore / features.Count - sizePenalty;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantumInspiredSelector has not been fitted.");

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
        throw new NotSupportedException("QuantumInspiredSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantumInspiredSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
