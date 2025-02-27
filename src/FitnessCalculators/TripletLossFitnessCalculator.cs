namespace AiDotNet.FitnessCalculators;

public class TripletLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    private readonly T _margin;

    public TripletLossFitnessCalculator(T? margin = default, DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
        _margin = margin ?? _numOps.FromDouble(1.0);
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        var (anchor, positive, negative) = PrepareTripletData(dataSet.Features, dataSet.Actual);
        return NeuralNetworkHelper<T>.TripletLoss(anchor, positive, negative, _margin);
    }

    private (Matrix<T> Anchor, Matrix<T> Positive, Matrix<T> Negative) PrepareTripletData(Matrix<T> X, Vector<T> y)
    {
        var classes = y.Distinct().ToList();
        var anchorList = new List<Vector<T>>();
        var positiveList = new List<Vector<T>>();
        var negativeList = new List<Vector<T>>();

        for (int i = 0; i < X.Rows; i++)
        {
            var anchor = X.GetRow(i);
            var anchorClass = y[i];

            // Find a positive example (same class as anchor)
            var positiveIndices = Enumerable.Range(0, y.Length)
                .Where(j => j != i && _numOps.Equals(y[j], anchorClass))
                .ToList();

            if (positiveIndices.Count == 0) continue; // Skip if no positive example found

            var positiveIndex = positiveIndices[new Random().Next(positiveIndices.Count)];
            var positive = X.GetRow(positiveIndex);

            // Find a negative example (different class from anchor)
            var negativeClass = classes.Where(c => !_numOps.Equals(c, anchorClass)).RandomElement();
            var negativeIndices = Enumerable.Range(0, y.Length)
                .Where(j => _numOps.Equals(y[j], negativeClass))
                .ToList();

            if (negativeIndices.Count == 0) continue; // Skip if no negative example found

            var negativeIndex = negativeIndices[new Random().Next(negativeIndices.Count)];
            var negative = X.GetRow(negativeIndex);

            anchorList.Add(anchor);
            positiveList.Add(positive);
            negativeList.Add(negative);
        }

        return (
            new Matrix<T>([.. anchorList]),
            new Matrix<T>([.. positiveList]),
            new Matrix<T>([.. negativeList])
        );
    }
}