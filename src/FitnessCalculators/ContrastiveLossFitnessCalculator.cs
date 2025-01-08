namespace AiDotNet.FitnessCalculators;

public class ContrastiveLossFitnessCalculator<T> : FitnessCalculatorBase<T>
{
    private readonly T _margin;

    public ContrastiveLossFitnessCalculator(T? margin = default, DataSetType dataSetType = DataSetType.Validation) 
        : base(false, dataSetType)
    {
        _margin = margin ?? _numOps.FromDouble(1.0);
    }

    protected override T GetFitnessScore(DataSetStats<T> dataSet)
    {
        var (output1, output2) = SplitOutputs(dataSet.Predicted);
        var (actual1, actual2) = SplitOutputs(dataSet.Actual);
        T totalLoss = _numOps.Zero;

        for (int i = 0; i < output1.Length; i++)
        {
            T similarityLabel = CalculateSimilarityLabel(actual1[i], actual2[i]);
            T pairLoss = NeuralNetworkHelper<T>.ContrastiveLoss(output1, output2, similarityLabel, _margin);
            totalLoss = _numOps.Add(totalLoss, pairLoss);
        }

        // Return average loss
        return _numOps.Divide(totalLoss, _numOps.FromDouble(output1.Length));
    }

    private static (Vector<T> Output1, Vector<T> Output2) SplitOutputs(Vector<T> predicted)
    {
        int halfLength = predicted.Length / 2;
        var output1 = new Vector<T>(halfLength);
        var output2 = new Vector<T>(halfLength);

        for (int i = 0; i < halfLength; i++)
        {
            output1[i] = predicted[i];
            output2[i] = predicted[i + halfLength];
        }

        return (output1, output2);
    }

    private T CalculateSimilarityLabel(T sample1, T sample2)
    {
        // Return 1 if the samples are the same, 0 otherwise
        return _numOps.Equals(sample1, sample2) ? _numOps.One : _numOps.Zero;
    }
}