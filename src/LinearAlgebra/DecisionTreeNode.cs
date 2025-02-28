﻿namespace AiDotNet.LinearAlgebra;

public class DecisionTreeNode<T>
{
    public int FeatureIndex { get; set; }
    public T Threshold { get; set; }
    public T SplitValue { get; set; }
    public T Prediction { get; set; }
    public DecisionTreeNode<T>? Left { get; set; }
    public DecisionTreeNode<T>? Right { get; set; }
    public bool IsLeaf { get; set; }
    public List<Sample<T>> Samples { get; set; } = [];
    public int LeftSampleCount { get; set; }
    public int RightSampleCount { get; set; }
    public List<T> SampleValues { get; set; } = [];
    public SimpleRegression<T>? LinearModel { get; set; }
    public Vector<T>? Predictions { get; set; }
    public T SumSquaredError { get; set; }

    private INumericOperations<T> NumOps { get; set; }

    public DecisionTreeNode()
    {
        Left = null;
        Right = null;
        IsLeaf = true;

        NumOps = MathHelper.GetNumericOperations<T>();
        SplitValue = NumOps.Zero;
        Prediction = NumOps.Zero;
        Threshold = NumOps.Zero;
        SumSquaredError = NumOps.Zero;
    }

    public DecisionTreeNode(int featureIndex, T splitValue)
    {
        FeatureIndex = featureIndex;
        SplitValue = splitValue;
        IsLeaf = false;

        NumOps = MathHelper.GetNumericOperations<T>();
        SplitValue = NumOps.Zero;
        Prediction = NumOps.Zero;
        Threshold = NumOps.Zero;
        SumSquaredError = NumOps.Zero;
    }

    public DecisionTreeNode(T prediction)
    {
        Prediction = prediction;
        IsLeaf = true;

        NumOps = MathHelper.GetNumericOperations<T>();
        SplitValue = NumOps.Zero;
        Prediction = NumOps.Zero;
        Threshold = NumOps.Zero;
        SumSquaredError = NumOps.Zero;
    }

    public void UpdateNodeStatistics()
    {
        SumSquaredError = CalculateSumSquaredError();
    }

    private T CalculateSumSquaredError()
    {
        if (Samples == null || Samples.Count == 0)
        {
            return NumOps.Zero;
        }

        return Samples.Aggregate(NumOps.Zero, (sum, sample) =>
        {
            var error = NumOps.Subtract(sample.Target, Prediction);
            return NumOps.Add(sum, NumOps.Multiply(error, error));
        });
    }
}