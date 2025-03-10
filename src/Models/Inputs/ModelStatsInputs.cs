﻿namespace AiDotNet.Models.Inputs;

public class ModelStatsInputs<T>
{
    public Vector<T> Actual { get; set; } = Vector<T>.Empty();
    public Vector<T> Predicted { get; set; } = Vector<T>.Empty();
    public int FeatureCount { get; set; }
    public Matrix<T> XMatrix { get; set; } = Matrix<T>.Empty();
    public Func<Matrix<T>, Vector<T>, Vector<T>>? FitFunction { get; set; }
    public Vector<T> Coefficients { get; set; } = Vector<T>.Empty();
    public IPredictiveModel<T>? Model { get; set; }
    public List<string> FeatureNames { get; set; } = [];
    public Dictionary<string, Vector<T>> FeatureValues { get; set; } = [];
}