﻿namespace AiDotNet.Models;

internal class PredictionStatsInputs<T>
{
    public Vector<T> Actual { get; set; } = Vector<T>.Empty();
    public Vector<T> Predicted { get; set; } = Vector<T>.Empty();
    public int NumberOfParameters { get; set; }
    public double ConfidenceLevel { get; set; }
    public int LearningCurveSteps { get; set; }
}