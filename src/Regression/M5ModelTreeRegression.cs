namespace AiDotNet.Regression;

public class M5ModelTree<T> : AsyncDecisionTreeRegressionBase<T>
{
    private readonly M5ModelTreeOptions _options;

    public M5ModelTree(M5ModelTreeOptions? options = null, IRegularization<T>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new M5ModelTreeOptions();
    }

    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        Root = await BuildTreeAsync(x, y, 0);
        if (_options.UsePruning)
        {
            await PruneTreeAsync(Root);
        }
        await CalculateFeatureImportancesAsync(x.Columns);
    }

    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);
        var tasks = Enumerable.Range(0, input.Rows).Select(i => Task.Run(() => PredictSingle(input.GetRow(i))));
        var results = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);
        for (int i = 0; i < results.Count; i++)
        {
            predictions[i] = results[i];
        }

        return predictions;
    }

    private async Task<DecisionTreeNode<T>> BuildTreeAsync(Matrix<T> x, Vector<T> y, int depth)
    {
        if (x.Rows <= _options.MinInstancesPerLeaf || depth >= _options.MaxDepth)
        {
            return CreateLeafNode(x, y);
        }

        var bestSplit = await FindBestSplitAsync(x, y);
        if (bestSplit == null)
        {
            return CreateLeafNode(x, y);
        }

        var (leftX, leftY, rightX, rightY) = SplitData(x, y, bestSplit.Value.Feature, bestSplit.Value.Threshold);

        var leftChildTask = BuildTreeAsync(leftX, leftY, depth + 1);
        var rightChildTask = BuildTreeAsync(rightX, rightY, depth + 1);

        await Task.WhenAll(leftChildTask, rightChildTask);

        return new DecisionTreeNode<T>(bestSplit.Value.Feature, bestSplit.Value.Threshold)
        {
            Left = await leftChildTask,
            Right = await rightChildTask
        };
    }

    private async Task<(int Feature, T Threshold)?> FindBestSplitAsync(Matrix<T> x, Vector<T> y)
    {
        var tasks = Enumerable.Range(0, x.Columns).Select(feature => 
            Task.Run(() => FindBestSplitForFeature(x, y, feature)));

        var results = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

        var bestSplit = results.OrderByDescending(r => r.SDR).FirstOrDefault();
        return bestSplit.Feature == -1 ? null : (bestSplit.Feature, bestSplit.Threshold);
    }

    private (int Feature, T Threshold, T SDR) FindBestSplitForFeature(Matrix<T> x, Vector<T> y, int feature)
    {
        var featureValues = x.GetColumn(feature);
        var sortedIndices = featureValues.Select((value, index) => (value, index))
                                            .OrderBy(pair => pair.value)
                                            .Select(pair => pair.index)
                                            .ToArray();
        var bestSplit = (Threshold: NumOps.Zero, SDR: NumOps.Zero);

        for (int i = 1; i < sortedIndices.Length; i++)
        {
            var threshold = NumOps.Divide(NumOps.Add(featureValues[sortedIndices[i - 1]], featureValues[sortedIndices[i]]), NumOps.FromDouble(2));
            var sdr = CalculateSDR(x, y, feature, threshold);

            if (NumOps.GreaterThan(sdr, bestSplit.SDR))
            {
                bestSplit = (threshold, sdr);
            }
        }

        return (feature, bestSplit.Threshold, bestSplit.SDR);
    }

    private T CalculateSDR(Matrix<T> x, Vector<T> y, int feature, T threshold)
    {
        var (leftX, leftY, rightX, rightY) = SplitData(x, y, feature, threshold);
        var totalVariance = StatisticsHelper<T>.CalculateVariance(y);
        var leftVariance = StatisticsHelper<T>.CalculateVariance(leftY);
        var rightVariance = StatisticsHelper<T>.CalculateVariance(rightY);

        var leftWeight = NumOps.Divide(NumOps.FromDouble(leftY.Length), NumOps.FromDouble(y.Length));
        var rightWeight = NumOps.Divide(NumOps.FromDouble(rightY.Length), NumOps.FromDouble(y.Length));

        var weightedVariance = NumOps.Add(
            NumOps.Multiply(leftWeight, leftVariance),
            NumOps.Multiply(rightWeight, rightVariance)
        );

        return NumOps.Subtract(totalVariance, weightedVariance);
    }

    private DecisionTreeNode<T> CreateLeafNode(Matrix<T> x, Vector<T> y)
    {
        if (_options.UseLinearRegressionAtLeaves)
        {
            var model = FitLinearModel(x, y);
            return new DecisionTreeNode<T>(model.Intercept) { LinearModel = model };
        }
        else
        {
            var mean = StatisticsHelper<T>.CalculateMean(y);
            return new DecisionTreeNode<T>(mean);
        }
    }

    private SimpleRegression<T> FitLinearModel(Matrix<T> x, Vector<T> y)
    {
        var regression = new SimpleRegression<T>(regularization: Regularization);
        regression.Train(x, y);

        return regression;
    }

    private async Task PruneTreeAsync(DecisionTreeNode<T>? node)
    {
        if (node == null || node.IsLeaf)
        {
            return;
        }

        await Task.WhenAll(
            PruneTreeAsync(node.Left),
            PruneTreeAsync(node.Right)
        );

        var subtreeError = CalculateSubtreeError(node);
        var leafError = CalculateLeafError(node);

        // Apply pruning factor
        var adjustedLeafError = NumOps.Multiply(leafError, NumOps.FromDouble(1 + _options.PruningFactor));

        if (NumOps.LessThanOrEquals(adjustedLeafError, subtreeError))
        {
            // Convert to leaf node
            node.Left = null;
            node.Right = null;
            node.IsLeaf = true;
            node.Prediction = CalculateAveragePrediction(node);
            node.Predictions = Vector<T>.CreateDefault(node.Samples.Count, node.Prediction);
            node.SumSquaredError = leafError;
        }
    }

    private T PredictSingle(Vector<T> input)
    {
        var node = Root;
        while (node != null && !node.IsLeaf)
        {
            if (NumOps.LessThanOrEquals(input[node.FeatureIndex], node.Threshold))
            {
                node = node?.Left;
            }
            else
            {
                node = node?.Right;
            }
        }

        if (node != null)
        {
            if (_options.UseLinearRegressionAtLeaves && node.LinearModel != null)
            {
                // Convert the input vector to a single-column matrix using the new method
                var inputMatrix = Matrix<T>.FromVector(input);
                return node.LinearModel.Predict(inputMatrix)[0];
            }
            else
            {
                return node.Prediction;
            }
        }

        throw new InvalidOperationException("Invalid tree structure");
    }

    protected override async Task CalculateFeatureImportancesAsync(int featureCount)
    {
        FeatureImportances = new Vector<T>(featureCount);
        await CalculateFeatureImportancesRecursiveAsync(Root, NumOps.One);
        FeatureImportances = FeatureImportances.Divide(FeatureImportances.Sum());
    }

    private async Task CalculateFeatureImportancesRecursiveAsync(DecisionTreeNode<T>? node, T weight)
    {
        if (node == null || node.IsLeaf)
        {
            return;
        }

        FeatureImportances[node.FeatureIndex] = NumOps.Add(FeatureImportances[node.FeatureIndex], weight);

        var leftWeight = NumOps.Multiply(weight, NumOps.FromDouble(0.5));
        var rightWeight = NumOps.Multiply(weight, NumOps.FromDouble(0.5));

        await Task.WhenAll(
            CalculateFeatureImportancesRecursiveAsync(node.Left, leftWeight),
            CalculateFeatureImportancesRecursiveAsync(node?.Right, rightWeight)
        );
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.M5ModelTree,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MaxDepth", _options.MaxDepth },
                { "MinInstancesPerLeaf", _options.MinInstancesPerLeaf },
                { "PruningFactor", _options.PruningFactor },
                { "UseLinearRegressionAtLeaves", _options.UseLinearRegressionAtLeaves },
                { "UsePruning", _options.UsePruning },
                { "SmoothingConstant", _options.SmoothingConstant },
                { "FeatureImportances", FeatureImportances },
                { "Regularization", Regularization.GetType().Name }
            }
        };
    }

    private T CalculateSubtreeError(DecisionTreeNode<T>? node)
    {
        if (node == null)
        {
            return NumOps.Zero;
        }

        if (node.IsLeaf)
        {
            return node.SumSquaredError;
        }

        return NumOps.Add(
            CalculateSubtreeError(node.Left),
            CalculateSubtreeError(node.Right)
        );
    }

    private T CalculateLeafError(DecisionTreeNode<T>? node)
    {
        if (node == null || node.Samples == null)
        {
            return NumOps.Zero;
        }

        var meanPrediction = CalculateAveragePrediction(node);
        var error = node.Samples.Select(sample => 
            NumOps.Multiply(
                NumOps.Subtract(sample.Target, meanPrediction),
                NumOps.Subtract(sample.Target, meanPrediction)
            )
        ).Aggregate(NumOps.Zero, NumOps.Add);

        // Apply regularization if a linear model exists
        if (node.LinearModel != null && node.LinearModel.Coefficients != null)
        {
            var regularizedCoefficients = Regularization.RegularizeCoefficients(node.LinearModel.Coefficients);
            var regularizationTerm = regularizedCoefficients.Subtract(node.LinearModel.Coefficients).L2Norm();
            error = NumOps.Add(error, regularizationTerm);
        }

        return error;
    }

    private T CalculateAveragePrediction(DecisionTreeNode<T> node)
    {
        if (node.Samples == null || node.Samples.Count == 0)
        {
            return NumOps.Zero;
        }

        var sum = node.Samples.Aggregate(NumOps.Zero, (acc, sample) => NumOps.Add(acc, sample.Target));
        return NumOps.Divide(sum, NumOps.FromDouble(node.Samples.Count));
    }

    private int CalculateTreeDepth(DecisionTreeNode<T>? node)
    {
        if (node == null || node.IsLeaf)
        {
            return 0;
        }
        return 1 + Math.Max(CalculateTreeDepth(node.Left), CalculateTreeDepth(node?.Right));
    }

    private int CountNodes(DecisionTreeNode<T>? node)
    {
        if (node == null)
        {
            return 0;
        }

        return 1 + CountNodes(node?.Left) + CountNodes(node?.Right);
    }

    private (Matrix<T> LeftX, Vector<T> LeftY, Matrix<T> RightX, Vector<T> RightY) SplitData(Matrix<T> x, Vector<T> y, int feature, T threshold)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        for (int i = 0; i < x.Rows; i++)
        {
            if (NumOps.LessThanOrEquals(x[i, feature], threshold))
            {
                leftIndices.Add(i);
            }
            else
            {
                rightIndices.Add(i);
            }
        }

        return (
            x.GetRows(leftIndices),
            y.GetElements(leftIndices),
            x.GetRows(rightIndices),
            y.GetElements(rightIndices)
        );
    }
}