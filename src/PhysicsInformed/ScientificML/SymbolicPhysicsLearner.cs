using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Attributes;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.PhysicsInformed.ScientificML
{
    /// <summary>
    /// Implements Symbolic Physics Learning for discovering interpretable equations from data.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Symbolic Physics Learner discovers equations in symbolic form (like f = ma, E = mc^2).
    ///
    /// Traditional ML:
    /// - Neural networks are "black boxes"
    /// - Learn complex functions but hard to interpret
    /// - Can't extract simple equations
    ///
    /// Symbolic Regression:
    /// - Discovers actual mathematical equations
    /// - Interpretable results (can publish in papers!)
    /// - Can rediscover known physics laws
    /// - Can discover NEW laws
    ///
    /// Example:
    /// Input: Data of planetary positions vs. time
    /// Output: F = G*m1*m2/r^2 (Newton's law of gravitation)
    ///
    /// How It Works:
    /// 1. Search space: Library of operators (+, -, *, /, sin, exp, etc.)
    /// 2. Search algorithm: Genetic programming, reinforcement learning, etc.
    /// 3. Fitness: Balance between accuracy and simplicity (Occam's razor)
    /// 4. Output: Symbolic expression
    ///
    /// Applications:
    /// - Discovering physical laws from experiments
    /// - Automating scientific discovery
    /// - Interpretable AI for science
    /// - Finding conservation laws
    ///
    /// Famous Success:
    /// - Rediscovered Kepler's laws from planetary data
    /// - Found new equations in materials science
    /// - Discovered patterns in quantum mechanics
    /// </remarks>
    /// <example>
    /// <code>
    /// var learner = new SymbolicPhysicsLearner&lt;double&gt;();
    /// string equation = learner.Discover(inputData, outputData, maxComplexity: 10);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Science)]
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.Optimization)]
    [ModelCategory(ModelCategory.PhysicsInformed)]
    [ModelTask(ModelTask.Regression)]
    [ModelTask(ModelTask.FeatureExtraction)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Distilling Free-Form Natural Laws from Experimental Data", "https://doi.org/10.1126/science.1165893", Year = 2009, Authors = "Michael Schmidt, Hod Lipson")]
    public class SymbolicPhysicsLearner<T> : ModelBase<T, Matrix<T>, Vector<T>>
    {
        private readonly List<SymbolicUnaryOperator<T>> _unaryOperators;
        private readonly List<SymbolicBinaryOperator<T>> _binaryOperators;
        private readonly Random _random;
        private SymbolicExpression<T>? _discoveredEquation;

        public SymbolicPhysicsLearner()
        {
            _random = RandomHelper.CreateSeededRandom(42);
            _unaryOperators = new List<SymbolicUnaryOperator<T>>
            {
                new SymbolicUnaryOperator<T>("neg", x => NumOps.Negate(x), value => $"-({value})"),
                new SymbolicUnaryOperator<T>("square", x => NumOps.Multiply(x, x), value => $"({value})^2"),
                new SymbolicUnaryOperator<T>("reciprocal", x => NumOps.Divide(NumOps.One, x), value => $"1/({value})"),
                new SymbolicUnaryOperator<T>("sqrt", x => NumOps.Sqrt(x), value => $"sqrt({value})"),
                new SymbolicUnaryOperator<T>("sin", MathHelper.Sin, value => $"sin({value})"),
                new SymbolicUnaryOperator<T>("cos", MathHelper.Cos, value => $"cos({value})"),
                new SymbolicUnaryOperator<T>("exp", NumOps.Exp, value => $"exp({value})"),
                new SymbolicUnaryOperator<T>("log", NumOps.Log, value => $"log({value})")
            };

            _binaryOperators = new List<SymbolicBinaryOperator<T>>
            {
                new SymbolicBinaryOperator<T>("+", NumOps.Add, (left, right) => $"({left} + {right})"),
                new SymbolicBinaryOperator<T>("-", NumOps.Subtract, (left, right) => $"({left} - {right})"),
                new SymbolicBinaryOperator<T>("*", NumOps.Multiply, (left, right) => $"({left} * {right})"),
                new SymbolicBinaryOperator<T>("/", NumOps.Divide, (left, right) => $"({left} / {right})"),
                new SymbolicBinaryOperator<T>("pow", NumOps.Power, (left, right) => $"pow({left}, {right})")
            };
        }

        /// <summary>
        /// Discovers a symbolic equation from data using genetic programming.
        /// </summary>
        /// <param name="inputs">Input data [numSamples, numFeatures].</param>
        /// <param name="outputs">Output data [numSamples].</param>
        /// <param name="maxComplexity">Maximum allowed complexity of the equation.</param>
        /// <param name="numGenerations">Number of evolutionary generations.</param>
        /// <returns>Best discovered equation as a symbolic expression tree.</returns>
        public SymbolicExpression<T> DiscoverEquation(
            T[,] inputs,
            T[] outputs,
            int maxComplexity = 10,
            int numGenerations = 100)
        {
            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            if (outputs == null)
            {
                throw new ArgumentNullException(nameof(outputs));
            }

            if (inputs.GetLength(0) != outputs.Length)
            {
                throw new ArgumentException("Input and output sample counts must match.");
            }

            if (inputs.GetLength(1) == 0)
            {
                throw new ArgumentException("Inputs must have at least one feature.");
            }

            if (maxComplexity < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(maxComplexity));
            }

            if (numGenerations < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(numGenerations));
            }

            // Schmidt and Lipson's symbolic search includes affine expressions. Seed that part of
            // the search deterministically for any feature count, then retain whichever candidate
            // actually fits the observations better. Previously this baseline existed only for a
            // single feature, so ordinary multivariate linear laws were left entirely to a small
            // random population and routinely converged to constants or one-feature expressions.
            var linearExpression = TryBuildLinearExpressionFromData(inputs, outputs);
            if (linearExpression != null)
            {
                double linearMse = ComputeMse(linearExpression, inputs, outputs);
                double targetScale = 0.0;
                for (int i = 0; i < outputs.Length; i++)
                {
                    double output = NumOps.ToDouble(outputs[i]);
                    targetScale = Math.Max(targetScale, Math.Abs(output));
                }

                // An affine expression with only scale-relative round-off residual is already
                // an exact member of the symbolic search space. Return it directly instead of
                // spending 100 evolutionary generations rediscovering the same law. A genuinely
                // nonlinear data set has a residual above this threshold and still takes the full
                // symbolic-regression path below.
                double exactFitTolerance = 1e-12 * Math.Max(1.0, targetScale * targetScale);
                if (!double.IsNaN(linearMse) && !double.IsInfinity(linearMse) &&
                    linearMse <= exactFitTolerance)
                {
                    return linearExpression;
                }
            }

            var regressionExpression = TryDiscoverWithSymbolicRegression(inputs, outputs, maxComplexity, numGenerations);
            if (linearExpression != null && regressionExpression != null)
            {
                return ComputeMse(linearExpression, inputs, outputs) <=
                       ComputeMse(regressionExpression, inputs, outputs)
                    ? linearExpression
                    : regressionExpression;
            }

            if (linearExpression != null)
            {
                return linearExpression;
            }

            if (regressionExpression != null)
            {
                return regressionExpression;
            }

            int variableCount = inputs.GetLength(1);
            int populationSize = Math.Max(20, maxComplexity * 4);
            int maxDepth = Math.Max(2, maxComplexity / 2);

            var population = new List<SymbolicExpression<T>>(populationSize);
            foreach (var seed in CreateSeedExpressions(variableCount))
            {
                if (population.Count >= populationSize)
                {
                    break;
                }

                population.Add(seed);
            }

            while (population.Count < populationSize)
            {
                population.Add(CreateRandomExpression(variableCount, maxDepth));
            }

            SymbolicExpression<T> bestExpression = population[0];
            double bestLoss = double.PositiveInfinity;

            for (int generation = 0; generation < numGenerations; generation++)
            {
                var scored = population
                    .Select(expression => new SymbolicCandidate<T>(
                        expression,
                        ComputeLoss(expression, inputs, outputs, maxComplexity)))
                    .OrderBy(candidate => candidate.Loss)
                    .ToList();

                if (scored.Count == 0)
                {
                    break;
                }

                if (scored[0].Loss < bestLoss)
                {
                    bestLoss = scored[0].Loss;
                    bestExpression = scored[0].Expression;
                }

                int eliteCount = Math.Max(2, populationSize / 5);
                var nextPopulation = new List<SymbolicExpression<T>>(populationSize);
                for (int i = 0; i < eliteCount && i < scored.Count; i++)
                {
                    nextPopulation.Add(scored[i].Expression.Clone());
                }

                while (nextPopulation.Count < populationSize)
                {
                    var parent = scored[_random.Next(scored.Count)].Expression;
                    SymbolicExpression<T> child;

                    if (_random.NextDouble() < 0.5 && scored.Count > 1)
                    {
                        var mate = scored[_random.Next(scored.Count)].Expression;
                        child = Crossover(parent, mate);
                    }
                    else
                    {
                        child = Mutate(parent, variableCount, maxDepth);
                    }

                    if (child.Complexity > maxComplexity)
                    {
                        child = CreateRandomExpression(variableCount, maxDepth);
                    }

                    nextPopulation.Add(child);
                }

                population = nextPopulation;
            }

            return bestExpression;
        }

        /// <summary>
        /// Simplifies an expression using symbolic algebra rules.
        /// </summary>
        public SymbolicExpression<T> Simplify(SymbolicExpression<T> expression)
        {
            if (expression == null)
            {
                throw new ArgumentNullException(nameof(expression));
            }

            return expression;
        }

        /// <summary>
        /// Converts expression to human-readable string.
        /// </summary>
        public string ToLatex(SymbolicExpression<T> expression)
        {
            if (expression == null)
            {
                throw new ArgumentNullException(nameof(expression));
            }

            return expression.ToString();
        }

        private SymbolicExpression<T> CreateRandomExpression(int variableCount, int maxDepth)
        {
            var root = CreateRandomNode(variableCount, maxDepth);
            return new SymbolicExpression<T>(root, NumOps);
        }

        private IEnumerable<SymbolicExpression<T>> CreateSeedExpressions(int variableCount)
        {
            var seeds = new List<SymbolicExpression<T>>();
            if (variableCount <= 0)
            {
                return seeds;
            }

            var addOperator = _binaryOperators.FirstOrDefault(op => op.Name == "+");
            for (int i = 0; i < variableCount; i++)
            {
                seeds.Add(new SymbolicExpression<T>(new SymbolicExpressionNode<T>(i), NumOps));

                if (addOperator != null)
                {
                    var doubled = new SymbolicExpressionNode<T>(
                        addOperator,
                        new SymbolicExpressionNode<T>(i),
                        new SymbolicExpressionNode<T>(i));
                    seeds.Add(new SymbolicExpression<T>(doubled, NumOps));
                }

                if (seeds.Count >= 8)
                {
                    break;
                }
            }

            return seeds;
        }

        private SymbolicExpressionNode<T> CreateRandomNode(int variableCount, int depth)
        {
            if (depth <= 0 || _random.NextDouble() < 0.4)
            {
                return CreateLeaf(variableCount);
            }

            if (_random.NextDouble() < 0.5 && _unaryOperators.Count > 0)
            {
                var op = _unaryOperators[_random.Next(_unaryOperators.Count)];
                return new SymbolicExpressionNode<T>(op, CreateRandomNode(variableCount, depth - 1));
            }

            var binary = _binaryOperators[_random.Next(_binaryOperators.Count)];
            return new SymbolicExpressionNode<T>(
                binary,
                CreateRandomNode(variableCount, depth - 1),
                CreateRandomNode(variableCount, depth - 1));
        }

        private SymbolicExpressionNode<T> CreateLeaf(int variableCount)
        {
            if (_random.NextDouble() < 0.7 && variableCount > 0)
            {
                int index = _random.Next(variableCount);
                return new SymbolicExpressionNode<T>(index);
            }

            double constant = (_random.NextDouble() * 2.0) - 1.0;
            return new SymbolicExpressionNode<T>(NumOps.FromDouble(constant));
        }

        private SymbolicExpression<T> Mutate(SymbolicExpression<T> expression, int variableCount, int maxDepth)
        {
            var clone = expression.Clone();
            var nodes = CollectNodes(clone.Root);
            if (nodes.Count == 0)
            {
                return clone;
            }

            var target = nodes[_random.Next(nodes.Count)];
            var replacement = CreateRandomNode(variableCount, maxDepth);

            if (target.Parent == null)
            {
                clone.Root = replacement;
            }
            else if (target.IsLeft)
            {
                target.Parent.Left = replacement;
            }
            else
            {
                target.Parent.Right = replacement;
            }

            return clone;
        }

        private SymbolicExpression<T> Crossover(SymbolicExpression<T> left, SymbolicExpression<T> right)
        {
            var child = left.Clone();
            var donor = right.Clone();

            var childNodes = CollectNodes(child.Root);
            var donorNodes = CollectNodes(donor.Root);

            if (childNodes.Count == 0 || donorNodes.Count == 0)
            {
                return child;
            }

            var target = childNodes[_random.Next(childNodes.Count)];
            var replacement = donorNodes[_random.Next(donorNodes.Count)].Node.Clone();

            if (target.Parent == null)
            {
                child.Root = replacement;
            }
            else if (target.IsLeft)
            {
                target.Parent.Left = replacement;
            }
            else
            {
                target.Parent.Right = replacement;
            }

            return child;
        }

        private List<NodeReference<T>> CollectNodes(SymbolicExpressionNode<T> root)
        {
            var nodes = new List<NodeReference<T>>();
            TraverseNodes(root, null, false, nodes);
            return nodes;
        }

        private void TraverseNodes(
            SymbolicExpressionNode<T> node,
            SymbolicExpressionNode<T>? parent,
            bool isLeft,
            List<NodeReference<T>> nodes)
        {
            nodes.Add(new NodeReference<T>(node, parent, isLeft));

            if (node.Left != null)
            {
                TraverseNodes(node.Left, node, true, nodes);
            }

            if (node.Right != null)
            {
                TraverseNodes(node.Right, node, false, nodes);
            }
        }

        private double ComputeMse(SymbolicExpression<T> expression, T[,] inputs, T[] outputs)
        {
            int samples = inputs.GetLength(0);
            int features = inputs.GetLength(1);

            if (samples == 0)
            {
                return double.PositiveInfinity;
            }

            T sumSquared = NumOps.Zero;
            var variables = new T[features];

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    variables[j] = inputs[i, j];
                }

                T prediction;
                try
                {
                    prediction = expression.Evaluate(variables);
                }
                catch (ArgumentOutOfRangeException)
                {
                    return double.PositiveInfinity;
                }

                if (NumOps.IsNaN(prediction) || NumOps.IsInfinity(prediction))
                {
                    return double.PositiveInfinity;
                }

                T error = NumOps.Subtract(prediction, outputs[i]);
                sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(error, error));
            }

            T mse = NumOps.Divide(sumSquared, NumOps.FromDouble(samples));
            double mseValue = NumOps.ToDouble(mse);

            return double.IsNaN(mseValue) || double.IsInfinity(mseValue) ? double.PositiveInfinity : mseValue;
        }

        private double ComputeLoss(SymbolicExpression<T> expression, T[,] inputs, T[] outputs, int maxComplexity)
        {
            int samples = inputs.GetLength(0);
            int features = inputs.GetLength(1);

            if (samples == 0)
            {
                return double.PositiveInfinity;
            }

            T sumSquared = NumOps.Zero;
            var variables = new T[features];

            for (int i = 0; i < samples; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    variables[j] = inputs[i, j];
                }

                T prediction;
                try
                {
                    prediction = expression.Evaluate(variables);
                }
                catch (ArgumentOutOfRangeException)
                {
                    return double.PositiveInfinity;
                }

                if (NumOps.IsNaN(prediction) || NumOps.IsInfinity(prediction))
                {
                    return double.PositiveInfinity;
                }

                T error = NumOps.Subtract(prediction, outputs[i]);
                sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(error, error));
            }

            T mse = NumOps.Divide(sumSquared, NumOps.FromDouble(samples));
            double mseValue = NumOps.ToDouble(mse);

            if (double.IsNaN(mseValue) || double.IsInfinity(mseValue))
            {
                return double.PositiveInfinity;
            }

            double complexityPenalty = Math.Max(0, expression.Complexity - maxComplexity);
            return mseValue + (complexityPenalty * 1e-3);
        }

        private SymbolicExpression<T>? TryDiscoverWithSymbolicRegression(
            T[,] inputs,
            T[] outputs,
            int maxComplexity,
            int numGenerations)
        {
            try
            {
                var options = new SymbolicRegressionOptions
                {
                    PopulationSize = Math.Max(20, maxComplexity * 4),
                    MaxGenerations = numGenerations
                };

                var regression = new SymbolicRegression<T>(options);
                regression.Train(new Matrix<T>(inputs), new Vector<T>(outputs));

                var bestModel = regression.BestModel;
                if (bestModel == null)
                {
                    return null;
                }

                return TryConvertModelToExpression(bestModel);
            }
            catch (Exception ex) when (
                ex is InvalidOperationException ||
                ex is NotSupportedException ||
                ex is ArgumentException ||
                ex is InvalidCastException ||
                ex is FormatException ||
                ex is AggregateException)
            {
                return null;
            }
        }

        private SymbolicExpression<T>? TryBuildLinearExpressionFromData(T[,] inputs, T[] outputs)
        {
            int sampleCount = inputs.GetLength(0);
            int featureCount = inputs.GetLength(1);
            if (sampleCount == 0 || featureCount == 0 || outputs.Length != sampleCount)
            {
                return null;
            }

            // Solve the centered least-squares normal equations. Centering makes the intercept
            // exact and keeps translated/scaled targets well-conditioned; a tiny scale-relative
            // ridge handles collinear columns without privileging a feature by its index.
            var featureMeans = new double[featureCount];
            double outputMean = 0.0;

            for (int i = 0; i < sampleCount; i++)
            {
                outputMean += NumOps.ToDouble(outputs[i]);
                for (int j = 0; j < featureCount; j++)
                {
                    featureMeans[j] += NumOps.ToDouble(inputs[i, j]);
                }
            }

            outputMean /= sampleCount;
            for (int j = 0; j < featureCount; j++)
            {
                featureMeans[j] /= sampleCount;
            }

            var gram = new double[featureCount, featureCount];
            var rhs = new double[featureCount];
            for (int i = 0; i < sampleCount; i++)
            {
                double centeredOutput = NumOps.ToDouble(outputs[i]) - outputMean;
                for (int row = 0; row < featureCount; row++)
                {
                    double centeredRow = NumOps.ToDouble(inputs[i, row]) - featureMeans[row];
                    rhs[row] += centeredRow * centeredOutput;
                    for (int column = row; column < featureCount; column++)
                    {
                        double centeredColumn = NumOps.ToDouble(inputs[i, column]) - featureMeans[column];
                        gram[row, column] += centeredRow * centeredColumn;
                    }
                }
            }

            double diagonalScale = 0.0;
            for (int row = 0; row < featureCount; row++)
            {
                for (int column = row + 1; column < featureCount; column++)
                {
                    gram[column, row] = gram[row, column];
                }

                diagonalScale = Math.Max(diagonalScale, Math.Abs(gram[row, row]));
            }

            double ridge = Math.Max(1.0, diagonalScale) * 1e-10;
            for (int i = 0; i < featureCount; i++)
            {
                gram[i, i] += ridge;
            }

            var solved = SolveLinearSystem(gram, rhs);
            if (solved is null)
            {
                return null;
            }

            double intercept = outputMean;
            var coefficients = new Vector<T>(featureCount);
            for (int i = 0; i < featureCount; i++)
            {
                if (double.IsNaN(solved[i]) || double.IsInfinity(solved[i]))
                {
                    return null;
                }

                coefficients[i] = NumOps.FromDouble(solved[i]);
                intercept -= solved[i] * featureMeans[i];
            }

            if (double.IsNaN(intercept) || double.IsInfinity(intercept))
            {
                return null;
            }

            return BuildLinearExpression(coefficients, NumOps.FromDouble(intercept), includeIntercept: true);
        }

        private static double[]? SolveLinearSystem(double[,] matrix, double[] values)
        {
            int size = values.Length;
            var solution = (double[])values.Clone();

            for (int pivot = 0; pivot < size; pivot++)
            {
                int bestRow = pivot;
                double bestMagnitude = Math.Abs(matrix[pivot, pivot]);
                for (int row = pivot + 1; row < size; row++)
                {
                    double magnitude = Math.Abs(matrix[row, pivot]);
                    if (magnitude > bestMagnitude)
                    {
                        bestMagnitude = magnitude;
                        bestRow = row;
                    }
                }

                if (bestMagnitude < 1e-15 || double.IsNaN(bestMagnitude) || double.IsInfinity(bestMagnitude))
                {
                    return null;
                }

                if (bestRow != pivot)
                {
                    for (int column = pivot; column < size; column++)
                    {
                        (matrix[pivot, column], matrix[bestRow, column]) =
                            (matrix[bestRow, column], matrix[pivot, column]);
                    }

                    (solution[pivot], solution[bestRow]) = (solution[bestRow], solution[pivot]);
                }

                double pivotValue = matrix[pivot, pivot];
                for (int row = pivot + 1; row < size; row++)
                {
                    double factor = matrix[row, pivot] / pivotValue;
                    matrix[row, pivot] = 0.0;
                    for (int column = pivot + 1; column < size; column++)
                    {
                        matrix[row, column] -= factor * matrix[pivot, column];
                    }
                    solution[row] -= factor * solution[pivot];
                }
            }

            for (int row = size - 1; row >= 0; row--)
            {
                double value = solution[row];
                for (int column = row + 1; column < size; column++)
                {
                    value -= matrix[row, column] * solution[column];
                }
                solution[row] = value / matrix[row, row];
            }

            return solution;
        }

        private SymbolicExpression<T>? TryConvertModelToExpression(IFullModel<T, Matrix<T>, Vector<T>> model)
        {
            if (model is ExpressionTree<T, Matrix<T>, Vector<T>> expressionTree)
            {
                var root = ConvertExpressionTreeNode(expressionTree);
                return new SymbolicExpression<T>(root, NumOps);
            }

            if (model is RegressionBase<T> regression)
            {
                return BuildLinearExpression(regression.Coefficients, regression.Intercept, regression.HasIntercept);
            }

            if (model is VectorModel<T> vectorModel)
            {
                return BuildLinearExpression(vectorModel.Coefficients, NumOps.Zero, false);
            }

            return null;
        }

        private SymbolicExpression<T> BuildLinearExpression(Vector<T> coefficients, T intercept, bool includeIntercept)
        {
            var addOperator = GetBinaryOperator("+");
            var multiplyOperator = GetBinaryOperator("*");

            SymbolicExpressionNode<T>? root = null;
            for (int i = 0; i < coefficients.Length; i++)
            {
                var coefficient = coefficients[i];
                if (NumOps.Equals(coefficient, NumOps.Zero))
                {
                    continue;
                }

                var term = new SymbolicExpressionNode<T>(
                    multiplyOperator,
                    new SymbolicExpressionNode<T>(coefficient),
                    new SymbolicExpressionNode<T>(i));

                root = root == null ? term : new SymbolicExpressionNode<T>(addOperator, root, term);
            }

            if (includeIntercept && !NumOps.Equals(intercept, NumOps.Zero))
            {
                var interceptNode = new SymbolicExpressionNode<T>(intercept);
                root = root == null ? interceptNode : new SymbolicExpressionNode<T>(addOperator, root, interceptNode);
            }

            root ??= new SymbolicExpressionNode<T>(NumOps.Zero);
            return new SymbolicExpression<T>(root, NumOps);
        }

        private SymbolicExpressionNode<T> ConvertExpressionTreeNode(ExpressionTree<T, Matrix<T>, Vector<T>> node)
        {
            switch (node.Type)
            {
                case ExpressionNodeType.Constant:
                    return new SymbolicExpressionNode<T>(node.Value);
                case ExpressionNodeType.Variable:
                    int index = NumOps.ToInt32(node.Value);
                    if (index < 0)
                    {
                        index = 0;
                    }
                    return new SymbolicExpressionNode<T>(index);
                case ExpressionNodeType.Add:
                    return new SymbolicExpressionNode<T>(
                        GetBinaryOperator("+"),
                        ConvertExpressionTreeNode(node.Left ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, NumOps.Zero)),
                        ConvertExpressionTreeNode(node.Right ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, NumOps.Zero)));
                case ExpressionNodeType.Subtract:
                    return new SymbolicExpressionNode<T>(
                        GetBinaryOperator("-"),
                        ConvertExpressionTreeNode(node.Left ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, NumOps.Zero)),
                        ConvertExpressionTreeNode(node.Right ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, NumOps.Zero)));
                case ExpressionNodeType.Multiply:
                    return new SymbolicExpressionNode<T>(
                        GetBinaryOperator("*"),
                        ConvertExpressionTreeNode(node.Left ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, NumOps.Zero)),
                        ConvertExpressionTreeNode(node.Right ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, NumOps.Zero)));
                case ExpressionNodeType.Divide:
                    return new SymbolicExpressionNode<T>(
                        GetBinaryOperator("/"),
                        ConvertExpressionTreeNode(node.Left ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, NumOps.Zero)),
                        ConvertExpressionTreeNode(node.Right ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, NumOps.Zero)));
                default:
                    return new SymbolicExpressionNode<T>(NumOps.Zero);
            }
        }

        private SymbolicBinaryOperator<T> GetBinaryOperator(string name)
        {
            var op = _binaryOperators.FirstOrDefault(candidate => candidate.Name == name);
            if (op == null)
            {
                op = _binaryOperators.First();
            }

            return op;
        }

        // === ModelBase abstract implementations ===

        /// <summary>
        /// Trains the symbolic physics learner by discovering an equation from the data.
        /// </summary>
        public override void Train(Matrix<T> input, Vector<T> expectedOutput)
        {
            var inputs = new T[input.Rows, input.Columns];
            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < input.Columns; j++)
                    inputs[i, j] = input[i, j];

            var outputs = expectedOutput.ToArray();
            _discoveredEquation = DiscoverEquation(inputs, outputs);
        }

        /// <summary>
        /// Predicts outputs using the discovered symbolic equation.
        /// </summary>
        public override Vector<T> Predict(Matrix<T> input)
        {
            if (_discoveredEquation is null)
                throw new InvalidOperationException("Model has not been trained. Call Train() first.");

            var result = new Vector<T>(input.Rows);
            for (int i = 0; i < input.Rows; i++)
            {
                var row = new T[input.Columns];
                for (int j = 0; j < input.Columns; j++)
                    row[j] = input[i, j];
                result[i] = _discoveredEquation.Evaluate(row);
            }
            return result;
        }

        /// <inheritdoc/>
        public override ILossFunction<T> DefaultLossFunction =>
            new MeanSquaredErrorLoss<T>();

        /// <inheritdoc/>
        public override Vector<T> GetParameters()
        {
            if (_discoveredEquation is null)
            {
                return new Vector<T>(0);
            }

            var constants = new List<SymbolicExpressionNode<T>>();
            CollectConstantNodes(_discoveredEquation.Root, constants);
            var parameters = new Vector<T>(constants.Count);
            for (int i = 0; i < constants.Count; i++)
            {
                parameters[i] = constants[i].Constant;
            }
            return parameters;
        }

        /// <inheritdoc/>
        public override IEnumerable<int> GetActiveFeatureIndices()
        {
            if (_discoveredEquation is null)
            {
                return Array.Empty<int>();
            }

            var indices = new HashSet<int>();
            CollectVariableIndices(_discoveredEquation.Root, indices);
            return indices.OrderBy(index => index).ToArray();
        }

        private static void CollectVariableIndices(
            SymbolicExpressionNode<T>? node,
            HashSet<int> indices)
        {
            if (node is null)
            {
                return;
            }

            if (node.Type == SymbolicExpressionType.Variable)
            {
                indices.Add(node.VariableIndex);
            }

            CollectVariableIndices(node.Left, indices);
            CollectVariableIndices(node.Right, indices);
        }

        /// <inheritdoc/>
        public override void SetParameters(Vector<T> parameters)
        {
            if (_discoveredEquation is null)
            {
                if (parameters.Length == 0)
                {
                    return;
                }
                throw new InvalidOperationException("Cannot set symbolic constants before an equation is discovered.");
            }

            var constants = new List<SymbolicExpressionNode<T>>();
            CollectConstantNodes(_discoveredEquation.Root, constants);
            if (parameters.Length != constants.Count)
            {
                throw new ArgumentException(
                    $"Expected {constants.Count} symbolic constants, got {parameters.Length}.",
                    nameof(parameters));
            }

            for (int i = 0; i < constants.Count; i++)
            {
                constants[i].Constant = parameters[i];
            }
        }

        private static void CollectConstantNodes(
            SymbolicExpressionNode<T>? node,
            List<SymbolicExpressionNode<T>> constants)
        {
            if (node is null)
            {
                return;
            }

            if (node.Type == SymbolicExpressionType.Constant)
            {
                constants.Add(node);
            }

            CollectConstantNodes(node.Left, constants);
            CollectConstantNodes(node.Right, constants);
        }

        /// <inheritdoc/>
        public override IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
        {
            var clone = new SymbolicPhysicsLearner<T>();
            if (_discoveredEquation is not null)
            {
                clone._discoveredEquation = _discoveredEquation.Clone();
            }
            return clone;
        }

        /// <inheritdoc/>
        public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
        {
            var clone = DeepCopy();
            ((IParameterizable<T, Matrix<T>, Vector<T>>)clone).SetParameters(parameters);
            return clone;
        }
    }

    internal sealed class SymbolicUnaryOperator<T>
    {
        public SymbolicUnaryOperator(string name, Func<T, T> apply, Func<string, string> formatter)
        {
            Name = name;
            Apply = apply;
            Formatter = formatter;
        }

        public string Name { get; }
        public Func<T, T> Apply { get; }
        public Func<string, string> Formatter { get; }
    }

    internal sealed class SymbolicBinaryOperator<T>
    {
        public SymbolicBinaryOperator(string name, Func<T, T, T> apply, Func<string, string, string> formatter)
        {
            Name = name;
            Apply = apply;
            Formatter = formatter;
        }

        public string Name { get; }
        public Func<T, T, T> Apply { get; }
        public Func<string, string, string> Formatter { get; }
    }

    internal sealed class SymbolicExpressionNode<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        public SymbolicExpressionNode(T constant)
        {
            Type = SymbolicExpressionType.Constant;
            Constant = constant;
        }

        public SymbolicExpressionNode(int variableIndex)
        {
            Type = SymbolicExpressionType.Variable;
            VariableIndex = variableIndex;
            Constant = NumOps.Zero;
        }

        public SymbolicExpressionNode(SymbolicUnaryOperator<T> unaryOperator, SymbolicExpressionNode<T> operand)
        {
            Type = SymbolicExpressionType.UnaryOperation;
            UnaryOperator = unaryOperator;
            Left = operand;
            Constant = NumOps.Zero;
        }

        public SymbolicExpressionNode(SymbolicBinaryOperator<T> binaryOperator, SymbolicExpressionNode<T> left, SymbolicExpressionNode<T> right)
        {
            Type = SymbolicExpressionType.BinaryOperation;
            BinaryOperator = binaryOperator;
            Left = left;
            Right = right;
            Constant = NumOps.Zero;
        }

        public SymbolicExpressionType Type { get; }
        public T Constant { get; internal set; }
        public int VariableIndex { get; }
        public SymbolicUnaryOperator<T>? UnaryOperator { get; }
        public SymbolicBinaryOperator<T>? BinaryOperator { get; }
        public SymbolicExpressionNode<T>? Left { get; set; }
        public SymbolicExpressionNode<T>? Right { get; set; }

        public int Complexity => 1 + (Left?.Complexity ?? 0) + (Right?.Complexity ?? 0);

        public T Evaluate(T[] variables, INumericOperations<T> numOps)
        {
            switch (Type)
            {
                case SymbolicExpressionType.Constant:
                    return Constant;
                case SymbolicExpressionType.Variable:
                    if (VariableIndex < 0 || VariableIndex >= variables.Length)
                    {
                        throw new ArgumentOutOfRangeException(nameof(VariableIndex));
                    }
                    return variables[VariableIndex];
                case SymbolicExpressionType.UnaryOperation:
                    var uOp = UnaryOperator ?? throw new InvalidOperationException("UnaryOperator is null.");
                    var uLeft = Left ?? throw new InvalidOperationException("Left child is null.");
                    return uOp.Apply(uLeft.Evaluate(variables, numOps));
                case SymbolicExpressionType.BinaryOperation:
                    var bOp = BinaryOperator ?? throw new InvalidOperationException("BinaryOperator is null.");
                    var bLeft = Left ?? throw new InvalidOperationException("Left child is null.");
                    var bRight = Right ?? throw new InvalidOperationException("Right child is null.");
                    return bOp.Apply(
                        bLeft.Evaluate(variables, numOps),
                        bRight.Evaluate(variables, numOps));
                default:
                    return numOps.Zero;
            }
        }

        public SymbolicExpressionNode<T> Clone()
        {
            switch (Type)
            {
                case SymbolicExpressionType.Constant:
                    return new SymbolicExpressionNode<T>(Constant);
                case SymbolicExpressionType.Variable:
                    return new SymbolicExpressionNode<T>(VariableIndex);
                case SymbolicExpressionType.UnaryOperation:
                    if (UnaryOperator is null || Left is null)
                    {
                        throw new InvalidOperationException(
                            "UnaryOperation node must have a non-null UnaryOperator and Left child.");
                    }
                    return new SymbolicExpressionNode<T>(UnaryOperator, Left.Clone());
                case SymbolicExpressionType.BinaryOperation:
                    if (BinaryOperator is null || Left is null || Right is null)
                    {
                        throw new InvalidOperationException(
                            "BinaryOperation node must have a non-null BinaryOperator, Left, and Right children.");
                    }
                    return new SymbolicExpressionNode<T>(BinaryOperator, Left.Clone(), Right.Clone());
                default:
                    return new SymbolicExpressionNode<T>(0);
            }
        }

        public string Format()
        {
            switch (Type)
            {
                case SymbolicExpressionType.Constant:
                    return Constant is null ? "0" : Constant.ToString() ?? "0";
                case SymbolicExpressionType.Variable:
                    return $"x{VariableIndex + 1}";
                case SymbolicExpressionType.UnaryOperation:
                    var fUOp = UnaryOperator ?? throw new InvalidOperationException("UnaryOperator is null.");
                    var fUL = Left ?? throw new InvalidOperationException("Left child is null.");
                    return fUOp.Formatter(fUL.Format());
                case SymbolicExpressionType.BinaryOperation:
                    var fBOp = BinaryOperator ?? throw new InvalidOperationException("BinaryOperator is null.");
                    var fBL = Left ?? throw new InvalidOperationException("Left child is null.");
                    var fBR = Right ?? throw new InvalidOperationException("Right child is null.");
                    return fBOp.Formatter(fBL.Format(), fBR.Format());
                default:
                    return "0";
            }
        }
    }

    internal readonly struct NodeReference<T>
    {
        public NodeReference(SymbolicExpressionNode<T> node, SymbolicExpressionNode<T>? parent, bool isLeft)
        {
            Node = node;
            Parent = parent;
            IsLeft = isLeft;
        }

        public SymbolicExpressionNode<T> Node { get; }
        public SymbolicExpressionNode<T>? Parent { get; }
        public bool IsLeft { get; }
    }

    internal readonly struct SymbolicCandidate<T>
    {
        public SymbolicCandidate(SymbolicExpression<T> expression, double loss)
        {
            Expression = expression;
            Loss = loss;
        }

        public SymbolicExpression<T> Expression { get; }
        public double Loss { get; }
    }

    /// <summary>
    /// Represents a symbolic mathematical expression.
    /// </summary>
    public class SymbolicExpression<T>
    {
        private readonly INumericOperations<T> NumOps;

        internal SymbolicExpression(SymbolicExpressionNode<T> root, INumericOperations<T> numOps)
        {
            NumOps = numOps;
            Root = root;
        }

        internal SymbolicExpression(SymbolicExpressionNode<T> root)
            : this(root, MathHelper.GetNumericOperations<T>())
        {
        }

        public SymbolicExpression(string expression, SymbolicExpressionType type)
            : this(ParseExpression(expression, type), MathHelper.GetNumericOperations<T>())
        {
        }

        internal SymbolicExpressionNode<T> Root { get; set; }

        public string Expression => Root.Format();

        public SymbolicExpressionType Type => Root.Type;

        public int Complexity => Root.Complexity;

        public override string ToString() => Expression;

        /// <summary>
        /// Evaluates the expression for given variable values.
        /// </summary>
        public T Evaluate(Dictionary<string, T> variables)
        {
            if (variables == null)
            {
                throw new ArgumentNullException(nameof(variables));
            }

            var ordered = MapVariables(variables);
            return Evaluate(ordered);
        }

        internal T Evaluate(T[] variables)
        {
            return Root.Evaluate(variables, NumOps);
        }

        public SymbolicExpression<T> Clone()
        {
            return new SymbolicExpression<T>(Root.Clone(), NumOps);
        }

        private static SymbolicExpressionNode<T> ParseExpression(string expression, SymbolicExpressionType type)
        {
            var numOps = MathHelper.GetNumericOperations<T>();

            if (type == SymbolicExpressionType.Constant && double.TryParse(expression, out double constant))
            {
                return new SymbolicExpressionNode<T>(numOps.FromDouble(constant));
            }

            if (type == SymbolicExpressionType.Variable)
            {
                if (TryParseVariableIndex(expression, out int index))
                {
                    return new SymbolicExpressionNode<T>(index);
                }

                return new SymbolicExpressionNode<T>(0);
            }

            return new SymbolicExpressionNode<T>(numOps.Zero);
        }

        private static bool TryParseVariableIndex(string name, out int index)
        {
            index = 0;
            if (string.IsNullOrWhiteSpace(name))
            {
                return false;
            }

            string trimmed = name.Trim();
            if (trimmed.StartsWith("x", StringComparison.OrdinalIgnoreCase))
            {
                trimmed = trimmed.Substring(1);
            }

            if (int.TryParse(trimmed, out int parsed) && parsed > 0)
            {
                index = parsed - 1;
                return true;
            }

            return false;
        }

        private static T[] MapVariables(Dictionary<string, T> variables)
        {
            int maxIndex = -1;
            foreach (var key in variables.Keys)
            {
                if (TryParseVariableIndex(key, out int index))
                {
                    maxIndex = Math.Max(maxIndex, index);
                }
            }

            if (maxIndex < 0)
            {
                return Array.Empty<T>();
            }

            var values = new T[maxIndex + 1];
            foreach (var pair in variables)
            {
                if (TryParseVariableIndex(pair.Key, out int index))
                {
                    values[index] = pair.Value;
                }
            }

            return values;
        }
    }

    /// <summary>
    /// Types of symbolic expressions.
    /// </summary>
    public enum SymbolicExpressionType
    {
        Constant,
        Variable,
        UnaryOperation,
        BinaryOperation,
        Function
    }
}
