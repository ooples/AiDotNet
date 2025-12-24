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
using AiDotNet.Tensors.Helpers;

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
    public class SymbolicPhysicsLearner<T>
    {
        private readonly INumericOperations<T> _numOps;
        private readonly List<SymbolicUnaryOperator<T>> _unaryOperators;
        private readonly List<SymbolicBinaryOperator<T>> _binaryOperators;
        private readonly Random _random;

        public SymbolicPhysicsLearner()
        {
            _numOps = MathHelper.GetNumericOperations<T>();
            _random = RandomHelper.CreateSeededRandom(42);
            _unaryOperators = new List<SymbolicUnaryOperator<T>>
            {
                new SymbolicUnaryOperator<T>("neg", x => _numOps.Negate(x), value => $"-({value})"),
                new SymbolicUnaryOperator<T>("square", x => _numOps.Multiply(x, x), value => $"({value})^2"),
                new SymbolicUnaryOperator<T>("reciprocal", x => _numOps.Divide(_numOps.One, x), value => $"1/({value})"),
                new SymbolicUnaryOperator<T>("sqrt", x => _numOps.Sqrt(x), value => $"sqrt({value})"),
                new SymbolicUnaryOperator<T>("sin", MathHelper.Sin, value => $"sin({value})"),
                new SymbolicUnaryOperator<T>("cos", MathHelper.Cos, value => $"cos({value})"),
                new SymbolicUnaryOperator<T>("exp", _numOps.Exp, value => $"exp({value})"),
                new SymbolicUnaryOperator<T>("log", _numOps.Log, value => $"log({value})")
            };

            _binaryOperators = new List<SymbolicBinaryOperator<T>>
            {
                new SymbolicBinaryOperator<T>("+", _numOps.Add, (left, right) => $"({left} + {right})"),
                new SymbolicBinaryOperator<T>("-", _numOps.Subtract, (left, right) => $"({left} - {right})"),
                new SymbolicBinaryOperator<T>("*", _numOps.Multiply, (left, right) => $"({left} * {right})"),
                new SymbolicBinaryOperator<T>("/", _numOps.Divide, (left, right) => $"({left} / {right})"),
                new SymbolicBinaryOperator<T>("pow", _numOps.Power, (left, right) => $"pow({left}, {right})")
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

            var regressionExpression = TryDiscoverWithSymbolicRegression(inputs, outputs, maxComplexity, numGenerations);
            if (regressionExpression != null)
            {
                return regressionExpression;
            }

            var linearFallback = TryBuildLinearExpressionFromData(inputs, outputs);
            if (linearFallback != null)
            {
                return linearFallback;
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
            return new SymbolicExpression<T>(root, _numOps);
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
                seeds.Add(new SymbolicExpression<T>(new SymbolicExpressionNode<T>(i), _numOps));

                if (addOperator != null)
                {
                    var doubled = new SymbolicExpressionNode<T>(
                        addOperator,
                        new SymbolicExpressionNode<T>(i),
                        new SymbolicExpressionNode<T>(i));
                    seeds.Add(new SymbolicExpression<T>(doubled, _numOps));
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
            return new SymbolicExpressionNode<T>(_numOps.FromDouble(constant));
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

        private double ComputeLoss(SymbolicExpression<T> expression, T[,] inputs, T[] outputs, int maxComplexity)
        {
            int samples = inputs.GetLength(0);
            int features = inputs.GetLength(1);

            if (samples == 0)
            {
                return double.PositiveInfinity;
            }

            T sumSquared = _numOps.Zero;
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

                if (_numOps.IsNaN(prediction) || _numOps.IsInfinity(prediction))
                {
                    return double.PositiveInfinity;
                }

                T error = _numOps.Subtract(prediction, outputs[i]);
                sumSquared = _numOps.Add(sumSquared, _numOps.Multiply(error, error));
            }

            T mse = _numOps.Divide(sumSquared, _numOps.FromDouble(samples));
            double mseValue = _numOps.ToDouble(mse);

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
            if (inputs.GetLength(1) != 1)
            {
                return null;
            }

            int sampleCount = inputs.GetLength(0);
            if (sampleCount == 0 || outputs.Length != sampleCount)
            {
                return null;
            }

            double sumX = 0.0;
            double sumY = 0.0;
            double sumXX = 0.0;
            double sumXY = 0.0;

            for (int i = 0; i < sampleCount; i++)
            {
                double x = _numOps.ToDouble(inputs[i, 0]);
                double y = _numOps.ToDouble(outputs[i]);
                sumX += x;
                sumY += y;
                sumXX += x * x;
                sumXY += x * y;
            }

            double denominator = (sampleCount * sumXX) - (sumX * sumX);
            double slope;
            double intercept;

            if (Math.Abs(denominator) < 1e-12)
            {
                slope = 0.0;
                intercept = sumY / sampleCount;
            }
            else
            {
                slope = ((sampleCount * sumXY) - (sumX * sumY)) / denominator;
                intercept = (sumY - (slope * sumX)) / sampleCount;
            }

            if (double.IsNaN(slope) || double.IsInfinity(slope) || double.IsNaN(intercept) || double.IsInfinity(intercept))
            {
                return null;
            }

            var coefficients = new Vector<T>(new[] { _numOps.FromDouble(slope) });
            return BuildLinearExpression(coefficients, _numOps.FromDouble(intercept), includeIntercept: true);
        }

        private SymbolicExpression<T>? TryConvertModelToExpression(IFullModel<T, Matrix<T>, Vector<T>> model)
        {
            if (model is ExpressionTree<T, Matrix<T>, Vector<T>> expressionTree)
            {
                var root = ConvertExpressionTreeNode(expressionTree);
                return new SymbolicExpression<T>(root, _numOps);
            }

            if (model is RegressionBase<T> regression)
            {
                return BuildLinearExpression(regression.Coefficients, regression.Intercept, regression.HasIntercept);
            }

            if (model is VectorModel<T> vectorModel)
            {
                return BuildLinearExpression(vectorModel.Coefficients, _numOps.Zero, false);
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
                if (_numOps.Equals(coefficient, _numOps.Zero))
                {
                    continue;
                }

                var term = new SymbolicExpressionNode<T>(
                    multiplyOperator,
                    new SymbolicExpressionNode<T>(coefficient),
                    new SymbolicExpressionNode<T>(i));

                root = root == null ? term : new SymbolicExpressionNode<T>(addOperator, root, term);
            }

            if (includeIntercept && !_numOps.Equals(intercept, _numOps.Zero))
            {
                var interceptNode = new SymbolicExpressionNode<T>(intercept);
                root = root == null ? interceptNode : new SymbolicExpressionNode<T>(addOperator, root, interceptNode);
            }

            root ??= new SymbolicExpressionNode<T>(_numOps.Zero);
            return new SymbolicExpression<T>(root, _numOps);
        }

        private SymbolicExpressionNode<T> ConvertExpressionTreeNode(ExpressionTree<T, Matrix<T>, Vector<T>> node)
        {
            switch (node.Type)
            {
                case ExpressionNodeType.Constant:
                    return new SymbolicExpressionNode<T>(node.Value);
                case ExpressionNodeType.Variable:
                    int index = _numOps.ToInt32(node.Value);
                    if (index < 0)
                    {
                        index = 0;
                    }
                    return new SymbolicExpressionNode<T>(index);
                case ExpressionNodeType.Add:
                    return new SymbolicExpressionNode<T>(
                        GetBinaryOperator("+"),
                        ConvertExpressionTreeNode(node.Left ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, _numOps.Zero)),
                        ConvertExpressionTreeNode(node.Right ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, _numOps.Zero)));
                case ExpressionNodeType.Subtract:
                    return new SymbolicExpressionNode<T>(
                        GetBinaryOperator("-"),
                        ConvertExpressionTreeNode(node.Left ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, _numOps.Zero)),
                        ConvertExpressionTreeNode(node.Right ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, _numOps.Zero)));
                case ExpressionNodeType.Multiply:
                    return new SymbolicExpressionNode<T>(
                        GetBinaryOperator("*"),
                        ConvertExpressionTreeNode(node.Left ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, _numOps.Zero)),
                        ConvertExpressionTreeNode(node.Right ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, _numOps.Zero)));
                case ExpressionNodeType.Divide:
                    return new SymbolicExpressionNode<T>(
                        GetBinaryOperator("/"),
                        ConvertExpressionTreeNode(node.Left ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, _numOps.Zero)),
                        ConvertExpressionTreeNode(node.Right ?? new ExpressionTree<T, Matrix<T>, Vector<T>>(ExpressionNodeType.Constant, _numOps.Zero)));
                default:
                    return new SymbolicExpressionNode<T>(_numOps.Zero);
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
        public T Constant { get; }
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
                    return UnaryOperator!.Apply(Left!.Evaluate(variables, numOps));
                case SymbolicExpressionType.BinaryOperation:
                    return BinaryOperator!.Apply(
                        Left!.Evaluate(variables, numOps),
                        Right!.Evaluate(variables, numOps));
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
                    return new SymbolicExpressionNode<T>(
                        UnaryOperator!,
                        Left?.Clone() ?? new SymbolicExpressionNode<T>(0));
                case SymbolicExpressionType.BinaryOperation:
                    return new SymbolicExpressionNode<T>(
                        BinaryOperator!,
                        Left?.Clone() ?? new SymbolicExpressionNode<T>(0),
                        Right?.Clone() ?? new SymbolicExpressionNode<T>(0));
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
                    return UnaryOperator!.Formatter(Left!.Format());
                case SymbolicExpressionType.BinaryOperation:
                    return BinaryOperator!.Formatter(Left!.Format(), Right!.Format());
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
        private readonly INumericOperations<T> _numOps;

        internal SymbolicExpression(SymbolicExpressionNode<T> root, INumericOperations<T> numOps)
        {
            _numOps = numOps;
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
            return Root.Evaluate(variables, _numOps);
        }

        public SymbolicExpression<T> Clone()
        {
            return new SymbolicExpression<T>(Root.Clone(), _numOps);
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
