using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

internal static class SyntheticGanOptionFixture
{
    internal static bool Handles<T>(INeuralNetworkModel<T> model) => model is CTABGANPlusGenerator<T> or CopulaGANGenerator<T> or CausalGANGenerator<T> or PATEGANGenerator<T>
        or OCTGANGenerator<T>;

    internal static void PrepareOptions(ModelOptions options, string option)
    {
        var type = options.GetType();
        var definition = type.IsGenericType ? type.GetGenericTypeDefinition() : null;
        if (definition == typeof(PATEGANOptions<>))
        {
            type.GetProperty("EmbeddingDimension")?.SetValue(options, 4);
            type.GetProperty("NumTeachers")?.SetValue(options, 3);
            type.GetProperty("BatchSize")?.SetValue(options, 4);
            type.GetProperty("Epochs")?.SetValue(options, 4);
            // With unit Laplace noise the first moment is always the tightest bound.
            // A larger noise scale makes the accountant's moment-order limit observable.
            if (option == "MomentOrders") type.GetProperty("LaplaceScale")?.SetValue(options, 10.0);
            if (option == "Epochs") type.GetProperty("Epsilon")?.SetValue(options, 100.0);
            return;
        }
        if (definition != typeof(CTABGANPlusOptions<>) && definition != typeof(CopulaGANOptions<>) && definition != typeof(CausalGANOptions<>)
            && definition != typeof(OCTGANOptions<>)) return;
        void Set(string name, object value) => type.GetProperty(name)?.SetValue(options, value);
        Set("EmbeddingDimension", 4);
        Set("GeneratorDimensions", new[] { 8, 8 });
        Set("DiscriminatorDimensions", new[] { 8, 8 });
        Set("HiddenDimensions", new[] { 8, 8 });
        Set("BatchSize", 4);
        // OCT-GAN decays its rate every LearningRateDecayPeriod (2) epochs: four epochs cross a decay.
        Set("Epochs", definition == typeof(OCTGANOptions<>) ? 4 : 2);
        Set("OdeSolverSteps", 1);
        Set("VGMModes", 2);
        Set("PacSize", 1);
        Set("DiscriminatorSteps", 1);
        if (option == "MaxGradNorm") Set("MaxGradNorm", .001);
        if (option != "DiscriminatorDropout") Set("DiscriminatorDropout", 0.0);
        if (option is "ClassifierWeight" or "TargetColumnIndex") Set("TargetColumnIndex", 1);
    }

    internal static (Matrix<T> Data, List<ColumnMetadata> Columns) Data<T>()
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var data = new Matrix<T>(12, 3);
        for (int row = 0; row < data.Rows; row++)
        {
            data[row, 0] = ops.FromDouble(row / 12.0);
            data[row, 1] = ops.FromDouble(row % 2);
            data[row, 2] = ops.FromDouble(Math.Sin(row + .5));
        }
        var columns = new List<ColumnMetadata>
        {
            new("value", ColumnDataType.Continuous, columnIndex: 0),
            new("category", ColumnDataType.Categorical, columnIndex: 1),
            new("other", ColumnDataType.Continuous, columnIndex: 2),
        };
        return (data, columns);
    }

    internal static void Train<T>(INeuralNetworkModel<T> model)
    {
        var (data, columns) = Data<T>();
        switch (model)
        {
            case CTABGANPlusGenerator<T> ctab: ctab.Fit(data, columns); break;
            case CopulaGANGenerator<T> copula: copula.Fit(data, columns); break;
            case CausalGANGenerator<T> causal: causal.Fit(data, columns); break;
            case PATEGANGenerator<T> pate: pate.Fit(data, columns); break;
            case OCTGANGenerator<T> oct: oct.Fit(data, columns); break;
        }
    }

    internal static long Observe<T>(INeuralNetworkModel<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        bool fitted = model switch { CTABGANPlusGenerator<T> ctab => ctab.IsFitted, CopulaGANGenerator<T> copula => copula.IsFitted, CausalGANGenerator<T> causal => causal.IsFitted, PATEGANGenerator<T> pate => pate.IsFitted,
            OCTGANGenerator<T> oct => oct.IsFitted, _ => false };
        if (fitted)
        {
            var rows = ((ISyntheticTabularGenerator<T>)model).Generate(4);
            var values = new List<double>();
            if (model is PATEGANGenerator<T> pate)
            {
                values.Add(pate.PrivacySpent);
                values.Add(pate.IterationsTrained);
                values.Add(pate.MeanTeacherScore(pate.EncodeRows(Data<T>().Data)));
            }
            for (int r = 0; r < rows.Rows; r++) for (int c = 0; c < rows.Columns; c++) values.Add(ops.ToDouble(rows[r, c]));
            // The critic and causal graph are learned state outside the generator parameter vector.
            const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            if (model.GetType().GetField("_discLayers", flags)?.GetValue(model) is IEnumerable<AiDotNet.Interfaces.ILayer<T>> critic)
                foreach (var layer in critic) values.AddRange(layer.GetParameters().Select(ops.ToDouble));
            if (model.GetType().GetField("_adjacency", flags)?.GetValue(model) is Matrix<T> adjacency)
                for (int r = 0; r < adjacency.Rows; r++) for (int c = 0; c < adjacency.Columns; c++) values.Add(ops.ToDouble(adjacency[r, c]));
            return TtsPaperNumerics.Fingerprint(values);
        }
        var options = ((AiDotNet.NeuralNetworks.NeuralNetworkBase<T>)model).GetOptions();
        int width = options.GetType().GetProperty("EmbeddingDimension")?.GetValue(options) is int embedding ? embedding : 0;
        // OCT-GAN's unfitted generator reads the width it was built for (the architecture's input size);
        // CTAB-GAN+ and CopulaGAN read noise followed by that input.
        if (model is OCTGANGenerator<T> oct)
            width = oct.GeneratorInputWidth;
        else if (model is not CausalGANGenerator<T> and not PATEGANGenerator<T>)
            width += model.GetArchitecture().CalculatedInputSize;
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var input = engine.Reshape(engine.TensorMultiplyScalar(engine.TensorArange<T>(1, width + 1, 1), ops.FromDouble(.1)), new[] { 1, width });
        return TtsPaperNumerics.Fingerprint(model.Predict(input).Select(ops.ToDouble));
    }
}
