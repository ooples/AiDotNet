using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.SyntheticData;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

internal static class SyntheticGanOptionFixture
{
    internal static bool Handles<T>(INeuralNetworkModel<T> model) => model is CTABGANPlusGenerator<T> or CopulaGANGenerator<T> or CausalGANGenerator<T>;

    internal static void PrepareOptions(ModelOptions options, string option)
    {
        var type = options.GetType();
        var definition = type.IsGenericType ? type.GetGenericTypeDefinition() : null;
        if (definition != typeof(CTABGANPlusOptions<>) && definition != typeof(CopulaGANOptions<>) && definition != typeof(CausalGANOptions<>)) return;
        void Set(string name, object value) => type.GetProperty(name)?.SetValue(options, value);
        Set("EmbeddingDimension", 4);
        Set("GeneratorDimensions", new[] { 8, 8 });
        Set("DiscriminatorDimensions", new[] { 8, 8 });
        Set("HiddenDimensions", new[] { 8, 8 });
        Set("BatchSize", 4);
        Set("Epochs", 2);
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
        }
    }

    internal static long Observe<T>(INeuralNetworkModel<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        bool fitted = model switch { CTABGANPlusGenerator<T> ctab => ctab.IsFitted, CopulaGANGenerator<T> copula => copula.IsFitted, CausalGANGenerator<T> causal => causal.IsFitted, _ => false };
        if (fitted)
        {
            var rows = ((ISyntheticTabularGenerator<T>)model).Generate(4);
            var values = new List<double>();
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
        int width = (int)options.GetType().GetProperty("EmbeddingDimension")!.GetValue(options)!;
        if (model is not CausalGANGenerator<T>) width += model.GetArchitecture().CalculatedInputSize;
        var input = new Tensor<T>(new[] { 1, width });
        for (int i = 0; i < input.Length; i++) input[i] = ops.FromDouble(.1 * (i + 1));
        return TtsPaperNumerics.Fingerprint(model.Predict(input).Select(ops.ToDouble));
    }
}
