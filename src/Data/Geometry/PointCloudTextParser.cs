using System.Globalization;

namespace AiDotNet.Data.Geometry;

internal static class PointCloudTextParser
{
    private static readonly char[] Separators = new[] { ' ', '\t', ',' };

    public static async Task<List<double[]>> ReadRowsAsync(string path, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Path cannot be null or empty.", nameof(path));
        }

        if (!File.Exists(path))
        {
            throw new FileNotFoundException("Point cloud file not found.", path);
        }

        string[] lines = await FilePolyfill.ReadAllLinesAsync(path, cancellationToken);
        var rows = new List<double[]>(lines.Length);

        foreach (string line in lines)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] tokens = line.Split(Separators, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length < 3)
            {
                continue;
            }

            var values = new double[tokens.Length];
            for (int i = 0; i < tokens.Length; i++)
            {
                if (!double.TryParse(tokens[i], NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
                {
                    value = 0.0;
                }
                values[i] = value;
            }

            rows.Add(values);
        }

        return rows;
    }
}
