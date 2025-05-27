using StockTradingTest.Models;
using System.Text;

namespace StockTradingTest.Services
{
    public class TournamentReporter
    {
        private readonly List<TournamentResult> _results;
        private readonly SimulationLogger _logger;

        public TournamentReporter(List<TournamentResult> results, SimulationLogger logger)
        {
            _results = results;
            _logger = logger;
        }

        public void PrintSummary()
        {
            // Group results by round
            var resultsByRound = _results
                .GroupBy(r => r.Round)
                .OrderBy(g => g.Key)
                .ToList();
                
            Console.WriteLine("\n===== TOURNAMENT SUMMARY =====");
                
            foreach (var roundGroup in resultsByRound)
            {
                int round = roundGroup.Key;
                var roundResults = roundGroup.OrderByDescending(r => r.FinalProfitLossPercent).ToList();
                
                Console.WriteLine($"\nROUND {round} RESULTS");
                Console.WriteLine(new string('-', 80));
                Console.WriteLine($"{"Rank",-4}{"Model",-20}{"Profit",-10}{"Win Rate",-10}{"Sharpe",-10}{"MaxDD",-10}{"Trades",-8}");
                Console.WriteLine(new string('-', 80));
                
                for (int i = 0; i < roundResults.Count; i++)
                {
                    var result = roundResults[i];
                    Console.WriteLine(
                        $"{i+1,-4}{result.ModelName,-20}{result.FinalProfitLossPercent:P2,-10}" +
                        $"{result.WinRate:P2,-10}{result.Sharpe:F2,-10}{result.MaxDrawdown:P2,-10}" +
                        $"{result.TotalTrades,-8}");
                }
            }
            
            // Final rankings
            var finalRound = resultsByRound.Last().Key;
            var finalResults = _results
                .Where(r => r.Round == finalRound)
                .OrderByDescending(r => r.FinalProfitLossPercent)
                .ToList();
                
            Console.WriteLine("\n===== FINAL RANKINGS =====");
            Console.WriteLine(new string('-', 80));
            Console.WriteLine($"{"Rank",-4}{"Model",-20}{"Profit",-10}{"Win Rate",-10}{"Sharpe",-10}{"Sortino",-10}{"Calmar",-10}");
            Console.WriteLine(new string('-', 80));
            
            for (int i = 0; i < finalResults.Count; i++)
            {
                var result = finalResults[i];
                Console.WriteLine(
                    $"{i+1,-4}{result.ModelName,-20}{result.FinalProfitLossPercent:P2,-10}" +
                    $"{result.WinRate:P2,-10}{result.Sharpe:F2,-10}{result.Sortino:F2,-10}" +
                    $"{result.CalmarRatio:F2,-10}");
            }
            
            Console.WriteLine(new string('-', 80));
            
            // Winner
            var winner = finalResults.First();
            Console.WriteLine($"\nTOURNAMENT WINNER: {winner.ModelName}");
            Console.WriteLine($"Final balance: {winner.FinalBalance:C2} (Initial: {winner.InitialBalance:C2})");
            Console.WriteLine($"Profit: {winner.FinalProfitLoss:C2} ({winner.FinalProfitLossPercent:P2})");
            Console.WriteLine($"Total trades: {winner.TotalTrades} (Wins: {winner.WinningTrades}, Losses: {winner.LosingTrades})");
            Console.WriteLine($"Win rate: {winner.WinRate:P2}");
            Console.WriteLine($"Performance metrics: Sharpe: {winner.Sharpe:F2}, Sortino: {winner.Sortino:F2}, Calmar: {winner.CalmarRatio:F2}");
        }

        public void GenerateDetailedReport(string filePath)
        {
            var html = new StringBuilder();
            
            // Start HTML
            html.AppendLine("<!DOCTYPE html>");
            html.AppendLine("<html lang=\"en\">");
            html.AppendLine("<head>");
            html.AppendLine("    <meta charset=\"UTF-8\">");
            html.AppendLine("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">");
            html.AppendLine("    <title>Trading Model Tournament Report</title>");
            html.AppendLine("    <style>");
            html.AppendLine("        body { font-family: Arial, sans-serif; margin: 20px; }");
            html.AppendLine("        h1, h2, h3 { color: #333; }");
            html.AppendLine("        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }");
            html.AppendLine("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }");
            html.AppendLine("        th { background-color: #f2f2f2; }");
            html.AppendLine("        tr:nth-child(even) { background-color: #f9f9f9; }");
            html.AppendLine("        .positive { color: green; }");
            html.AppendLine("        .negative { color: red; }");
            html.AppendLine("        .chart { width: 100%; height: 400px; margin: 20px 0; }");
            html.AppendLine("    </style>");
            html.AppendLine("    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>");
            html.AppendLine("</head>");
            html.AppendLine("<body>");
            html.AppendLine("    <h1>Trading Model Tournament Report</h1>");
            
            // Summary section
            html.AppendLine("    <h2>Tournament Summary</h2>");
            
            // Group results by round
            var resultsByRound = _results
                .GroupBy(r => r.Round)
                .OrderBy(g => g.Key)
                .ToList();
                
            foreach (var roundGroup in resultsByRound)
            {
                int round = roundGroup.Key;
                var roundResults = roundGroup.OrderByDescending(r => r.FinalProfitLossPercent).ToList();
                
                html.AppendLine($"    <h3>Round {round} Results</h3>");
                html.AppendLine("    <table>");
                html.AppendLine("        <tr>");
                html.AppendLine("            <th>Rank</th>");
                html.AppendLine("            <th>Model</th>");
                html.AppendLine("            <th>Profit</th>");
                html.AppendLine("            <th>Win Rate</th>");
                html.AppendLine("            <th>Sharpe</th>");
                html.AppendLine("            <th>Max Drawdown</th>");
                html.AppendLine("            <th>Trades</th>");
                html.AppendLine("        </tr>");
                
                for (int i = 0; i < roundResults.Count; i++)
                {
                    var result = roundResults[i];
                    string profitClass = result.FinalProfitLossPercent >= 0 ? "positive" : "negative";
                    
                    html.AppendLine("        <tr>");
                    html.AppendLine($"            <td>{i+1}</td>");
                    html.AppendLine($"            <td>{result.ModelName}</td>");
                    html.AppendLine($"            <td class=\"{profitClass}\">{result.FinalProfitLossPercent:P2}</td>");
                    html.AppendLine($"            <td>{result.WinRate:P2}</td>");
                    html.AppendLine($"            <td>{result.Sharpe:F2}</td>");
                    html.AppendLine($"            <td>{result.MaxDrawdown:P2}</td>");
                    html.AppendLine($"            <td>{result.TotalTrades}</td>");
                    html.AppendLine("        </tr>");
                }
                
                html.AppendLine("    </table>");
                
                // Add performance chart for this round
                html.AppendLine($"    <h3>Round {round} Performance Chart</h3>");
                html.AppendLine($"    <div class=\"chart\">");
                html.AppendLine($"        <canvas id=\"performanceChartRound{round}\"></canvas>");
                html.AppendLine($"    </div>");
            }
            
            // Final rankings
            var finalRound = resultsByRound.Last().Key;
            var finalResults = _results
                .Where(r => r.Round == finalRound)
                .OrderByDescending(r => r.FinalProfitLossPercent)
                .ToList();
                
            html.AppendLine("    <h2>Final Rankings</h2>");
            html.AppendLine("    <table>");
            html.AppendLine("        <tr>");
            html.AppendLine("            <th>Rank</th>");
            html.AppendLine("            <th>Model</th>");
            html.AppendLine("            <th>Profit</th>");
            html.AppendLine("            <th>Win Rate</th>");
            html.AppendLine("            <th>Sharpe</th>");
            html.AppendLine("            <th>Sortino</th>");
            html.AppendLine("            <th>Calmar</th>");
            html.AppendLine("        </tr>");
            
            for (int i = 0; i < finalResults.Count; i++)
            {
                var result = finalResults[i];
                string profitClass = result.FinalProfitLossPercent >= 0 ? "positive" : "negative";
                
                html.AppendLine("        <tr>");
                html.AppendLine($"            <td>{i+1}</td>");
                html.AppendLine($"            <td>{result.ModelName}</td>");
                html.AppendLine($"            <td class=\"{profitClass}\">{result.FinalProfitLossPercent:P2}</td>");
                html.AppendLine($"            <td>{result.WinRate:P2}</td>");
                html.AppendLine($"            <td>{result.Sharpe:F2}</td>");
                html.AppendLine($"            <td>{result.Sortino:F2}</td>");
                html.AppendLine($"            <td>{result.CalmarRatio:F2}</td>");
                html.AppendLine("        </tr>");
            }
            
            html.AppendLine("    </table>");
            
            // Winner details
            var winner = finalResults.First();
            html.AppendLine("    <h2>Tournament Winner</h2>");
            html.AppendLine("    <table>");
            html.AppendLine("        <tr><th>Model</th><td>" + winner.ModelName + "</td></tr>");
            html.AppendLine($"        <tr><th>Final Balance</th><td>{winner.FinalBalance:C2} (Initial: {winner.InitialBalance:C2})</td></tr>");
            html.AppendLine($"        <tr><th>Profit</th><td class=\"{(winner.FinalProfitLossPercent >= 0 ? "positive" : "negative")}\">{winner.FinalProfitLoss:C2} ({winner.FinalProfitLossPercent:P2})</td></tr>");
            html.AppendLine($"        <tr><th>Total Trades</th><td>{winner.TotalTrades} (Wins: {winner.WinningTrades}, Losses: {winner.LosingTrades})</td></tr>");
            html.AppendLine($"        <tr><th>Win Rate</th><td>{winner.WinRate:P2}</td></tr>");
            html.AppendLine($"        <tr><th>Sharpe Ratio</th><td>{winner.Sharpe:F2}</td></tr>");
            html.AppendLine($"        <tr><th>Sortino Ratio</th><td>{winner.Sortino:F2}</td></tr>");
            html.AppendLine($"        <tr><th>Calmar Ratio</th><td>{winner.CalmarRatio:F2}</td></tr>");
            html.AppendLine($"        <tr><th>Max Drawdown</th><td>{winner.MaxDrawdown:P2}</td></tr>");
            html.AppendLine("    </table>");
            
            // Winner performance chart
            html.AppendLine("    <h3>Winner Performance</h3>");
            html.AppendLine("    <div class=\"chart\">");
            html.AppendLine("        <canvas id=\"winnerPerformanceChart\"></canvas>");
            html.AppendLine("    </div>");
            
            // Winner trades table
            html.AppendLine("    <h3>Winner Trades</h3>");
            html.AppendLine("    <table>");
            html.AppendLine("        <tr>");
            html.AppendLine("            <th>Date</th>");
            html.AppendLine("            <th>Type</th>");
            html.AppendLine("            <th>Symbol</th>");
            html.AppendLine("            <th>Quantity</th>");
            html.AppendLine("            <th>Price</th>");
            html.AppendLine("            <th>Commission</th>");
            html.AppendLine("            <th>Reason</th>");
            html.AppendLine("        </tr>");
            
            foreach (var trade in winner.Trades.OrderBy(t => t.Date))
            {
                string tradeType = trade.Type.ToString();
                string tradeClass = (trade.Type == TradeType.Buy || trade.Type == TradeType.ShortCover) ? "" : "";
                
                html.AppendLine("        <tr>");
                html.AppendLine($"            <td>{trade.Date:yyyy-MM-dd}</td>");
                html.AppendLine($"            <td>{tradeType}</td>");
                html.AppendLine($"            <td>{trade.Symbol}</td>");
                html.AppendLine($"            <td>{trade.Quantity}</td>");
                html.AppendLine($"            <td>{trade.Price:C2}</td>");
                html.AppendLine($"            <td>{trade.Commission:C2}</td>");
                html.AppendLine($"            <td>{trade.Reason}</td>");
                html.AppendLine("        </tr>");
            }
            
            html.AppendLine("    </table>");
            
            // Chart generation scripts
            html.AppendLine("    <script>");
            
            // Performance charts for each round
            foreach (var roundGroup in resultsByRound)
            {
                int round = roundGroup.Key;
                var roundResults = roundGroup.ToList();
                
                html.AppendLine($"        const ctxRound{round} = document.getElementById('performanceChartRound{round}').getContext('2d');");
                html.AppendLine($"        new Chart(ctxRound{round}, {{");
                html.AppendLine("            type: 'line',");
                html.AppendLine("            data: {");
                
                // Labels (dates)
                html.Append("                labels: [");
                if (roundResults.Any() && roundResults.First().DailySnapshots.Any())
                {
                    var dates = roundResults.First().DailySnapshots.Select(s => s.Date.ToString("yyyy-MM-dd")).ToList();
                    html.Append(string.Join(", ", dates.Select(d => $"'{d}'")));
                }
                html.AppendLine("],");
                
                // Datasets
                html.AppendLine("                datasets: [");
                
                string[] colors = { "#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D01", "#46BDC6", "#7BAAF7", "#F07B72" };
                
                for (int i = 0; i < roundResults.Count; i++)
                {
                    var result = roundResults[i];
                    string color = colors[i % colors.Length];
                    
                    html.AppendLine("                    {");
                    html.AppendLine($"                        label: '{result.ModelName}',");
                    html.AppendLine($"                        data: [{string.Join(", ", result.DailySnapshots.Select(s => s.ProfitLossPercent.ToString("0.####")))}],");
                    html.AppendLine($"                        borderColor: '{color}',");
                    html.AppendLine($"                        backgroundColor: '{color}20',");
                    html.AppendLine($"                        borderWidth: 2,");
                    html.AppendLine($"                        pointRadius: 0,");
                    html.AppendLine($"                        tension: 0.1");
                    html.AppendLine(i == roundResults.Count - 1 ? "                    }" : "                    },");
                }
                
                html.AppendLine("                ]");
                html.AppendLine("            },");
                html.AppendLine("            options: {");
                html.AppendLine("                responsive: true,");
                html.AppendLine("                plugins: {");
                html.AppendLine("                    title: {");
                html.AppendLine($"                        display: true,");
                html.AppendLine($"                        text: 'Round {round} Performance (% Return)'");
                html.AppendLine("                    },");
                html.AppendLine("                    tooltip: {");
                html.AppendLine("                        mode: 'index',");
                html.AppendLine("                        intersect: false");
                html.AppendLine("                    }");
                html.AppendLine("                },");
                html.AppendLine("                scales: {");
                html.AppendLine("                    y: {");
                html.AppendLine("                        ticks: {");
                html.AppendLine("                            callback: function(value) {");
                html.AppendLine("                                return (value * 100).toFixed(2) + '%';");
                html.AppendLine("                            }");
                html.AppendLine("                        }");
                html.AppendLine("                    }");
                html.AppendLine("                }");
                html.AppendLine("            }");
                html.AppendLine("        });");
            }
            
            // Winner performance chart
            html.AppendLine("        const ctxWinner = document.getElementById('winnerPerformanceChart').getContext('2d');");
            html.AppendLine("        new Chart(ctxWinner, {");
            html.AppendLine("            type: 'line',");
            html.AppendLine("            data: {");
            
            // Labels (dates)
            html.Append("                labels: [");
            if (winner.DailySnapshots.Any())
            {
                var dates = winner.DailySnapshots.Select(s => s.Date.ToString("yyyy-MM-dd")).ToList();
                html.Append(string.Join(", ", dates.Select(d => $"'{d}'")));
            }
            html.AppendLine("],");
            
            // Portfolio value dataset
            html.AppendLine("                datasets: [");
            html.AppendLine("                    {");
            html.AppendLine($"                        label: 'Portfolio Value',");
            html.AppendLine($"                        data: [{string.Join(", ", winner.DailySnapshots.Select(s => s.TotalValue.ToString("0.##")))}],");
            html.AppendLine($"                        borderColor: '#4285F4',");
            html.AppendLine($"                        backgroundColor: '#4285F420',");
            html.AppendLine($"                        borderWidth: 2,");
            html.AppendLine($"                        pointRadius: 0,");
            html.AppendLine($"                        tension: 0.1,");
            html.AppendLine($"                        yAxisID: 'y'");
            html.AppendLine("                    },");
            
            // Cash dataset
            html.AppendLine("                    {");
            html.AppendLine($"                        label: 'Cash',");
            html.AppendLine($"                        data: [{string.Join(", ", winner.DailySnapshots.Select(s => s.Cash.ToString("0.##")))}],");
            html.AppendLine($"                        borderColor: '#34A853',");
            html.AppendLine($"                        backgroundColor: '#34A85320',");
            html.AppendLine($"                        borderWidth: 2,");
            html.AppendLine($"                        pointRadius: 0,");
            html.AppendLine($"                        tension: 0.1,");
            html.AppendLine($"                        yAxisID: 'y'");
            html.AppendLine("                    },");
            
            // Number of positions dataset
            html.AppendLine("                    {");
            html.AppendLine($"                        label: 'Positions',");
            html.AppendLine($"                        data: [{string.Join(", ", winner.DailySnapshots.Select(s => s.NumPositions))}],");
            html.AppendLine($"                        borderColor: '#EA4335',");
            html.AppendLine($"                        backgroundColor: '#EA433520',");
            html.AppendLine($"                        borderWidth: 2,");
            html.AppendLine($"                        pointRadius: 0,");
            html.AppendLine($"                        tension: 0.1,");
            html.AppendLine($"                        yAxisID: 'y1'");
            html.AppendLine("                    }");
            html.AppendLine("                ]");
            html.AppendLine("            },");
            html.AppendLine("            options: {");
            html.AppendLine("                responsive: true,");
            html.AppendLine("                plugins: {");
            html.AppendLine("                    title: {");
            html.AppendLine($"                        display: true,");
            html.AppendLine($"                        text: '{winner.ModelName} Performance'");
            html.AppendLine("                    }");
            html.AppendLine("                },");
            html.AppendLine("                scales: {");
            html.AppendLine("                    y: {");
            html.AppendLine("                        type: 'linear',");
            html.AppendLine("                        display: true,");
            html.AppendLine("                        position: 'left',");
            html.AppendLine("                        title: {");
            html.AppendLine("                            display: true,");
            html.AppendLine("                            text: 'Value ($)'");
            html.AppendLine("                        }");
            html.AppendLine("                    },");
            html.AppendLine("                    y1: {");
            html.AppendLine("                        type: 'linear',");
            html.AppendLine("                        display: true,");
            html.AppendLine("                        position: 'right',");
            html.AppendLine("                        title: {");
            html.AppendLine("                            display: true,");
            html.AppendLine("                            text: 'Number of Positions'");
            html.AppendLine("                        },");
            html.AppendLine("                        min: 0,");
            html.AppendLine($"                        max: {_config.MaxPositions},");
            html.AppendLine("                        grid: {");
            html.AppendLine("                            drawOnChartArea: false");
            html.AppendLine("                        }");
            html.AppendLine("                    }");
            html.AppendLine("                }");
            html.AppendLine("            }");
            html.AppendLine("        });");
            
            html.AppendLine("    </script>");
            html.AppendLine("</body>");
            html.AppendLine("</html>");
            
            // Write to file
            File.WriteAllText(filePath, html.ToString());
            
            _logger.Log(StockTradingLogLevel.Information, $"Detailed report generated at {filePath}");
        }
    }
}