using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;
using System.IO;

namespace SupportTriage
{
    // 1. Классы для данных (структура CSV)
    public class TicketInput
    {
        public string Text { get; set; }
        public float TextLength { get; set; }

        [ColumnName("Label")]
        public string Sentiment { get; set; }

        // Вес класса — используется при обучении, при предсказании игнорируется
        [ColumnName("Weight")]
        public float Weight { get; set; }
    }

    // Класс для предсказания
    public class TicketPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedSentiment { get; set; }
        public float[] Score { get; set; }
    }

    // Класс для приоритизированного вывода
    public class PrioritizedTicket
    {
        public string Text { get; set; }
        public string PredictedSentiment { get; set; }
        public float Confidence { get; set; }
        public float PriorityScore { get; set; }
    }

    class Program
    {
        static string PreprocessText(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;
            return text
                .Replace(":D", " смех ")
                .Replace(":)", " улыбка ")
                .Replace("))))", " смех ")
                .Replace(")))", " смех ")
                .Replace("))", " смех ")
                .Replace("❤", " любовь ")
                .Replace("💕", " любовь ")
                .Replace("😍", " восторг ")
                .Replace("😊", " радость ")
                .Replace("(", " ")
                .Replace("  ", " ");
        }
        // 2.
        static void Main(string[] args)
        {
            // Создаём контекст ML.NET
            var mlContext = new MLContext(seed: 0);

            var tickets = new List<TicketInput>();
            const string dataPath = "rusentiment_preselected_posts.csv";

            using (var reader = new StreamReader(dataPath, System.Text.Encoding.UTF8))
            using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
                BadDataFound = null,
                MissingFieldFound = null
            }))
            {
                if (csv.Read())
                {
                    csv.ReadHeader();
                    while (csv.Read())
                    {
                        var label = csv.GetField("label")?.Trim();
                        var text = csv.GetField("text")?.Trim();

                        if (label != "positive" && label != "negative" && label != "neutral") continue;
                        if (string.IsNullOrEmpty(text)) continue;

                        // МОЖНО ОТКЛЮЧИТЬ: раскомментируй строку ниже, чтобы все веса были 1.0
                        float weight = label == "negative" ? 2.0f
                                     : label == "positive" ? 1.5f
                                     : 1.0f;

                        var processedText = PreprocessText(text);

                        tickets.Add(new TicketInput
                        {
                            Text = processedText,
                            TextLength = processedText.Length,
                            Sentiment = label,
                            Weight = weight
                        });
                    }
                }
            }

            Console.WriteLine($"Загружено записей: {tickets.Count}");
            if (tickets.Count == 0)
            {
                Console.WriteLine("Данные не загрузились. Проверь имя файла, кодировку, наличие заголовка label,text.");
                Console.ReadKey();
                return;
            }

            Console.WriteLine("\nПримеры загруженных записей:");
            foreach (var t in tickets.Take(3))
            {
                Console.WriteLine($"  [{t.Sentiment}] {t.Text}");
            }

            IDataView dataView = mlContext.Data.LoadFromEnumerable(tickets);
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // 4. Строим пайплайн
            var pipeline = mlContext.Transforms.Conversion
                .MapValueToKey("Label")
                .Append(mlContext.Transforms.Text.FeaturizeText("TextFeatures", nameof(TicketInput.Text)))
                .Append(mlContext.Transforms.Concatenate("Features",
                    "TextFeatures", nameof(TicketInput.TextLength)))
                .Append(mlContext.MulticlassClassification.Trainers
                    .LbfgsMaximumEntropy(
                    labelColumnName: "Label",
                    featureColumnName: "Features",
                    exampleWeightColumnName: "Weight"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // 5. Обучаем модель
            Console.WriteLine("Обучение модели...");
            var model = pipeline.Fit(split.TrainSet);
            Console.WriteLine("Модель обучена.");

            // 6. Оцениваем качество на тестовой выборке
            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine("\n=== Метрики качества ===");
            Console.WriteLine($"Accuracy (Точность): {metrics.MacroAccuracy:F2}");
            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:F2}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss:F2}");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

            // ========== BASELINE: ВСЕГДА NEUTRAL + MACRO-F1 ==========
            var testLabels = mlContext.Data.CreateEnumerable<TicketInput>(split.TestSet, reuseRowObject: false)
                .Select(t => t.Sentiment).ToList();
            int neutralCount = testLabels.Count(l => l == "neutral");
            double baselineAccuracy = (double)neutralCount / testLabels.Count;

            Console.WriteLine($"\n=== Базовая линия «всегда neutral» ===");
            Console.WriteLine($"Доля neutral в тесте: {baselineAccuracy:P2} ({neutralCount}/{testLabels.Count})");
            Console.WriteLine($"Accuracy модели: {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"Прирост над baseline: {(metrics.MacroAccuracy - baselineAccuracy):P2}");

            Console.WriteLine("\n=== Macro-F1 по классам ===");
            var cm = metrics.ConfusionMatrix;
            double[] f1Scores = new double[3];
            string[] classNames = { "neutral", "negative", "positive" };
            for (int i = 0; i < 3; i++)
            {
                double rowSum = 0, colSum = 0;
                for (int j = 0; j < 3; j++) { rowSum += cm.Counts[i][j]; colSum += cm.Counts[j][i]; }
                double recall = rowSum > 0 ? (double)cm.Counts[i][i] / rowSum : 0;
                double precision = colSum > 0 ? (double)cm.Counts[i][i] / colSum : 0;
                double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
                f1Scores[i] = f1;
                Console.WriteLine($"  {classNames[i],-8}: Precision={precision:F3}, Recall={recall:F3}, F1={f1:F3}");
            }
            Console.WriteLine($"  Macro-F1: {f1Scores.Average():F3}");

            // 7. Создаём движок для предсказаний
            var predEngine = mlContext.Model.CreatePredictionEngine<TicketInput, TicketPrediction>(model);

            // 8. Приоритизация новых тикетов
            var newTickets = new[]
            {
                "Приложение постоянно вылетает при открытии, это абсолютно неприемлемо!",
                "Как изменить адрес для выставления счёта?",
                "Спасибо за быструю помощь, всё работает отлично!",
                "Ну конечно, опять ошибка. Просто замечательно."
            };

            Console.WriteLine("\n=== Приоритизированный список ===");
            var results = new List<PrioritizedTicket>();

            foreach (var text in newTickets)
            {
                var processed = PreprocessText(text);
                var input = new TicketInput
                {
                    Text = processed,
                    TextLength = processed.Length,
                    Weight = 1.0f
                };
                var pred = predEngine.Predict(input);

                // Уверенность модели (максимальная вероятность)
                float confidence = pred.Score.Max();
                // Простая формула приоритета: негатив + уверенность
                float priority = pred.PredictedSentiment == "negative"
                    ? confidence * 1.0f
                    : confidence * 0.3f;

                results.Add(new PrioritizedTicket
                {
                    Text = text,
                    PredictedSentiment = pred.PredictedSentiment,
                    Confidence = confidence,
                    PriorityScore = priority
                });
            }

            foreach (var item in results.OrderByDescending(r => r.PriorityScore))
            {
                Console.WriteLine($"Приоритет: {item.PriorityScore:F2} | " +
                                  $"Тональность: {item.PredictedSentiment} ({item.Confidence:F2}) | " +
                                  $"Текст: {item.Text}");
            }

            if (File.Exists("rusentiment_test.csv"))
            {
                var testTickets = new List<TicketInput>();
                using (var reader = new StreamReader("rusentiment_test.csv", System.Text.Encoding.UTF8))
                using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
                {
                    HasHeaderRecord = true,
                    BadDataFound = null,
                    MissingFieldFound = null
                }))
                {
                    if (csv.Read())
                    {
                        csv.ReadHeader();
                        while (csv.Read())
                        {
                            var lbl = csv.GetField("label")?.Trim();
                            var txt = csv.GetField("text")?.Trim();
                            if (lbl != "positive" && lbl != "negative" && lbl != "neutral") continue;
                            if (string.IsNullOrEmpty(txt)) continue;
                            var processed = PreprocessText(txt);
                            testTickets.Add(new TicketInput
                            {
                                Text = processed,
                                TextLength = processed.Length,
                                Sentiment = lbl,
                                Weight = 1.0f
                            });
                        }
                    }
                }

                // Стратифицированная выборка: ~17 на каждый класс, seed фиксирован для воспроизводимости
                var rnd = new Random(42);
                var goldSet = new List<TicketInput>();
                foreach (var group in testTickets.GroupBy(t => t.Sentiment))
                {
                    goldSet.AddRange(group.OrderBy(_ => rnd.Next()).Take(17));
                }

                Console.WriteLine($"\n=== Проверка на золотом наборе ({goldSet.Count} записей из rusentiment_test.csv) ===");
                int correct = 0;
                foreach (var item in goldSet)
                {
                    var pred = predEngine.Predict(new TicketInput { Text = item.Text, Weight = 1.0f });
                    bool match = pred.PredictedSentiment == item.Sentiment;
                    if (match) correct++;
                    var shortText = item.Text.Length > 70 ? item.Text.Substring(0, 70) + "..." : item.Text;
                    Console.WriteLine($"[{(match ? "OK  " : "MISS")}] Истина: {item.Sentiment,-8} | Модель: {pred.PredictedSentiment,-8} | {shortText}");
                }
                Console.WriteLine($"\nСовпадение с ручной разметкой: {correct}/{goldSet.Count} = {(double)correct / goldSet.Count:P1}");
            }

            // ========== ПРОВЕРКА НА ТИКЕТАХ ПОДДЕРЖКИ ==========
            if (File.Exists("support_tickets.csv"))
            {
                var supportTickets = new List<TicketInput>();
                using (var reader = new StreamReader("support_tickets.csv", System.Text.Encoding.UTF8))
                using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
                {
                    HasHeaderRecord = true,
                    BadDataFound = null,
                    MissingFieldFound = null
                }))
                {
                    if (csv.Read())
                    {
                        csv.ReadHeader();
                        while (csv.Read())
                        {
                            var lbl = csv.GetField("label")?.Trim();
                            var txt = csv.GetField("text")?.Trim();
                            if (lbl != "positive" && lbl != "negative" && lbl != "neutral") continue;
                            if (string.IsNullOrEmpty(txt)) continue;
                            var pt = PreprocessText(txt);
                            supportTickets.Add(new TicketInput { Text = pt, TextLength = pt.Length, Sentiment = lbl, Weight = 1.0f });
                        }
                    }
                }

                Console.WriteLine($"\n=== Проверка на тикетах поддержки ({supportTickets.Count} записей) ===");
                int correctTickets = 0;
                var byClass = new Dictionary<string, (int total, int ok)>();
                foreach (var item in supportTickets)
                {
                    var pred = predEngine.Predict(item);
                    bool match = pred.PredictedSentiment == item.Sentiment;
                    if (match) correctTickets++;
                    if (!byClass.ContainsKey(item.Sentiment)) byClass[item.Sentiment] = (0, 0);
                    var (t, o) = byClass[item.Sentiment];
                    byClass[item.Sentiment] = (t + 1, o + (match ? 1 : 0));
                }
                Console.WriteLine($"Общая точность: {correctTickets}/{supportTickets.Count} = {(double)correctTickets / supportTickets.Count:P1}");
                foreach (var kv in byClass)
                {
                    Console.WriteLine($"  {kv.Key,-8}: {kv.Value.ok}/{kv.Value.total} = {(double)kv.Value.ok / kv.Value.total:P1}");
                }
            }

            // ========== СРАВНЕНИЕ ТРЕНЕРОВ ==========
            Console.WriteLine("\n=== Сравнение тренеров ===");

            var trainers = new (string Name, IEstimator<ITransformer> Pipeline)[]
            {
    ("Sdca", mlContext.Transforms.Conversion
        .MapValueToKey("Label")
        .Append(mlContext.Transforms.Text.FeaturizeText("TextFeatures", nameof(TicketInput.Text)))
        .Append(mlContext.Transforms.Concatenate("Features", "TextFeatures", nameof(TicketInput.TextLength)))
        .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features", exampleWeightColumnName: "Weight"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))),

    ("Lbfgs", mlContext.Transforms.Conversion
        .MapValueToKey("Label")
        .Append(mlContext.Transforms.Text.FeaturizeText("TextFeatures", nameof(TicketInput.Text)))
        .Append(mlContext.Transforms.Concatenate("Features", "TextFeatures", nameof(TicketInput.TextLength)))
        .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy("Label", "Features", exampleWeightColumnName: "Weight"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))),

    ("NaiveBayes", mlContext.Transforms.Conversion
        .MapValueToKey("Label")
        .Append(mlContext.Transforms.Text.FeaturizeText("Features", nameof(TicketInput.Text)))
        .Append(mlContext.MulticlassClassification.Trainers.NaiveBayes("Label", "Features"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))),
            };

            foreach (var (name, pipe) in trainers)
            {
                try
                {
                    var m = pipe.Fit(split.TrainSet);
                    var p = m.Transform(split.TestSet);
                    var met = mlContext.MulticlassClassification.Evaluate(p);
                    var cm2 = met.ConfusionMatrix;
                    double[] f1s = new double[3];
                    for (int i = 0; i < 3; i++)
                    {
                        double r = 0, c = 0;
                        for (int j = 0; j < 3; j++) { r += cm2.Counts[i][j]; c += cm2.Counts[j][i]; }
                        double rec = r > 0 ? (double)cm2.Counts[i][i] / r : 0;
                        double prec = c > 0 ? (double)cm2.Counts[i][i] / c : 0;
                        f1s[i] = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0;
                    }
                    Console.WriteLine($"  {name,-12}: Acc={met.MacroAccuracy:F3}, MicroAcc={met.MicroAccuracy:F3}, Macro-F1={f1s.Average():F3}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  {name,-12}: ошибка — {ex.Message}");
                }
            }

            Console.WriteLine("\nНажмите любую клавишу для выхода...");
            Console.ReadKey();
        }
    }
}