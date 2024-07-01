import evaluate

precision_metric = evaluate.load("precision")
results = precision_metric.compute(references=[0, 1], predictions=[0, 1])
print(evaluate.list_evaluation_modules())