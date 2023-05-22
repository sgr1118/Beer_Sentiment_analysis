# weighted 적용 - Alpaca label

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
import pandas as pd

df_result = pd.read_csv('/home/beerlab/result_run_multi_class.csv')
y_true = df_result['original']
y_pred = df_result['predicted']

f1 = f1_score(y_true, y_pred, average='weighted')
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

report = classification_report(y_true, y_pred)

print('result_run_len_ classification_report의 결과')
print(f"f1_score: {f1:.3f}, accuracy: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}")
print(len(df_result))
print(df_result['original'].value_counts())
print(df_result['predicted'].value_counts())
print(report)