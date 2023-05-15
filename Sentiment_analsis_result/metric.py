# macro 적용 - gpt api and MultinomialNB_labeling

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd

df_result = pd.read_csv('result_run_0.csv')
y_true = df_result['original']
y_pred = df_result['predicted']

f1 = f1_score(y_true, y_pred, average='macro')
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

print(f"f1_score: {f1:.3f}, accuracy: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}")
print(len(df_result))
print(df_result['original'].value_counts())
print(df_result['predicted'].value_counts())