import pandas as pd
from datasets import load_dataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

dataset = load_dataset("imodels/credit-card")
df = pd.DataFrame(dataset['train'])


df2 = [row for row in df.duplicated(keep=False) if row]
print('Duplicated rows: ', len(df2))

print('Dropping duplicated rows...')
df.drop_duplicates(keep=False, inplace=True)

df2 = [row for row in df.duplicated(keep=False) if row]
print('Duplicated rows: ', len(df2))
print()

p_corr, _ = pearsonr(df['limit_bal'], df['age'])
print('Pearsonr\'s correlation: ')
print(p_corr)

bill_amt = 'bill_amt'
bill_amt_name = 'bill_amt_X'
bill_amt_X = df['bill_amt1']
bill_amt_X.name = bill_amt_name
for i in range(2, 7):
    bill_amt_X += df[bill_amt + str(i)]

df[bill_amt_name] = bill_amt_X
print()

df2 = df['age'].sort_values(ascending=False)

# getting indexes for 10 the oldest
indexes = [df2.index.values[i] for i in range(10)]

labels = ('limit_bal', 'age', 'education:1', 'bill_amt_X')
print(f'{labels[0]}\t\t{labels[1]}\t\t{labels[2]}\t\t{labels[3]}')
for i in indexes:
    print(f'{df[labels[0]][i]}\t\t{df[labels[1]][i]}\t\t{df[labels[2]][i]}\t\t{df[labels[3]][i]}')


fig, ax = plt.subplots(2, 2)
# fig.tight_layout(pad=2.0)

plt.subplot(2, 2, 1)
plt.hist(df['limit_bal'])
plt.title('Histogram of limit bal')
plt.xlabel('Limit bal')
plt.ylabel('Quantity')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(df['age'], df['limit_bal'], '.')
plt.title('Age of limit bal')
plt.xlabel('Limit bal')
plt.ylabel('Age')
plt.grid()

plt.subplot(2, 2, 2)
plt.hist(df['age'])
plt.title('Histogram of age')
plt.xlabel('Age')
plt.ylabel('Quantity')
plt.grid()
plt.tight_layout()
plt.show()
