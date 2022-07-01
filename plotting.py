import pandas as pd
import math


df = pd.read_table('data/7x7_prefix.txt', names=['score'])
plt = df.plot()
plt.set(xlabel="Iterations",ylabel="Average score from top 10%")
plt.set_title("Prefix discrepancy for 7x7 matrix", pad=20)
fig = plt.get_figure()
fig.tight_layout()
fig.savefig('plots/7x7_prefix.pdf')

df = pd.read_table('data/7x7_prefix_zoom.txt', names=['score'])
plt = df.plot()
plt.set(xlabel="Iterations",ylabel="Average score from top 10%")
plt.set_title("Prefix discrepancy for 7x7 matrix (zoomed)", pad=20)
fig = plt.get_figure()
fig.tight_layout()
fig.savefig('plots/7x7_prefix_zoom.pdf')

df = pd.read_table('data/7x7_fractional_bigNN.txt', names=['score'])
max = df.shape[0]
index = range(0, 20*max, 20)
df.index = index
df = df.iloc[::20, :]
plt = df.plot()
plt.set(xlabel="Iterations",ylabel="Average score from top 10%")
plt.set_title("Fractional discrepancy for 7x7 matrix", pad=20)
fig = plt.get_figure()
fig.tight_layout()
fig.savefig('plots/7x7_fractional_bigNN.pdf')

df = pd.read_table('data/7x7_normal_20.txt', names=['score'])
max = df.shape[0]
index = range(0, 20*max, 20)
df.index = index
plt = df.plot()
plt.set(xlabel="Iterations",ylabel="Average score from top 10%")
plt.set_title("Classical discrepancy for 7x7 matrix", pad=20)
fig = plt.get_figure()
fig.tight_layout()
fig.savefig('plots/7x7_normal_20.pdf')



df = pd.read_table('data/7x7_normal_20.txt', names=['score'])

def calc_count(score):
    return math.exp((2-score)*10000)

df['count'] = df.apply(lambda row: calc_count(row['score']), axis=1)
max = df.shape[0]
index = range(0, 20*max, 20)
df.index = index
plt = df.plot(y="count", color="orange", yticks=[4,6,8,10,12,14,16,18,20])
plt.set(xlabel="Iterations",ylabel="Number of colorings from top 10%")
plt.set_title("Colorings of classical discrepancy for 7x7 matrix", pad=20)
fig = plt.get_figure()
fig.tight_layout()
fig.savefig('plots/7x7_normal_count.pdf')