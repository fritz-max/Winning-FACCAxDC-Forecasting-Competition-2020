import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()



df = pd.read_csv('Case_material/train/X_train.csv')
print(df.head())
plt.figure()
plt.plot(df.Madrid_tcc)
plt.hist(df.Madrid_tcc, 100)
plt.plot()