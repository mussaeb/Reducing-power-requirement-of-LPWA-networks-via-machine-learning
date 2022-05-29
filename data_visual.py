import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

def draw_hist(dataframe, column_name):
	""" Draws histogram of data """
	data_reader.hist(column=column_name,bins=100)
	plt.tight_layout()
	plt.show()

def draw_heatmap(dataframe): 
	""" Draws heatmap with correlation coefficient """ 
	rc={'axes.labelsize': 6, 'font.size': 4, 'legend.fontsize': 12.0, 'axes.titlesize': 12}
	sns.set_context(rc=rc)
	fig = plt.figure(figsize=(14, 6))
	corr = dataframe.corr()
	plt.title("Correlation Heatmap of Columns\nAfter data clean", fontsize=12)
	sns.heatmap(corr,cmap="YlGnBu",annot=True, annot_kws={"size": 3})
	plt.tight_layout()
	plt.show()

if __name__=='__main__': 
	filename = 'Data_set_no_formula.xlsx' 
	data_reader = pd.read_excel(
            filename, # path to the excel file that contains dataset
            )

	# Drawing heatmap with correlation and histogram
	cols = list(data_reader.columns) 
	# draw_heatmap(data_reader)
	for data in cols: 
		draw_hist(data_reader, data)
	sns.pairplot(data_reader, kind="scatter")
	plt.show()


