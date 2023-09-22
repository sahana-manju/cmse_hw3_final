import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.pyplot as plt

st.title('CMSE HW 3')
st.caption('EDA on various datasets')
st.text('Sahana Manjunath')
st.divider()

df_heart=pd.read_csv('heart.csv')
df_emp=pd.read_csv('Employee.csv')
df_cancer = pd.read_csv('data.csv')

st.subheader('EDA on Heart Disease Dataset')

class hist_graph():
  def hist_plot(self,data=None):
    fig = plt.figure(figsize=(10,6)); ax = fig.gca()
    data.hist(bins=30, ax=ax)
    plt.suptitle('Heart Disease', y=1.03)    # this adds a "super" title and places it well
    plt.tight_layout()   # add some space between the plots
    return fig

st.subheader('Histogram')
h=hist_graph()
st.pyplot(h.hist_plot(df_heart))


st.subheader('Heatmap')
plt.figure(figsize=(25,15))
st.pyplot(sns.heatmap(df_heart.corr(), annot=True, cmap="crest").figure)
st.divider()

st.subheader('Displot')
st.pyplot(sns.displot(df_heart, x="thalach", hue = "target", col="target", kind="kde",rug=True).figure)

st.subheader('Relplot')
st.pyplot(sns.relplot(df_heart, x="thalach", y="trestbps", hue="target").figure)

st.subheader('Catplot')
st.pyplot(sns.catplot(df_heart, x="slope", y="thalach",hue='target', kind="swarm").figure)

st.subheader('Lmplot')
st.pyplot(sns.lmplot(x="age", y="oldpeak", hue="target", col="target", data=df_heart).figure)

#st.subheader('')
#st.pyplot(.figure)

st.subheader('EDA on Employee Dataset')

st.subheader('Histogram')
h=hist_graph()
st.pyplot(h.hist_plot(df_emp))

st.subheader('Heatmap')
plt.figure(figsize=(10,5))
st.pyplot(sns.heatmap(df_emp.corr(), annot=True, cmap="crest").figure)

st.subheader('Displot')
st.pyplot(sns.displot(df_emp, x="Age", hue = "LeaveOrNot", col="LeaveOrNot", kind="kde",rug=True).figure)

st.subheader('Catplot')
st.pyplot(sns.catplot(df_emp, x="Gender", y="Age",hue='LeaveOrNot', kind="swarm").figure)

st.subheader('Lmplot')
st.pyplot(sns.lmplot(x="Age", y="ExperienceInCurrentDomain", hue="LeaveOrNot", col="LeaveOrNot", lowess=True, data=df_emp).figure)


st.subheader('EDA on Breast Cancer Dataset')

st.subheader('Heatmap')
plt.figure(figsize=(32,30))
st.pyplot(sns.heatmap(df_cancer.corr(), annot=True, cmap="crest").figure)

#Distribution
st.subheader('Distribution plot')
plot2=st.pyplot(sns.displot(df_cancer, x="radius_mean", hue = "diagnosis", col="diagnosis", kind="kde", rug=True))

#Categorical
st.subheader('Categorical plot')
plot3=st.pyplot(sns.catplot(df_cancer, x="diagnosis", y="concavity_mean", kind="swarm"))

#Relational
st.subheader('Relational plot')
plot4=st.pyplot(sns.relplot(df_cancer, x="radius_mean", y="fractal_dimension_worst", hue="diagnosis"))

df_mpg=sns.load_dataset('mpg')


st.subheader('EDA on mpg Dataset')

st.subheader('Heatmap')
plt.figure(figsize=(20,10))
st.pyplot(sns.heatmap(df_mpg.corr(), annot=True, cmap="crest").figure)

st.subheader('Displot')
st.pyplot(sns.displot(df_mpg, x="horsepower", hue="origin", kind="kde", rug=True).figure)

st.subheader('Lmplot')
st.pyplot(sns.lmplot(x="horsepower", y="displacement", hue="origin", col="origin", lowess=True, data=df_mpg).figure)

#st.subheader('FacetGrid')
#st.pyplot(sns.FacetGrid(df_mpg, col="origin", height=2.5, col_wrap=3).figure)
#st.pyplot((g.map(sns.histplot, "displacement")).figure)

class FG():
    def draw_facet_grid(self):
      g = sns.FacetGrid(df_mpg, col="origin", height=2.5, col_wrap=3)
      g.map(sns.histplot, "displacement")
      return g


st.subheader('FacetGrid')
f = FG()
st.pyplot(f.draw_facet_grid())


