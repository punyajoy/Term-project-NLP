import matplotlib.style as style
import matplotlib.pyplot as plt
import pandas as pd
style.use('default') #sets the size of the charts
import seaborn as sns
sns.set(style="whitegrid")

sns.set(font_scale=1)  # crazy big
### font_size
fs=14

def plotbox(list_of_lists,names,x_axis,y_axis,title=None,to_save=False):
    list_all=[]
    assert len(list_of_lists)==len(names)
    for i in range(len(list_of_lists)):
        for element in list_of_lists[i]:
            list_all.append((names[i],element))

    df=pd.DataFrame(list_all,columns=[x_axis,y_axis])
    ax = sns.boxplot(x=x_axis, y=y_axis, data=df,width=0.4,orient="v",palette="Spectral")
    if(to_save==True):
        fig = ax.get_figure()
        fig.savefig('Results/'+title+'.pdf')
    else:
        plt.show()



        
def plotmultibox(lists,x_axis,y_axis,category,title=None,to_save=False):
    
    df=pd.DataFrame(lists,columns=[x_axis,y_axis,category])
    print(df.head(5))
    ax = sns.boxplot(x=y_axis, y=x_axis,hue=category,data=df,width=0.4,orient="v",palette="Spectral")
    if(to_save==True):
        fig = ax.get_figure()
        
        fig.savefig('Results/'+title+'.pdf',bbox='tight')
    else:
        plt.show()


def plotmultibar(lists,x_axis,y_axis,category,title=None,to_save=False):
    
    df=pd.DataFrame(lists,columns=[x_axis,y_axis,category])
    print(df.head(5))
    ax = sns.barplot(x=y_axis, y=x_axis,hue=category,data=df,orient="v",palette="bright")
    ax.set_xticklabels(
        ax.get_xticklabels(), 
        rotation=45, 
        horizontalalignment='right',
        fontweight='bold',
        fontsize='medium'

    )
    
    
    if(to_save==True):
        fig = ax.get_figure()
        plt.tight_layout()
        plt.savefig('Results/'+title+'.jpg',bbox='tight',dpi=400)
    else:
        plt.show()




    

