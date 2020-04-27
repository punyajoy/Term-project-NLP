###function templates 
params={
    'model':'SVM'
}

def inference_template(dataframe,params):
    ### this function takes in a dataframe 
    ### model can be LR/SVM to be provided in the params(dict)
    ### loads the model 
    ### read unlabeled data from text columns 
    ### adds a pred and pred_probab column to the dataframe and return the dataframe
    all_sentences = dataframe.text    
    
    pred_labels=[]
    pred_probab=[]
    
    for step, sent in tqdm(enumerate(all_sentences)):
        
        #### add your codes here
        probab,pred=0,0
        pred_labels.append(pred)
        pred_probab.append(probab)
    # Report the final accuracy for this validation run.
    dataframe['preds']=pred_labels
    dataframe['pred_probab']=pred_probab
    return dataframe




class inference_LIME:
    def __init__(self,model_path):
        self.model = pickle.load(open(model_path, 'rb'))
        ### load models here
    def feature_generator(sent):
        #### this will return the features for a model
        pass
       
    def inference(self,list_of_sentences):
        ### this will take in a list of sentence and return a list of probablities ( for scikit learn proba)
        pass
        

