#!/usr/bin/env python
# coding: utf-8
# In[2]:

from bert_codes.feature_generation import combine_features,return_dataloader,return_cnngru_dataloader
from bert_codes.data_extractor import data_collector
from bert_codes.own_bert_models import *
from bert_codes.utils import *
from transformers import *
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from sklearn.metrics import accuracy_score,f1_score
from utils_function import pandas_classification_report
from sklearn.utils.class_weight import compute_class_weight
# In[3]:


if torch.cuda.is_available():    
	# Tell PyTorch to use the GPU.    
	device = torch.device("cuda")
	print('There are %d GPU(s) available.' % torch.cuda.device_count())
	print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")


# In[4]:


torch.cuda.set_device(1)


# In[ ]:





# In[ ]:





# In[5]:




# In[6]:


def Eval_phase(params,test_dataloader,which_files='test',model=None):
    model.eval()
    print("Running eval on ",which_files,"...")
    t0 = time.time()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables 
    eval_loss=0.0
    nb_eval_steps=0
    true_labels=[]
    pred_labels=[]
    # Evaluate data for one epoch
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Accumulate the total accuracy.
        pred_labels+=list(np.argmax(logits, axis=1).flatten())
        true_labels+=list(label_ids.flatten())

        # Track the number of batches
        nb_eval_steps += 1

    testf1=f1_score(true_labels, pred_labels, average='macro')
    testacc=accuracy_score(true_labels,pred_labels)

    # Report the final accuracy for this validation run.
    print(" Accuracy: {0:.2f}".format(testacc))
    print(" Fscore: {0:.2f}".format(testf1))
    print(" Test took: {:}".format(format_time(time.time() - t0)))
    return testf1,testacc,true_labels,pred_labels


# In[7]:
def save_trained_model(params):
    tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)
    total_data=pd.read_csv('Total_data_annotated.csv')
    all_sentences = total_data.text
    all_labels=total_data.label
    
    input_total_ids,att_masks_total=combine_features(all_sentences,tokenizer,params['max_length'],
                                                     take_pair=False,take_target=False)
    
    train_dataloader = return_dataloader(input_total_ids,all_labels,att_masks_total,batch_size=params['batch_size'],is_train=params['is_train'])
        
    model=select_model(params['what_bert'],params['path_files'],params['weights'])
    model.cuda()
    optimizer = AdamW(model.parameters(),
                  lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
                )
    total_steps = len(train_dataloader) * params['epochs']

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = int(total_steps/5), # Default value in run_glue.py
                                                    num_training_steps = total_steps)

    
    for epoch_i in range(0, params['epochs']):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0
            model.train()

            # For each batch of training data...
            for step, batch in tqdm(enumerate(train_dataloader)):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()        

                outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)

                # The call to `model` always returns a tuple, so we need to pull the 
                # loss value out of the tuple.
                loss = outputs[0]
                # if(params['logging']=='neptune'):
                # 	neptune.log_metric('batch_loss',loss)
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                #print(loss.item())
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()
                # Update the learning rate.
                scheduler.step()

    
    
    save_model(model,tokenizer,params)




def cross_validate_bert(params):
    
    total_data=pd.read_csv('Total_data_annotated.csv')
    all_sentences = total_data.text
    all_labels=total_data.label
    
    
    params['weights']=list(compute_class_weight("balanced", np.unique(all_labels),all_labels).astype(float))
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(params['path_files'], do_lower_case=False)
    input_total_ids,att_masks_total=combine_features(all_sentences,tokenizer,params['max_length'],
                                                     take_pair=False,take_target=False)
    
    ###optimizer
    
        
    skf=StratifiedKFold(n_splits=10, random_state=params['random_seed'], shuffle=False)
    
    list_val_accuracy=[]
    list_val_fscore=[]
    list_epoch=[]
    
    list_total_preds=[]
    list_total_truth=[]
    
    for train_index, test_index in skf.split(input_total_ids, all_labels):
        print("TRAIN:", train_index, "TEST:", test_index)
        input_train_ids,att_masks_train,labels_train=input_total_ids[train_index],att_masks_total[train_index],all_labels[train_index]
        input_val_ids,att_masks_val,labels_val=input_total_ids[test_index],att_masks_total[test_index],all_labels[test_index]
        
        model=select_model(params['what_bert'],params['path_files'],params['weights'])
        model.cuda()
        optimizer = AdamW(model.parameters(),
                      lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
                    )

        
        train_dataloader = return_dataloader(input_train_ids,labels_train,att_masks_train,batch_size=params['batch_size'],is_train=params['is_train'])
        validation_dataloader=return_dataloader(input_val_ids,labels_val,att_masks_val,batch_size=params['batch_size'],is_train=False)
        total_steps = len(train_dataloader) * params['epochs']

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        # Set the seed value all over the place to make this reproducible.
        fix_the_random(seed_val = params['random_seed'])
        # Store the averaggit pull origin master --allow-unrelated-historiese loss after each epoch so we can plot them.
        loss_values = []

        bert_model = params['path_files']
        best_val_fscore=0
        best_val_accuracy=0
        epoch_count=0
        best_true_labels=[]
        best_pred_labels=[]
        for epoch_i in range(0, params['epochs']):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0
            model.train()

            # For each batch of training data...
            for step, batch in tqdm(enumerate(train_dataloader)):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()        

                outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)

                # The call to `model` always returns a tuple, so we need to pull the 
                # loss value out of the tuple.
                loss = outputs[0]
                # if(params['logging']=='neptune'):
                # 	neptune.log_metric('batch_loss',loss)
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                #print(loss.item())
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()
                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)
            train_fscore,train_accuracy,_,_=Eval_phase(params,train_dataloader,'train',model)
            print('avg_train_loss',avg_train_loss)
            print('train_fscore',train_fscore)
            print('train_accuracy',train_accuracy)
            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)
            val_fscore,val_accuracy,true_labels,pred_labels=Eval_phase(params,validation_dataloader,'val',model)		
            #Report the final accuracy for this validation run.
            if(val_fscore > best_val_fscore):
                print(val_fscore,best_val_fscore)
                best_val_fscore=val_fscore
                best_val_accuracy=val_accuracy
                epoch_count=epoch_i
                best_pred_labels=pred_labels
                best_true_labels=true_labels
        list_total_preds+=best_pred_labels
        list_total_truth+=best_true_labels
        list_val_fscore.append(best_val_fscore)
        list_val_accuracy.append(best_val_accuracy)
        list_epoch.append(epoch_count)
       
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.array(list_val_accuracy).mean(), np.array(list_val_accuracy).std() * 2))
    print("Fscore: %0.2f (+/- %0.2f)" % (np.array(list_val_fscore).mean(), np.array(list_val_fscore).std() * 2))
    print("Epoch: %0.2f (+/- %0.2f)" % (np.array(list_epoch).mean(), np.array(list_epoch).std() * 2))
    print(pandas_classification_report(list_total_truth, list_total_preds))
    

# In[8]:
params={
    'max_length':512,
    'path_files': '../../multilingual_hatespeech/multilingual_bert',
    'what_bert':'weighted',
    'batch_size':8,
    'is_train':True,
    'learning_rate':2e-5,
    'epsilon':1e-8,
    'random_seed':2020,
    'epochs':10,
    'to_save':True,
    'weights':[1.0,9.0]

}


cross_validate_bert(params)
#save_trained_model(params)
