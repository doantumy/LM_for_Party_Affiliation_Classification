import os
import pandas as pd
import torch
from torch.utils import data
from transformers import AdamW
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from random import random, randint
from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed, EarlyStoppingCallback
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, GPT2ForSequenceClassification
import gc

gc.collect()
set_seed(456)
torch.cuda.empty_cache()

# StortingetDS class
class StortingetDataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)

def compute_metrics(pred):
	labels = pred.label_ids
	preds = pred.predictions.argmax(-1)
	acc = accuracy_score(labels, preds)
	print("\nAccuracy: ", acc)
	table = pd.DataFrame({'preds':preds,
						'labels':labels})
	table.to_csv("./{0}/{1}_{2}_{3}_{4}_{5}_{6}.tsv".format(outputdir,modelname,maxlength,learningrate,weightdecay,batchsize,language), sep="\t")
	return {
		'accuracy': acc,
	}

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg("--modelpath", default='NbAiLab/nb-bert-base')
	arg("--modelname", default='nb-bert') # Either gpt2 or any other names
	arg("--language", default='nor')
	arg("--dataset",default='./data/npsc_clean.tsv')
	arg("--epoch", type=int, default=20)
	arg("--maxlength", type=int, default=512)
	arg("--learningrate", type=float, default=5e-5)
	arg("--weightdecay", type=float, default=1e-4)
	arg("--warmupstep", type=int, default=100)
	arg("--batchsize", type=int, default=32)
	arg("--numlabels", type=int, default=7)
	arg("--accumsteps", type=int, default=1)

	args = parser.parse_args()
	modelpath = args.modelpath
	modelname = args.modelname
	language = args.language
	dataset = args.dataset
	epoch = args.epoch
	maxlength = args.maxlength
	learningrate = args.learningrate
	weightdecay = args.weightdecay
	warmupstep = args.warmupstep
	batchsize = args.batchsize
	numlabels = args.numlabels
	accumsteps = args.accumsteps

	# Create folder name for saving model
	outputdir = '{0}-L{1}-LR{2}-W{3}-B{4}-AC{5}-{6}'.format(modelname, maxlength, learningrate, weightdecay, batchsize, accumsteps, language)
	
	# Load dataset
	raw_data = pd.read_csv(dataset, sep='\t')
	if language == 'nor':
		values_dict = {'Arbeiderpartiet':0,
               'Fremskrittspartiet':1,
               'Høyre':2,
               'Kristelig Folkeparti':3,
               'SV – Sosialistisk Venstreparti':4,
               'Senterpartiet':5,
               'Venstre':6}
	elif language == 'ger':
		values_dict = {'Alternative für Deutschland':0,
               'Bündnis 90/Die Grünen':1,
               'Christlich Demokratische Union Deutschlands/Christlich-Soziale Union in Bayern':2,
               'DIE LINKE.':3,
               'Fraktionslos':4,
               'Freie Demokratische Partei':5,
               'Partei des Demokratischen Sozialismus':6,
               'Sozialdemokratische Partei Deutschlands':7
               }
	else:
		values_dict = {'conservative':0,
               'dup':1,
               'independent':2,
               'labour':3,
               'labourco-operative':4,
               'liberal-democrat':5,
               'plaid-cymru':6,
               'scottish-national-party':7,
               'social-democratic-and-labour-party':8}

	raw_data['labels'] = raw_data['party'].map(values_dict)

	x_train, x_testval, y_train, y_testval = train_test_split(list(raw_data['speech']), 
														list(raw_data['labels']), 
														random_state=456, shuffle=True, 
														stratify=list(raw_data['labels']), 
														test_size=.25)
	x_val, x_test, y_val, y_test = train_test_split(x_testval, y_testval, 
													random_state=456,
													shuffle=True,
													stratify=y_testval,
													test_size=.75)
	# Tokenizer
	if modelname == 'gpt2':
		tokenizer = AutoTokenizer.from_pretrained(modelpath)
		tokenizer.padding_side = 'left' # default to left padding
		tokenizer.pad_token = tokenizer.eos_token # Define PAD Token = <|endoftext|> = 50256
		data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	else:
		tokenizer = AutoTokenizer.from_pretrained(modelpath)
		data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

	train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=maxlength)
	val_encodings = tokenizer(x_val, truncation=True, padding=True, max_length=maxlength)
	test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=maxlength)

	train_dataset = StortingetDataset(train_encodings, y_train)
	val_dataset = StortingetDataset(val_encodings, y_val)
	test_dataset = StortingetDataset(test_encodings, y_test)

	if modelname == 'gpt2':
		model = GPT2ForSequenceClassification.from_pretrained(modelpath, num_labels=numlabels)
		model.resize_token_embeddings(len(tokenizer))
		# resize model embedding to match new tokenizer
		model.resize_token_embeddings(len(tokenizer))
		# fix model padding token id
		model.config.pad_token_id = model.config.eos_token_id
	else:
		model = AutoModelForSequenceClassification.from_pretrained(modelpath, num_labels=numlabels)
	
	os.environ["TOKENIZERS_PARALLELISM"] = "false"

	training_args = TrainingArguments(
		evaluation_strategy='epoch',
		save_strategy='epoch',
		learning_rate=learningrate,
		optim='adamw_hf',
		per_device_train_batch_size=batchsize,
		per_device_eval_batch_size=batchsize,
		num_train_epochs=epoch,
		weight_decay=weightdecay,
		warmup_steps=warmupstep,
		load_best_model_at_end=True,
		metric_for_best_model='accuracy',
		output_dir=outputdir,
		logging_steps=50,
		fp16=True,
		gradient_accumulation_steps=accumsteps,
	)
	print('Starting to train model.')
	trainer = Trainer(
		model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		tokenizer=tokenizer,
		compute_metrics=compute_metrics,
		callbacks=[EarlyStoppingCallback(2, 0.0)],
	)

	trainer.train()

	print("Testing model on test set \n")
	trainer_test = Trainer(
	  model,
	  args=training_args,
	  train_dataset=train_dataset,
	  eval_dataset=test_dataset,
	  compute_metrics=compute_metrics,
	)
	trainer_test.evaluate()
	

