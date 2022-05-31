# Adapted code from: https://github.com/gmihaila/ml_things/blob/master/notebooks/pytorch/pretrain_transformers_pytorch.ipynb
# Copyright 2020 George Mihaila.
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import os
import math
import torch
import argparse
import warnings
import transformers
from transformers.optimization import AdamW, Adafactor, AdafactorSchedule
from transformers import (
                          CONFIG_MAPPING,
                          MODEL_FOR_MASKED_LM_MAPPING,
                          MODEL_FOR_CAUSAL_LM_MAPPING,
                          PreTrainedTokenizer,
                          TrainingArguments,
                          AutoConfig,
                          AutoTokenizer,
                          AutoModelWithLMHead,
                          AutoModelForCausalLM,
                          AutoModelForMaskedLM,
                          LineByLineTextDataset,
                          TextDataset,
                          DataCollatorForLanguageModeling,
                          DataCollatorForWholeWordMask,
                          DataCollatorForPermutationLanguageModeling,
                          PretrainedConfig,
                          Trainer,
                          set_seed,
                          EarlyStoppingCallback,
                          )

# Set seed for reproducibility,
set_seed(123)
# Look for gpu to use. Will use `cpu` by default if no gpu found.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelDataArguments(object):
  def __init__(self, train_data_file=None, eval_data_file=None, 
               line_by_line=False, mlm=False, mlm_probability=0.15, 
               whole_word_mask=False, plm_probability=float(1/6), 
               max_span_length=5, block_size=-1, overwrite_cache=False, 
               model_type=None, model_config_name=None, tokenizer_name=None, 
               model_name_or_path=None, model_cache_dir=None):
    
    # Make sure CONFIG_MAPPING is imported from transformers module.
    if 'CONFIG_MAPPING' not in globals():
      raise ValueError('Could not find `CONFIG_MAPPING` imported! Make sure' \
                       ' to import it from `transformers` module!')

    # Make sure model_type is valid.
    if (model_type is not None) and (model_type not in CONFIG_MAPPING.keys()):
      raise ValueError('Invalid `model_type`! Use one of the following: %s' % 
                       (str(list(CONFIG_MAPPING.keys()))))
      
    # Make sure that model_type, model_config_name and model_name_or_path 
    # variables are not all `None`.
    if not any([model_type, model_config_name, model_name_or_path]):
      raise ValueError('You can`t have all `model_type`, `model_config_name`,' \
                       ' `model_name_or_path` be `None`! You need to have' \
                       'at least one of them set!')
    
    # Check if a new model will be loaded from scratch.
    if not any([model_config_name, model_name_or_path]):
      # Setup warning to show pretty. This is an overkill
      warnings.formatwarning = lambda message,category,*args,**kwargs: \
                               '%s: %s\n' % (category.__name__, message)
      # Display warning.
      warnings.warn('You are planning to train a model from scratch! 🙀')

    # Check if a new tokenizer wants to be loaded.
    # This feature is not supported!
    if not any([tokenizer_name, model_name_or_path]):
      # Can't train tokenizer from scratch here! Raise error.
      raise ValueError('You want to train tokenizer from scratch! ' \
                    'That is not possible yet! You can train your own ' \
                    'tokenizer separately and use path here to load it!')
      
    # Set all data related arguments.
    self.train_data_file = train_data_file
    self.eval_data_file = eval_data_file
    self.line_by_line = line_by_line
    self.mlm = mlm
    self.whole_word_mask = whole_word_mask
    self.mlm_probability = mlm_probability
    self.plm_probability = plm_probability
    self.max_span_length = max_span_length
    self.block_size = block_size
    self.overwrite_cache = overwrite_cache

    # Set all model and tokenizer arguments.
    self.model_type = model_type
    self.model_config_name = model_config_name
    self.tokenizer_name = tokenizer_name
    self.model_name_or_path = model_name_or_path
    self.model_cache_dir = model_cache_dir
    return


def get_model_config(args: ModelDataArguments):
  # Check model configuration.
  if args.model_config_name is not None:
    # Use model configure name if defined.
    model_config = AutoConfig.from_pretrained(args.model_config_name, 
                                      cache_dir=args.model_cache_dir)

  elif args.model_name_or_path is not None:
    # Use model name or path if defined.
    model_config = AutoConfig.from_pretrained(args.model_name_or_path, 
                                      cache_dir=args.model_cache_dir)

  else:
    # Use config mapping if building model from scratch.
    model_config = CONFIG_MAPPING[args.model_type]()

  # Make sure `mlm` flag is set for Masked Language Models (MLM).
  if (model_config.model_type in ["bert", "roberta", "distilbert", 
                                  "camembert"]) and (args.mlm is False):
    raise ValueError('BERT and RoBERTa-like models do not have LM heads ' \
                    'butmasked LM heads. They must be run setting `mlm=True`')
  
  # Adjust block size for xlnet.
  if model_config.model_type == "xlnet":
    # xlnet used 512 tokens when training.
    args.block_size = 512
    # setup memory length
    model_config.mem_len = 1024
  
  return model_config


def get_tokenizer(args: ModelDataArguments):
  # Check tokenizer configuration.
  if args.tokenizer_name:
    # Use tokenizer name if define.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, 
                                              cache_dir=args.model_cache_dir)

  elif args.model_name_or_path:
    # Use tokenizer name of path if defined.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                              cache_dir=args.model_cache_dir)
    
  # Setp data block size.
  if args.block_size <= 0:
    # Set block size to maximum length of tokenizer.
    # Input block size will be the max possible for the model.
    # Some max lengths are very large and will cause a
    args.block_size = tokenizer.model_max_length
  else:
    # Never go beyond tokenizer maximum length.
    args.block_size = min(args.block_size, tokenizer.model_max_length)

  return tokenizer
  

def get_model(args: ModelDataArguments, model_config):
  # Make sure MODEL_FOR_MASKED_LM_MAPPING and MODEL_FOR_CAUSAL_LM_MAPPING are 
  # imported from transformers module.
  if ('MODEL_FOR_MASKED_LM_MAPPING' not in globals()) and \
                ('MODEL_FOR_CAUSAL_LM_MAPPING' not in globals()):
    raise ValueError('Could not find `MODEL_FOR_MASKED_LM_MAPPING` and' \
                     ' `MODEL_FOR_MASKED_LM_MAPPING` imported! Make sure to' \
                     ' import them from `transformers` module!')
    
  # Check if using pre-trained model or train from scratch.
  if args.model_name_or_path:
    # Use pre-trained model.
    if type(model_config) in MODEL_FOR_MASKED_LM_MAPPING.keys():
      # Masked language modeling head.
      return AutoModelForMaskedLM.from_pretrained(
                        args.model_name_or_path,
                        from_tf=bool(".ckpt" in args.model_name_or_path),
                        config=model_config,
                        cache_dir=args.model_cache_dir,
                        )
    elif type(model_config) in MODEL_FOR_CAUSAL_LM_MAPPING.keys():
      # Causal language modeling head.
      return AutoModelForCausalLM.from_pretrained(
                                          args.model_name_or_path, 
                                          from_tf=bool(".ckpt" in 
                                                        args.model_name_or_path),
                                          config=model_config, 
                                          cache_dir=args.model_cache_dir)
    else:
      raise ValueError(
          'Invalid `model_name_or_path`! It should be in %s or %s!' % 
          (str(MODEL_FOR_MASKED_LM_MAPPING.keys()), 
           str(MODEL_FOR_CAUSAL_LM_MAPPING.keys())))
    
  else:
    # Use model from configuration - train from scratch.
      print("Training new model from scratch!")
      return AutoModelWithLMHead.from_config(config)


def get_dataset(args: ModelDataArguments, tokenizer: PreTrainedTokenizer, 
                evaluate: bool=False):
  # Get file path for either train or evaluate.
  file_path = args.eval_data_file if evaluate else args.train_data_file

  # Check if `line_by_line` flag is set to `True`.
  if args.line_by_line:
    # Each example in data file is on each line.
    return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, 
                                 block_size=args.block_size)
    
  else:
    # All data in file is put together without any separation.
    return TextDataset(tokenizer=tokenizer, file_path=file_path, 
                       block_size=args.block_size, 
                       overwrite_cache=args.overwrite_cache)


def get_collator(args: ModelDataArguments, model_config: PretrainedConfig, 
                 tokenizer: PreTrainedTokenizer):
  # Special dataset handle depending on model type.
  if model_config.model_type == "xlnet":
    # Configure collator for XLNET.
    return DataCollatorForPermutationLanguageModeling(
                                          tokenizer=tokenizer,
                                          plm_probability=args.plm_probability,
                                          max_span_length=args.max_span_length,
                                          )
  else:
    # Configure data for rest of model types.
    if args.mlm and args.whole_word_mask:
      # Use whole word masking.
      return DataCollatorForWholeWordMask(
                                          tokenizer=tokenizer, 
                                          mlm_probability=args.mlm_probability,
                                          )
    else:
      # Regular language modeling.
      return DataCollatorForLanguageModeling(
                                          tokenizer=tokenizer, 
                                          mlm=args.mlm, 
                                          mlm_probability=args.mlm_probability,
                                          )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--modelpath", default='NbAiLab/nb-bert-base')
    arg("--modelname", default='nb-bert')
    arg("--modeltype", default='bert')
    arg("--traindata",default='./data/train_sentences.txt')
    arg("--valdata",default='./data/val_sentences.txt')
    arg("--maxsteps", type=int, default=10000)
    arg("--blocksize", type=int, default=512)
    arg("--learningrate", type=float, default=5e-5)
    arg("--weightdecay", type=float, default=1e-2)
    arg("--batchsize", type=int, default=32)
    arg("--dotrain", default=True)
    arg("--doeval", default=True)
    arg("--language", default='nor') # change to eng or ger for corresponding dataset

    args = parser.parse_args()
    modelpath = args.modelpath
    modelname = args.modelname
    modeltype = args.modeltype
    traindata = args.traindata
    valdata = args.valdata
    maxsteps = args.maxsteps
    blocksize = args.blocksize
    learningrate = args.learningrate
    weightdecay = args.weightdecay
    batchsize = args.batchsize
    dotrain = args.dotrain
    doeval = args.doeval
    language = args.language

    # Define arguments for data, tokenizer and model arguments.
    # See comments in `ModelDataArguments` class.
    model_data_args = ModelDataArguments(
                                        train_data_file=traindata, 
                                        eval_data_file=valdata, 
                                        line_by_line=True, 
                                        mlm=True,
                                        whole_word_mask=True,
                                        mlm_probability=0.15,
                                        plm_probability=float(1/6), 
                                        max_span_length=5,
                                        block_size=blocksize, # window block size
                                        overwrite_cache=False, 
                                        model_type=modeltype, 
                                        model_config_name=modelpath, 
                                        tokenizer_name=modelpath, 
                                        model_name_or_path=modelpath, 
                                        model_cache_dir=None,
                                        )
    output_dir = 'LM-{0}-BL{1}-LR{2}-B{3}-{4}'.format(modelname,blocksize,learningrate,batchsize,language)
    # Define arguments for training
    training_args = TrainingArguments(
                              # The output directory where the model predictions 
                              # and checkpoints will be written.
                              output_dir=output_dir,
                              # Overwrite the content of the output directory.
                              overwrite_output_dir=True,
                              # Whether to run training or not.
                              do_train=dotrain, 
                              # Whether to run evaluation on the dev or not.
                              do_eval=doeval,
                              per_device_train_batch_size=batchsize,
                              per_device_eval_batch_size=batchsize,
                              # evaluation strategy to adopt during training
                              # `no`: No evaluation during training.
                              # `steps`: Evaluate every `eval_steps`.
                              # `epoch`: Evaluate every end of epoch.
                              evaluation_strategy='steps',
                              save_strategy="steps",
                              # How often to show logs. I will se this to 
                              # plot history loss and calculate perplexity.
                              logging_steps=50,
                              # Number of update steps between two 
                              # evaluations if evaluation_strategy="steps".
                              # Will default to the same value as l
                              # logging_steps if not set.
                              eval_steps = 1000,
                              # Set prediction loss to `True` in order to 
                              # return loss for perplexity calculation.
                              prediction_loss_only=True,
                              # optim='adamw_hf',
                              # The initial learning rate for Adam. 
                              # Defaults to 5e-5.
                              learning_rate = learningrate,
                              # The weight decay to apply (if not zero).
                              weight_decay=weightdecay,
                              # Epsilon for the Adam optimizer. 
                              # Defaults to 1e-8
                              adam_epsilon = 1e-8,
                              # Maximum gradient norm (for gradient 
                              # clipping). Defaults to 0.
                              max_grad_norm = 1.0,
                              # Total number of training epochs to perform 
                              # (if not an integer, will perform the 
                              # decimal part percents of
                              # the last epoch before stopping training).
                              max_steps = maxsteps,
                              # Number of updates steps before two checkpoint saves. 
                              # Defaults to 500
                              save_steps = 1000,
                              metric_for_best_model='eval_loss',
                              load_best_model_at_end = True,
                              # ignore_data_skip =True,
                              )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Load model configuration.
    print('Loading model configuration...')
    config = get_model_config(model_data_args)

    # Load model tokenizer.
    print('Loading model`s tokenizer...')
    tokenizer = get_tokenizer(model_data_args)

    # Loading model.
    print('Loading actual model...')
    model = get_model(model_data_args, config)

    # Resize model to fit all tokens in tokenizer.
    model.resize_token_embeddings(len(tokenizer))

    # Setup train dataset if `do_train` is set.
    print('Creating train dataset...')
    train_dataset = get_dataset(model_data_args, tokenizer=tokenizer, evaluate=False) if training_args.do_train else None

    # Setup evaluation dataset if `do_eval` is set.
    print('Creating evaluate dataset...')
    eval_dataset = get_dataset(model_data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None

    # Get data collator to modify data format depending on type of model used.
    data_collator = get_collator(model_data_args, config, tokenizer)

    if (len(train_dataset) // training_args.per_device_train_batch_size \
        // training_args.logging_steps * training_args.num_train_epochs) > 100:
      # Display warning.
      warnings.warn('Your `logging_steps` value will will do a lot of printing!' \
                    ' Consider increasing `logging_steps` to avoid overflowing' \
                    ' the notebook with a lot of prints!')

    # Initialize Trainer.
    print('Loading `trainer`...')
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      callbacks=[EarlyStoppingCallback(20, 0.0)],
                      # optimizers=(optimizer, lr_scheduler),
                      )


    # Check model path to save.
    if training_args.do_train:
      print('Start training...')

      # Setup model path if the model to train loaded from a local path.
      model_path = (model_data_args.model_name_or_path 
                    if model_data_args.model_name_or_path is not None and 
                    os.path.isdir(model_data_args.model_name_or_path) 
                    else None
                    )
      # Run training.
      trainer.train(model_path=model_path)
      # Save model.
      trainer.save_model()

      # For convenience, we also re-save the tokenizer to the same directory,
      # so that you can share your model easily on huggingface.co/models =).
      if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
