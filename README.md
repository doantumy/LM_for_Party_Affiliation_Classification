# Using Language Models for Classifying the Party Affiliation of Political Texts

## Data

Link to data: 
```
https://drive.google.com/drive/folders/1v5KZc8em3_Ku7THdxR0dAGkT8sZED9SD?usp=sharing
```

## Model training

To train model for Norwegian/German/English dataset, use below command and replace the argument values corresponding to datasets. Details about parameters per language are available in the paper:

### [1] Training base model:

```
# Example for Norwegian dataset
python party_affiliation.py \
        --modelpath 'NbAiLab/nb-bert-base' \
        --modelname 'nb-bert'  \
        --dataset './data/norwegian/npsc_clean.tsv' \
        --epoch 13 \
        --maxlength 512 \
        --learningrate 5e-5 \
        --weightdecay 0.01 \
        --batchsize 64 \
        --numlabels 7
```

### [2] Training weighted model:

```
# Example for Norwegian dataset
python party_affiliation_weight.py 
        --dataset './data/norwegian/npsc_clean.tsv'
        --epoch 15
        --language 'nor' \
        --learningrate 4e-5 \
        --maxlength 512 \
        --modelname 'nb-bert' \
        --modelpath 'NbAiLab/nb-bert-base' \
        --batchsize 32 \
        --numlabels 7 \
        --weightdecay 0.01
```

### [3] Continue to train language on within task data:

```
# Example for Norwegian dataset
python languagemodel.py 
        --batchsize 32 \
        --blocksize 128 \
        --doeval True \
        --dotrain True \
        --language 'nor' \
        --learningrate 2e-5 \
        --maxsteps 4000 \
        --modelname 'nb-bert' \
        --modelpath 'NbAiLab/nb-bert-base' \
        --traindata './data/norwegian/train_sentences.txt' \
        --valdata './data/norwegian/val_sentences.txt' \
        --weightdecay 0.01
```

### [4] Training model using custom language model from [3]

The syntax is the same as in [1], simply replace the path of `modelpath` to the path of your newly trained language model in [3].

### [5] Training model using weighted and custom LM from [3]

The syntax is the same as in [2], simply replace the path of `modelpath` to the path of your newly trained language model in [3].

## To cite paper:

```
@inproceedings{doan2022using,
  title={Using Language Models for Classifying the Party Affiliation of Political Texts},
  author={Doan, Tu My and Kille, Benjamin and Gulla, Jon Atle},
  booktitle={International Conference on Applications of Natural Language to Information Systems},
  pages={382--393},
  year={2022},
  organization={Springer}
}
```
