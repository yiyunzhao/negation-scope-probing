1. To extract the attention for the data (e.g. stor_data)
python3 attention_extractor.py --modelpath 'bert-base-uncased' --outpath 'test_pretrain_story.pickle' --datapath 'data_story.json' --MODELTYPE bert --UNCASE True --MODE mean

2. To score the individual attention performance
python3 scoring.py test_pretrain_story.pickle