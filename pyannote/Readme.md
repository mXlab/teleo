PyAnnote Audio Teleo
====================

Installation
------------

```
conda create -n pyannote-env
conda activate pyannote-env
pip install -r requirements.txt
```

Training
--------

To train a model:

```
python pyannote_embeddings_training.py
```

This will perform the training and save the model in file ``trustnet.pt``.

Testing
-------

To test the model:

```
python pyannote_embeddings_testing.py
```

This will extract embeddings from a set of short voice utterances and compare
them with each other and generate a heat map.