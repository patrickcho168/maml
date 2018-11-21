# Reimplementation of MAML

```
python sinusoid_regression.py
```

## Training Graphs

![Alt text](images/sinusoid_training_graphs.png?raw=true "Sinusoid Training Graphs")

## Results

### Sinusoid Pretrained

![Alt text](images/sinusoid_0_grad_steps.jpg?raw=true "Pretrained")

### Sinusoid MAML with 1 Gradient Step Meta Training

![Alt text](images/sinusoid_1_grad_steps.jpg?raw=true "MAML with 1 Gradient Step Meta Training")

### Sinusoid MAML with 2 Gradient Steps Meta Training

![Alt text](images/sinusoid_2_grad_steps.jpg?raw=true "MAML with 2 Gradient Steps Meta Training")

## References

```
@article{DBLP:journals/corr/FinnAL17,
  author    = {Chelsea Finn and
               Pieter Abbeel and
               Sergey Levine},
  title     = {Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  journal   = {CoRR},
  volume    = {abs/1703.03400},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400},
  archivePrefix = {arXiv},
  eprint    = {1703.03400},
  timestamp = {Mon, 13 Aug 2018 16:47:43 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/FinnAL17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```