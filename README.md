# SNUH

The PyTorch implementation of "Integrating Semantics and Neighborhood Information with Graph-Driven Generative Models for Document Retrieval" (ACL 2021).

## Datasets

We follow the setting of VDSH [(Chaidaroon and Fang, 2017)](https://arxiv.org/pdf/1708.03436.pdf).  Please download the data from [here](https://github.com/unsuthee/VariationalDeepSemanticHashing/tree/master/dataset) and move them into the `./data/` directory.

## Quick start

Unsupervised document hashing on 20Newsgroups using 64 bits

```bash
python main.py ng64 data/ng20.tfidf.mat --train --cuda
```

To reproduce the results reported in the paper, please refer to the `run.sh` for detailed running comments.

## Acknowledgement

The coding logic follows the project organization in [AMMI](https://github.com/karlstratos/ammi).

## License

This code is offered under the [MIT License](https://opensource.org/licenses/MIT).