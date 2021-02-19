# Explaining Text Classifier

The code for our AAAI 2019 paper:  Interpreting Deep Models for Text Analysis via Optimization and Regularization Methods [[paper link]](https://ojs.aaai.org//index.php/AAAI/article/view/4517)

## How to use our code

First, train a text classifier using the train function in ``model.py''. 

Then, use the test function in ``model.py'' to load the model and explain the decision for test data. 

You need to modify the data path in data reader, word2vec path in helper, and other settings in the model.  

Reference
---------

    @inproceedings{yuan2019interpreting,
      title={Interpreting deep models for text analysis via optimization and regularization methods},
      author={Yuan, Hao and Chen, Yongjun and Hu, Xia and Ji, Shuiwang},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={33},
      number={01},
      pages={5717--5724},
      year={2019}
    }



