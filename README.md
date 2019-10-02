# Triplet loss in TensorFlow with three Strategies

This repository contains a triplet loss implementation with TensorFlow Estimator with online triplet mining using three different methods i.e Hard, Semi Hard and All. This repository also contains code to generate sprite image of labels to visualize on tensorflow projector. Please check [`assets`](assets) for loss png and projector video.


## Requirements


Install `gpu` requirements:
```bash
pip install -r requirements.txt
```

The interesting part, defining triplet loss with triplet mining can be found in [`model/triplet_loss.py`](model/triplet_loss.py).

## Training on FMNIST

When you run training script, it will download fashion mnist data itself and will place in [`data/fmnist`](data/fmnist)
To run a new experiment called `new_model`, do:
```bash
python train.py --model_dir ./experiments/new_model
```

To evaluate run:
```bash
python evaluate.py --model_dir ./experiments/new_model
```

You will first need to create a configuration file like this one: [`params.json`](experiments/batch_all/params.json).
This json file specifies all the hyperparameters for the model.
All the weights and summaries will be saved in the `model_dir`.

Once trained, you can visualize the embeddings by running:
```bash
python visualize_embeddings.py --model_dir ./experiments/new_model
```

And run tensorboard in the experiment directory:
```bash
tensorboard --logdir=./experiments/new_model/tf_projector
```

##TODO 
- Train all different combinations of models and triplet stategies.
- Tune hyperparameters as lenet loss function did not converge. 

## Resources

- Source code for the built-in TensorFlow function for semi hard online mining triplet loss: [`tf.contrib.losses.metric_learning.triplet_semihard_loss`][tf-triplet-loss].
- Tensorflow Estimators [`link`][link]
- [Facenet paper][facenet] introducing online triplet mining
- Detailed explanation of online triplet mining in [*In Defense of the Triplet Loss for Person Re-Identification*][in-defense]
- Some code structure is taken from this [`blog`][blog]

[link]: https://github.com/guillaumegenthial/tf-estimator-basics
[facenet]: https://arxiv.org/abs/1503.03832
[in-defense]: https://arxiv.org/abs/1703.07737
[tf-triplet-loss]: https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss
[blog]: https://omoindrot.github.io/triplet-loss
