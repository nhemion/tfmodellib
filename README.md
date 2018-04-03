TFModelLib
==========

A collection of neural network models implemented in Tensorflow. Much of the
usual overhead connected to implementing a model in Tensorflow (checkpoints,
summaries) is automated by a simple abstract class, `TFModel`, from which all
other models in the library inherit.

List of implemented models:

- [```tfmodellib/linreg.py```](tfmodellib/linreg.py): Linear regression model
- [```tfmodellib/mlp.py```](tfmodellib/mlp.py): Multilayer Perceptron
- [```tfmodellib/autoencoder.py```](tfmodellib/autoencoder.py): Autoencoder (fully connected)
- [```tfmodellib/cae2d.py```](tfmodellib/cae2d.py): Convolutional autoencoder for 2-dimensional inputs (e.g. images)
- [```tfmodellib/vae.py```](tfmodellib/vae.py): Variational autoencoder (fully connected)

A class inheriting from TFModel has to implement the following member functions:

- **build_graph**

    Defines the compute graph

- **run_update_and_loss**

    Updates the model parameters (runs the model's optimizer operation) and
    returns the loss

- **run_loss**

    Computes the loss

- **run_output**

    Computes the output

See `tfmodellib/linreg.py` for an example of how to implement a model based on
the TFModel class.
