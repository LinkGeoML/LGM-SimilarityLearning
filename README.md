# LGM-SimilarityLearning

We will use Pipfile as a symlink from the Pipfile.cpu or Pipfile.gpu depending
on the training


- Install Pipenv
- Navigate into the main directory
- Run `pipenv install` to install all the required packages needed for the experimentation
- In order to load the local env just run `pipenv shell`

### CLI
One can use the CLI to preprocess the original dataset or train a model based
on some settings.

In order to check the CLI command run the following:
```python -m similarity_learning.cli --help```


To preprocess the original dataset, one may run the following:

- Check the parameters
```python -m similarity_learning.cli dataset --help```

- Run the default parameters
```python -m similarity_learning.cli dataset```
