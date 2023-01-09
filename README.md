# Unsupervised learning

Note: You can see the README of the subdirectories for more details about them.

## Prerequisites

- Install [Python 3.11](https://www.python.org/downloads/release/python-3111/)
- Install [Pipenv](https://pipenv.pypa.io/): `pip install pipenv`

## Installation

Install the dependencies by doing:
```sh
pipenv install --dev
```

## Usage

Run a script by doing:
```sh
pipenv run <script>
```

### Example of workflow execution

1. Preprocess the raw data:
    ```sh
    pipenv run preprocess
    ```
2. Normalize the preprocessed data:
    ```sh
    pipenv run normalize
    ```
3. Run and plot the algorithms:
    ```sh
    pipenv run pca
    pipenv run tsne
    pipenv run kmeans
    pipenv run upgma
    pipenv run cl
    ```
4. Find the best parameters for the SOM by running the analysis scripts:
    ```sh
    pipenv run analysis
    ```
5. Run and plot the SOM with the best parameters according to the plots in results:
    ```sh
    pipenv run som <vertical> <horizontal> <dataset> <lr> <sigma> <classname> <epochs>
    ```
    You can also do `pipenv run som --help` to see the available parameters.

## Credits

Romain Monier [ [GitHub](https://github.com/rmonier) ] – Co-developer

Marlon Funk [ [GitHub](https://github.com/MarlonFunk) ] – Co-developer

## Contact

Project Link: https://github.com/rmonier/unsupervised-learning
