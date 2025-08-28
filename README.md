# data-science-katas

## Repo of katas for several data science topics and random exercises I stored on my computer for years

In order to work with the virtual environment created with Poetry and the jupyter notebooks follow the next steps:


### 1. Create and Activate the Poetry Environment
If you haven’t initialized the environment yet, run:

```sh
poetry install  # Create the environment and install dependencies
poetry shell
```

To verify the environment path:

```sh
poetry env info --path
```

### 2. Add the Kernel to Jupyter

```sh
poetry run python -m ipykernel install --user --name=myenv --display-name"name-to-display"
```
Replace `name-to-display` with a name that is easy to differentiate from the other environemts you work with

###  3. Select the Kernel in VSCode
1. Open any notebook you want to work in
2. Click on top right Kernel selector
3. Find the `name-to-display` that you previously decided


## Instructions on how to run server for SQL problems

### 1. Open the docker-compose.yaml

   ⚠️ Before moving forwards check the image version that better suits you  

### 2. Click the button to run the service


