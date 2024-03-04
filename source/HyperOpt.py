import torch
import torch.nn.functional as F
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch


class HyperOptmisation():
    def __init__(self, config, model):
        self.config = config
        self.model = model


    def create_objective():  # ①
        #train_loader, test_loader = load_data()  # Load some data
        #model = ConvNet().to("cpu")  # Create a PyTorch conv net
        optimizer = torch.optim.SGD(  # Tune the optimizer
            model.parameters(), lr=config["lr"], momentum=config["momentum"]
        )

        while True:
            train(model, optimizer, train_loader)  # Train the model
            acc = test(model, test_loader)  # Compute test accuracy
            train.report({"mean_accuracy": acc})  # Report to Tune

        search_space = {"lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.1, 0.9)}
        algo = OptunaSearch()

        tuner = tune.Tuner(objective, 
                tune_config=tune.TuneConfig(metric="mean_accuracy", mode="max", search_alg=algo),
                    run_config=train.RunConfig(stop={"training_iteration": 5}),
                    param_space=search_space,
                )
        results = tuner.fit()
        print("Best config is:", results.get_best_result().config)
