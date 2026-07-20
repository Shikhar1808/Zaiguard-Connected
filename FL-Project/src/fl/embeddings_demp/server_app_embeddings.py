"""fl_project: A Flower / PyTorch app for Zaiguard Embedding Classification."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.fl.tasks.classification_embeddings import AlertImportanceClassifier
from src.fl.tasks.classification_embeddings import load_embedding_data as load_data
from src.fl.tasks.classification_embeddings import test as test_fn

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)
    num_rounds: int = context.run_config.get("num-server-rounds", 3)
    lr: float = context.run_config.get("learning-rate", 0.01)

    # Load global model
    global_model = AlertImportanceClassifier()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config.get("save-model", True):
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_alert_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = AlertImportanceClassifier()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set (Simulating centralized test set with partition ID 999)
    _, test_dataloader = load_data(partition_id=999, num_partitions=1, batch_size=32, num_samples=1000)

    # Evaluate the global model on the test set
    test_loss, test_acc = test_fn(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
