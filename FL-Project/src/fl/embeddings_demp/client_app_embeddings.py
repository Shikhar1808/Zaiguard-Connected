"""fl_project: A Flower / PyTorch app for Zaiguard Embedding Classification."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.fl.tasks.classification_embeddings import AlertImportanceClassifier
from src.fl.tasks.classification_embeddings import load_embedding_data as load_data
from src.fl.tasks.classification_embeddings import test as test_fn
from src.fl.tasks.classification_embeddings import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local embedding data."""

    # Load the model and initialize it with the received weights
    model = AlertImportanceClassifier()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 1)
    batch_size = context.run_config.get("batch-size", 32)
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    lr = context.run_config.get("lr", 0.01)
    local_epochs = context.run_config.get("local-epochs", 5)
    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        lr,
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local embedding data."""

    # Load the model and initialize it with the received weights
    model = AlertImportanceClassifier()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 1)
    batch_size = context.run_config.get("batch-size", 32)
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
