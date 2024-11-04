import torch
import yaml
import wandb
from datasets.datasets_factory import get_dataset
from models.model_factory import load_model

MODEL_OPTIONS = {
    'CIFAR10': ['resnet18_cifar10', 'PreActResNet18'],
    'CIFAR100': ['resnet18_cifar10', 'PreActResNet18'],
    'MNIST': ['MLP', 'LeNet']
}


def get_optimizer(optimizer_name, model_parameters, learning_rate=0.001, momentum=0.9, weight_decay=0.0):
    if optimizer_name == 'SGD':
        return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'SGD_Momentum':
        return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=momentum)
    elif optimizer_name == 'SGD_Nesterov':
        return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, nesterov=True)
    elif optimizer_name == 'Adam':
        return torch.optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == 'AdamW':
        return torch.optim.AdamW(model_parameters, lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        return torch.optim.RMSprop(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")


def get_scheduler(scheduler_name, optimizer, config):
    if scheduler_name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    elif scheduler_name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'],
                                                          patience=config['patience'])
    else:
        return None


def main(config):
    wandb.init(project=config['project_name'], config=config)
    wandb.run.name = config.get('run_name', 'Run')
    wandb.config.update(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataset(
        config['dataset'],
        config['batch_size'],
        config['shuffle_train'],
        config['shuffle_test'],
        config['pin_memory'],
        augmentation=config.get('augmentation')
    )

    if config['dataset'] not in MODEL_OPTIONS:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    if config['model'] not in MODEL_OPTIONS[config['dataset']]:
        raise ValueError(f"Invalid model '{config['model']}' for dataset '{config['dataset']}'. "
                         f"Valid models are: {MODEL_OPTIONS[config['dataset']]}")

    model = load_model(config['model'], num_classes=100 if config['dataset'] == 'CIFAR100' else 10)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(config['optimizer'], model.parameters(), learning_rate=config['learning_rate'])
    scheduler = get_scheduler(config.get('scheduler'), optimizer, config)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 5)

    for epoch in range(config['epochs']):
        print("Training....")
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        wandb.log(
            {"Training Loss": avg_loss, "Training Accuracy": train_accuracy})
        print(f'Epoch [{epoch + 1}/{config["epochs"]}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, val_predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (val_predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        wandb.log({"Validation Loss": val_loss,
                   "Validation Accuracy": val_accuracy})
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print("Validation loss improved...")
        else:
            patience_counter += 1
            print(f'No improvement in validation loss. Patience counter: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

    wandb.watch(model)


if __name__ == "__main__":
    with open(r'D:\Masters\1st_year\advanced_neural_networks\Homeowork3\config\config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    wandb.login(key="114fe47a63bcf889f132e3792ba5cbfb93f3471a")
    main(config)