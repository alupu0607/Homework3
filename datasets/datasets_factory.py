import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
from pathlib import Path
import hashlib
import pickle


class CachedDataset(Dataset):
    def __init__(self, dataset, cache_dir="./cache", transform=None):
        self.dataset = dataset
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, index):
        sample_hash = hashlib.md5(str(index).encode()).hexdigest()
        return self.cache_dir / f"{sample_hash}.pkl"

    def __getitem__(self, index):
        cache_path = self._get_cache_path(index)

        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                sample = pickle.load(f)
        else:
            sample = self.dataset[index]
            with open(cache_path, 'wb') as f:
                pickle.dump(sample, f)

        if self.transform and isinstance(sample, (tuple, list)) and len(sample) == 2:
            data, label = sample
            if self.transform:
                data = self.transform(data)
            sample = (data, label)
        elif self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.dataset)

def get_dataset(name, batch_size, shuffle_train, shuffle_test, pin_memory=False, data_dir='./data', cache_dir='./cache', validation_split=0.1, augmentation=None):
    augmentation_schemes = {
        'none': transforms.Compose([]),
        'basic': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]),
        'advanced': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
        ]),
    }
    if name == 'MNIST':
        train_set = datasets.MNIST(root=data_dir, train=True, download=True)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True)
    elif name == 'CIFAR10':
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True)
    elif name == 'CIFAR100':
        train_set = datasets.CIFAR100(root=data_dir, train=True, download=True)
        test_set = datasets.CIFAR100(root=data_dir, train=False, download=True)
    else:
        raise ValueError(f"Dataset '{name}' is not supported.")
    if name == 'MNIST':
        if augmentation and augmentation in augmentation_schemes:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                augmentation_schemes[augmentation]
            ])
            print('Applied augmentation')
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            print("Did not apply augmentation")

    elif name in ['CIFAR10', 'CIFAR100']:
        if augmentation and augmentation in augmentation_schemes:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                augmentation_schemes[augmentation]
            ])
            print('Applied augmentation')
        else:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            print("Did not apply augmentation")


    full_train_set = CachedDataset(train_set, cache_dir=cache_dir, transform=transform)
    test_set = CachedDataset(test_set, cache_dir=cache_dir, transform=transform)

    num_validation_samples = int(validation_split * len(full_train_set))
    num_training_samples = len(full_train_set) - num_validation_samples

    train_set, val_set = random_split(full_train_set, [num_training_samples, num_validation_samples])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle_test, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
