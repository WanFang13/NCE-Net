
import torch


def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            results = models['backbone'](inputs)
            _, preds = torch.max(results[0].data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


