from matplotlib.style import available
import numpy as np
from sklearn.metrics import precision_score, recall_score
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def train_epoch(model, loader, optimizer, args):
    """Train a given model on a given dataset using a given optimizer for one epoch."""
    model.train()
    losses = []
    for batch_id, (data, target) in enumerate(loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = F.cross_entropy(output, target)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = np.mean(losses)
    return train_loss


def evaluate(model, loader, args):
    output = []
    with torch.no_grad():
        for data in loader:
            if args.cuda:
                data = data.cuda()
            data = Variable(data)
            output.append(model(data))
    output = torch.stack(output).detach().cpu()
    output = torch.squeeze(output)
    return output


def accuracy(model, loader, args):
    """Evaluate the accuracy of a given model on a given dataset."""
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            losses.append(F.cross_entropy(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    eval_loss = np.mean(losses)
    return eval_loss, 100. * correct / len(loader.dataset)


def accuracy_by_class(model, loader, args):
    """
    Evaluate the class-specific accuracy of a given model on a given dataset.

    Returns:
        A 1-D numpy array of length L = num-classes, containg the accuracy for each class.
    """
    model.eval()
    correct = np.zeros(args.num_classes, dtype=np.int64)
    wrong = np.zeros(args.num_classes, dtype=np.int64)
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            preds = output.data.max(dim=1)[1].cpu().numpy().astype(np.int64)
            target = target.data.cpu().numpy().astype(np.int64)
            for label, pred in zip(target, preds):
                if label == pred:
                    correct[label] += 1
                else:
                    wrong[label] += 1
    assert correct.sum() + wrong.sum() == len(loader.dataset)
    return 100. * correct / (correct + wrong)


def evaluate_precision(model, loader, args):
    """Evaluate the precision of a given model on a given dataset."""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            predictions.append(output.data.max(1, keepdim=True)[1][0])
            targets.append(target)
    y_pred = torch.stack(predictions).detach().cpu()
    y_true = torch.stack(targets).detach().cpu()
    precision = precision_score(
        y_pred=y_pred,
        y_true=y_true,
        average='micro',
    )
    by_class = precision_score(
        y_pred=y_pred,
        y_true=y_true,
        average=None,
    )
    return precision, by_class


def evaluate_recall(model, loader, args):
    """Evaluate the recall of a given model on a given dataset."""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            predictions.append(output.data.max(1, keepdim=True)[1][0])
            targets.append(target)
    y_pred = torch.stack(predictions).detach().cpu()
    y_true = torch.stack(targets).detach().cpu()
    recall = recall_score(
        y_pred=y_pred,
        y_true=y_true,
        average='micro',
    )
    by_class = recall_score(
        y_pred=y_pred,
        y_true=y_true,
        average=None,
    )
    return recall, by_class
