from .logger import Logger


import torch
from torch import autograd
import torch.nn.functional as F


from tqdm import tqdm_notebook


def svm_l1loss(a, y, weight, C=1, batch_fraction=1):
    """
    Calculate SVM-L1 loss:
    l(w) = 0.5 * w^T w + C \sum_{i=1}^N max(1 - w^Tx_n y_n, 0)

    Parameters:
        a: torch.Tensor --- activation from previous layer, i.e. a = w^T x
        y: torch.Tensor --- vector of labels
        weight: torch.Tensor --- weight tensor of previous layer
        C: float --- regularization constant
        batch_fraction: float --- fraction of samples in mini-batch (compared to the whole sample)
    """
    relu = F.relu(1 - a * y)
    loss = 0.5 * batch_fraction * (weight * weight).sum() + C * relu.sum()
    return loss


def svm_l1loss2(prediction, target, weight, C=1, batch_fraction=1):
    """
    Calculate SVM-L1 loss:
    l(w) = 0.5 * w^T w + C \sum_{i=1}^N max(1 - w^Tx_n y_n, 0)

    Parameters:
        a: torch.Tensor --- activation from previous layer, i.e. a = w^T x
        y: torch.Tensor --- vector of labels
        weight: torch.Tensor --- weight tensor of previous layer
        C: float --- regularization constant
        batch_fraction: float --- fraction of samples in mini-batch (compared to the whole sample)
    """
    relu = F.relu(1 + prediction - prediction[target].view(-1, 1)).sum(dim=1) - 1
    loss = 0.5 * batch_fraction * (weight * weight).sum() + C * relu.sum()
    return loss


def svm_l2loss(a, y, weight, C=1, batch_fraction=1):
    """
    Calculate SVM-L2 loss:
    l(w) = 0.5 * w^T w + C \sum_{i=1}^N max(1 - w^Tx_n y_n, 0)^2

    Parameters:
        a: torch.Tensor --- activation from previous layer, i.e. a = w^T x
        y: torch.Tensor --- vector of labels
        weight: torch.Tensor --- weight tensor of previous layer
        C: float --- regularization constant
        batch_fraction: float --- fraction of samples in mini-batch (compared to the whole sample)
    """
    relu = F.relu(1 - a * y)**2
    return 0.5 * batch_fraction * (weight * weight).sum() + C * relu.sum()


def svm_l2loss2(prediction, target, weight, C=1, batch_fraction=1):
    """
    Calculate SVM-L1 loss:
    l(w) = 0.5 * w^T w + C \sum_{i=1}^N max(1 - w^Tx_n y_n, 0)

    Parameters:
        a: torch.Tensor --- activation from previous layer, i.e. a = w^T x
        y: torch.Tensor --- vector of labels
        weight: torch.Tensor --- weight tensor of previous layer
        C: float --- regularization constant
        batch_fraction: float --- fraction of samples in mini-batch (compared to the whole sample)
    """
    relu = (F.relu(1 + prediction - prediction[target].view(-1, 1))**2).sum(dim=1) - 1
    loss = 0.5 * batch_fraction * (weight * weight).sum() + C * relu.sum()
    return loss


def one_class_svmloss(prediction, weight, rho, batch_fraction=1, nu=1):
    relu = F.relu(rho - prediction) - rho
    return 0.5 * batch_fraction * (weight * weight).sum() + relu.mean() / nu


def evaluate(model, test_loader):
    test_accuracy = 0
    n_test_samples = 0
    model.eval()
    for test_batch_idx, (test_data, test_target) in enumerate(test_loader):
        test_data, test_target = test_data.cuda(), test_target.cuda()
        test_data = autograd.Variable(test_data, volatile=True)
        output = model(test_data).data

        _, argmax = output.max(1)
        test_accuracy += test_target.eq(argmax).sum()
        n_test_samples += test_target.size(0)

    test_accuracy /= n_test_samples
    return test_accuracy


def save(model, path):
    print('Saving..')
    state = model.state_dict()
    torch.save(state, '{}'.format(path))


# load model
def load_model(basic_model, path):
    checkpoint = torch.load('{}'.format(path))
    basic_model.load_state_dict(checkpoint)


def train(train_loader, test_loader, model, optimizer, loss_type='svml1loss',
          C=1, scheduler=None, start_epoch=0, stop_epoch=20, logdir='../logs'):
    """
    Possible losses: 'svml1loss', 'svml2loss', 'cross_entropy'
    """

    try:
        os.mkdir(logdir)
    except:
        pass

    logger = Logger(logdir)

    # get the last layer of the model
    last_module = list(model.modules())[-1]

    n_epochs = stop_epoch - start_epoch
    n_steps = 0
    for epoch in tqdm_notebook(range(start_epoch, stop_epoch), desc='epochs', total=n_epochs):

        # train
        model.train()
        train_loss = 0
        n_train_samples = 0
        n_train_batches = 0
        if scheduler is not None:
            scheduler.step(epoch=epoch)
        for data, target in tqdm_notebook(train_loader, leave=False):
            data, target = data.cuda(), target

            if loss_type == 'cross_entropy':
                target = target.cuda()
            elif loss_type == 'svml1loss2' or loss_type == 'svml2loss2':
                target_onehot = torch.zeros(target.size()[0], 10)
                target = target_onehot.scatter_(1, target.view(-1, 1), 1).byte().cuda()
            else:
                target_onehot = -torch.ones(target.size()[0], 10)
                target = target_onehot.scatter_(1, target.view(-1, 1), 1).cuda()

            data, target = autograd.Variable(data), autograd.Variable(target)

            optimizer.zero_grad()
            output = model(data)

            batch_fraction = len(data) / float(len(train_loader))

            if loss_type == 'svml1loss':
                loss = svm_l1loss(output, target, last_module.weight, C=C, batch_fraction=batch_fraction)
            elif loss_type == 'svml1loss2':
                loss = svm_l1loss2(output, target, last_module.weight, C=C, batch_fraction=batch_fraction)
            elif loss_type == 'svml2loss':
                loss = svm_l2loss(output, target, last_module.weight, C=C, batch_fraction=batch_fraction)
            elif loss_type == 'svml2loss2':
                loss = svm_l2loss2(output, target, last_module.weight, C=C, batch_fraction=batch_fraction)
            elif loss_type == 'cross_entropy':
                loss = F.cross_entropy(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            n_train_samples += target.size(0)
            n_train_batches += 1
            n_steps += 1

            # (2) Log values and gradients of the parameters (histogram)
            if n_train_batches % 100 == 99:
                for tag, value in model.named_parameters():
                    if 'weight' in tag:

                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                        if value.requires_grad:
                            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)


        train_loss /= n_train_batches

        logger.scalar_summary('loss', loss.data[0], n_steps)

        # evaluate
        test_accuracy = evaluate(model, test_loader)

        # print progress
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTest acc: {:.3f}'.format(
              epoch, n_train_samples, len(train_loader.dataset),
              100. * n_train_batches / len(train_loader), train_loss,
              test_accuracy))

        #============ TensorBoard logging ============#
        # (1) Log the scalar values
        info = {
            'accuracy': test_accuracy
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)
