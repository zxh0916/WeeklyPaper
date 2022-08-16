from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils import data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_acc(output, target, criterion):
    pred = output.argmax(dim=1)
    acc = ((pred == target).sum() / target.numel()).item()
    loss = criterion(output, target)
    return loss, acc

def lr_ratio(emb_dim, warmup_steps, cur_step):
    if cur_step == 0:
        return 0
    lr = emb_dim ** -0.5
    lr *= min(cur_step ** -0.5, (cur_step * warmup_steps ** -1.5))
    return lr

def get_lr(optimizer):
    return (optimizer.state_dict()['param_groups'][0]['lr'])

def train_ViT(net, train_dataset, valid_dataset):
    cfg = net.cfg
    with open('configs.txt', 'a') as f:
        f.write('{\n')
        for k, v in net.cfg.__dict__.items():
            f.write(k + ': ' + str(v) + "\n")
        f.write('}\n')
    train_dataloader = data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    writer = SummaryWriter(f"runs/ViT_CIFAR_{cfg.version}")
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_lr = lambda cur_step: lr_ratio(net.emb_dim, cfg.warmup_steps, cur_step)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    criterion = torch.nn.CrossEntropyLoss()
    global_step = 0
    for epoch in range(cfg.num_epochs):
        net.train()
        train_loss, train_acc = [], []
        for input, target in train_dataloader:
            global_step += 1
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(input)
            loss, acc = loss_acc(output, target, criterion)
            train_loss.append(loss)
            train_acc.append(acc)
            loss.backward()
            writer.add_scalar('learning rate', get_lr(optimizer), global_step=global_step)
            optimizer.step()
            scheduler.step()
            writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            writer.add_scalar('train/accuracy', acc, global_step=global_step)
        with torch.no_grad():
            net.eval()
            valid_loss, valid_acc = [], []
            for input, target in valid_dataloader:
                input, target = input.to(device), target.to(device)
                output = net(input)
                loss, acc = loss_acc(output, target, criterion)
                valid_loss.append(loss.item())
                valid_acc.append(acc)
            writer.add_scalar('valid/loss', sum(valid_loss) / len(valid_loss), global_step=global_step)
            writer.add_scalar('valid/accuracy', sum(valid_acc) / len(valid_acc), global_step=global_step)
        message = list(map(lambda x:sum(x) / len(x), (train_loss, train_acc, valid_loss, valid_acc)))
        print(f'epoch {epoch+1:3d}, train loss: {message[0]:8.4f}, train accuracy: {message[1]:8.4f}, valid loss: {message[2]:8.4f}, valid accuracy: {message[3]:8.4f}')
        torch.save(net.state_dict(), f'ViT_CIFAR_{cfg.version}')