# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/7/4 20:40
import torch
import time
def train_on_batch(num_epochs, train_iter, test_iter, optimizer, criterion, net, device, lr_jitter=False):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_iter), eta_min=5e-6)
    for epoch in range(num_epochs):
        # training
        net.train()
        sum_loss = 0.0
        sum_acc = 0.0
        for (X, y) in train_iter:
            X = X.type(torch.FloatTensor)
            y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss = criterion(y_hat, y).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_jitter:
                scheduler.step()
            sum_loss += loss.item() / y.shape[0]
            sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()
        train_loss = sum_loss / len(train_iter)
        train_acc = sum_acc / len(train_iter)

        # test
        if epoch == num_epochs - 1:
            net.eval()
            sum_acc = 0.0
            for (X, y) in test_iter:
                X = X.type(torch.FloatTensor)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64)
                X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()
            val_acc = sum_acc / len(test_iter)
        print(f"epoch{epoch + 1}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}")
    print(
        f'training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc={val_acc:.3f}')
    torch.cuda.empty_cache()
    return val_acc.cpu().data
