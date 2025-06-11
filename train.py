from core.special import SScalar

def train(model, X_scalar, y_scalar, loss_fn, optimizer, epochs=150):
    print(f"Training model for {epochs} epochs...")
    losses = []

    for epoch in range(epochs):
        total_loss = SScalar(0.0)
        
        for x_batch, y_true in zip(X_scalar, y_scalar):
            logits = model(x_batch)
            y_pred = logits.sigmoid()
            loss = loss_fn(y_pred, y_true)
            total_loss += loss
            
            loss.backward()
        
        optimizer.step()
        model.zero_grad()
        
        avg_loss = total_loss.data / len(X_scalar)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3}/{epochs} | Loss: {avg_loss:.4f}')

    print(f'Final Loss: {losses[-1]:.4f}')

    return losses
