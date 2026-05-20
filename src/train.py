from src.utils import DiceLoss
import config as c

def train_model(model, train_loader, optimizer, criterion):
    for epoch in range(c.NUM_EPOCHS):
        total_loss = 0.0

        for batch in train_loader:
            #Set the gradients to zero
            optimizer.zero_grad()

            #1. Forward pass
            feature, target = batch
            prediction = model(feature)

            #2. Calculate Loss
            loss = criterion(prediction, target)
            total_loss += loss.item()

            #3. Backward pass
            loss.backward()

            #4. Update weights
            optimizer.step()

        #end loop
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} / {c.NUM_EPOCHS} - Loss: {avg_loss:.4f}")

    #end loop
#end function