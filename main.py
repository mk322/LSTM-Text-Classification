import os
from time import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from LSTM import LSTMClassifier
from sklearn.metrics import f1_score

TEXT, vocab_size, word_embeddings, train_iter, valid_iter = load_data.load_dataset()

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_epoch_f1 = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] != 64):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        pred = torch.max(prediction, 1)[1].view(target.size()).data
        num_corrects = (pred == target.data).float().sum()
        f1 = f1_score(target.data.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='macro')
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%, Training F1: {f1: .2f}')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        total_epoch_f1 += f1
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter), total_epoch_f1/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_epoch_f1 = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] != 64):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)

            pred = torch.max(prediction, 1)[1].view(target.size()).data
            num_corrects = (pred == target.data).float().sum()
            f1 = f1_score(target.data.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='macro')
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_f1 += f1
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter), total_epoch_f1/len(val_iter)
	

learning_rate = 0.001
batch_size = 64
output_size = 2
hidden_size = 256
embedding_length = 300
max_epochs = 10
loss_fn = F.cross_entropy

model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)


best_val_acc = 0
train_acc_l = []
train_f1_l = []
val_acc_l = []
val_f1_l = []
training_time = []
for epoch in range(max_epochs):
    start = time()
    train_loss, train_acc, train_f1 = train_model(model, train_iter, epoch)
    used_time = time() - start
    training_time.append(round(used_time,2))
    val_loss, val_acc, valid_f1 = eval_model(model, valid_iter)
    train_acc_l.append(round(train_acc,2))
    train_f1_l.append(round(train_f1,2))
    val_acc_l.append(round(val_acc,2))
    val_f1_l.append(round(valid_f1,2))
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.2f}, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%, Val. F1: {valid_f1:.2f}')
    if best_val_acc < val_acc:
        best_val_acc = val_acc
    else:
        break

train_loss, train_acc, train_f1 = eval_model(model, train_iter)
vali_loss, vali_acc, vali_f1 = eval_model(model, valid_iter)    
#test_loss, test_acc, test_f1 = eval_model(model, test_iter)
#print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.3f}')

print(f'Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.3f}')
print(f'Valid Acc: {vali_acc:.2f}%, Valid F1: {vali_f1:.3f}')


print('Training Accuracy: ', train_acc_l)
print('Training F1: ', train_f1_l)
print('Validation Accuracy: ', val_acc_l)
print('Validation F1: ', val_f1_l)
print('Training Time: ', training_time)



"""
''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

test_sen1 = TEXT.preprocess(test_sen1)
test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

test_sen2 = TEXT.preprocess(test_sen2)
test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

test_sen = np.asarray(test_sen1)
test_sen = torch.LongTensor(test_sen)
with torch.no_grad():   
    test_tensor = Variable(test_sen)
test_tensor = test_tensor.cuba()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
if (torch.argmax(out[0]) == 1):
    print ("Sentiment: Positive")
else:
    print ("Sentiment: Negative")
"""