# copyright 2024 moshe sipper
# www.moshesipper.com

import os
import argparse
import builtins
import random
import torch
from string import ascii_lowercase
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import CocoCaptions, CocoDetection
from torch.utils.data import DataLoader, Dataset

from coconames import COCO_CLASS_NAMES
from buildvocab import build_vocab

NUM_CLASSES = len(COCO_CLASS_NAMES) - 1
VOCAB_SIZE = 1000
TOP_C = 3 # number of top classes to output when predicting
TOP_W = 10 # number of top words to output when predicting 
HEIGHT, WIDTH = 400, 400
PRETRAINED = True # use pretrained weights or not
NUM_EPOCHS = 20
BATCH_SIZE = 32

RESDIR = './results'
PATH2TRAIN = '../datasets/coco/train2017'
PATH2TRAIN_INSTANCES = '../datasets/coco/annotations/instances_train2017.json'
PATH2TRAIN_CAPTIONS = '../datasets/coco/annotations/captions_train2017.json'
PATH2VAL = '../datasets/coco/val2017'
PATH2VAL_INSTANCES = '../datasets/coco/annotations/instances_val2017.json'
PATH2VAL_CAPTIONS = '../datasets/coco/annotations/captions_val2017.json'

vocab = build_vocab(PATH2TRAIN_CAPTIONS, vocab_size=VOCAB_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def rndstr(n=6):
    return ''.join(random.choices(ascii_lowercase, k=n))


def print(*args, **kwargs):
    builtins.print(*args, **kwargs, flush=True)


rn = rndstr()
print(rn)
os.makedirs(f'{RESDIR}/{rn}', exist_ok=True)
outdir = f'{RESDIR}/{rn}'
outfile = f'{outdir}/run.out'


parser = argparse.ArgumentParser()
parser.add_argument('-model', dest='modelname', type=str, action='store', default='resnet',
                    help='Face recognition model: resnet, mobilenet, regnet (default: resnet)')
args = parser.parse_args()
modelname = args.modelname
mods = ['resnet', 'mobilenet', 'regnet', 'convnext', 'swin', 'vit']
assert modelname in mods, f'{modelname}'
strmodel = f'{modelname},'.ljust(1+max([len(m) for m in mods])) # for printing
if modelname == 'vit':
    HEIGHT, WIDTH = 224, 224
elif modelname == 'swin':
    BATCH_SIZE = 8
    

assert device == torch.device('cuda'), f'{device}, {modelname}'

with open(outfile, 'a') as f:
    print('modelname', modelname, file=f)
    print('PRETRAINED', PRETRAINED, file=f)
    print('NUM_CLASSES', NUM_CLASSES, file=f)
    print('VOCAB_SIZE', VOCAB_SIZE, file=f)
    print('TOP_C', TOP_C, file=f)
    print('TOP_W', TOP_W, file=f)
    print('NUM_EPOCHS', NUM_EPOCHS, file=f)
    print('BATCH_SIZE', BATCH_SIZE, file=f)
    print('HEIGHT, WIDTH', HEIGHT, WIDTH, file=f)
    print('device', device, file=f)
    print('vocab', vocab, file=f)
    print('', file=f)


# Define a custom dataset class for multilabel (multi-word) classification
class CocoMultiWordDataset(Dataset):
    def __init__(self, root, instFile, capFile, transform=None, vocab=None):
        self.caption_dataset = CocoCaptions(root=root, annFile=capFile, transform=transform)
        self.classification_dataset = CocoDetection(root=root, annFile=instFile, transform=transform)
        self.vocab = vocab
    
    def __len__(self):
        return len(self.caption_dataset)
    
    def __getitem__(self, idx):
        images, captions = self.caption_dataset[idx]
        images_classification, annotations = self.classification_dataset[idx]

        # Convert captions to a multilabel format
        caps1 = ''.join(captions).replace('.',' ').replace(',',' ').lower() # 5 captions into 1 string
        ws = list(set(self.vocab).intersection(caps1.split())) # which caption words are in vocab
        labels = [1 if word in ws else 0 for word in self.vocab] # convert to labels        
        words = torch.tensor(labels, dtype=torch.float32) # convert labels to tensor

        # Get all class IDs present in the image        
        class_ids = [c['category_id'] - 1 for c in annotations] # classes start at 1 -- thus, minus 1
        classes = [1 if c in class_ids else 0 for c in range(NUM_CLASSES)]
        classes = torch.tensor(classes, dtype=torch.float32)

        return images, images_classification, words, classes


# Define a transform for both classification and captioning
transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
])

# Initialize the multiword dataset
train_dataset = CocoMultiWordDataset(root=PATH2TRAIN, instFile=PATH2TRAIN_INSTANCES,
                                     capFile=PATH2TRAIN_CAPTIONS, transform=transform, vocab=vocab)
val_dataset = CocoMultiWordDataset(root=PATH2VAL, instFile=PATH2VAL_INSTANCES,
                                   capFile=PATH2VAL_CAPTIONS, transform=transform, vocab=vocab)

# Initialize DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Define a shared backbone
if modelname == 'resnet':
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if PRETRAINED else None)
    backbone.fc = nn.Identity() # Remove fully connected layer for feature extraction
elif modelname == 'mobilenet':
    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if PRETRAINED else None)
    backbone.classifier[3] = nn.Identity() # Remove fully connected layer for feature extraction
elif modelname == 'regnet':
    backbone = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT if PRETRAINED else None)
    backbone.fc = nn.Identity() # Remove fully connected layer for feature extraction
elif modelname == 'convnext':
    backbone = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT if PRETRAINED else None)
    backbone.classifier[2] = nn.Identity() # Remove fully connected layer for feature extraction
elif modelname == 'swin':
    backbone = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT if PRETRAINED else None)
    backbone.head = nn.Identity() # Remove fully connected layer for feature extraction
elif modelname == 'vit':
    backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if PRETRAINED else None)
    backbone.heads.head = nn.Identity() # Remove fully connected layer for feature extraction
else:
    exit(f'Unknown model: {modelname}')


# Combine everything into a single model
class CombinedModel(nn.Module):
    def __init__(self, backbone):
        super(CombinedModel, self).__init__()
        self.backbone = backbone

        # Pass a sample through the backbone to get the number of output features
        with torch.no_grad():
            backbone_output = self.backbone(torch.randn(1, 3, HEIGHT, WIDTH))

        # Define separate heads for image classification and caption classification
        self.classification_head = nn.Linear(backbone_output.size(1), NUM_CLASSES)  
        self.caption_head = nn.Linear(backbone_output.size(1), VOCAB_SIZE)  

    def forward(self, images):
        features = self.backbone(images)
        classification_output = self.classification_head(features)
        caption_output = self.caption_head(features)
        return classification_output, caption_output


# Initialize combined model
combined_model = CombinedModel(backbone)
combined_model.to(device)

# Define optimizer
optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)


# Compute binary cross entropy using only top-k logits
def topk_binary_cross_entropy(logits, labels, topk=5):
    _, topk_indices = torch.topk(logits, k=topk)
    topk_mask = torch.zeros_like(logits)
    topk_mask.scatter_(1, topk_indices, 1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float() * topk_mask)
    return loss


def topk_indices(tensr, k):
    _, topk = torch.topk(tensr, k=k)
    topk = list(topk.detach().cpu().numpy())
    return topk


def topk_accuracy(classification_output, caption_output, words, classes):
    # compute accuracy over multilabel classification and words using topk prediction
    # returns a sum of accuracies, to be accumulutaed by calling function
    assert classification_output.size(0) == caption_output.size(0) == words.size(0) == classes.size(0)
    bsize = classification_output.size(0)

    acc_classification, acc_caption = 0.0, 0.0
    for idx in range(bsize): # loop over classification and caption outputs in batch
        # top classification
        topc = topk_indices(classification_output[idx], k=TOP_C)
        correct = sum(1 for i in topc if classes[idx][i] == 1) / TOP_C
        acc_classification += correct

        # top caption words
        topw = topk_indices(caption_output[idx], k=TOP_W)
        correct = sum(1 for i in topw if words[idx][i] == 1) / TOP_W
        acc_caption += correct
    
    return acc_classification, acc_caption


# Run a single epoch of training or validation
def run_epoch(epoch, model, training=True):
    loader = train_loader if training else val_loader
    torch.cuda.empty_cache()
    model.train(training)
    _ = torch.set_grad_enabled(training)

    num_pics = 0
    total_loss_classification = 0.0
    total_loss_caption = 0.0
    total_loss = 0.0
    total_acc_classification = 0.0
    total_acc_caption = 0.0

    for idx, (images, images_classification, words, classes) in enumerate(loader):
        images, images_classification, words, classes =\
            images.to(device), images_classification.to(device), words.to(device), classes.to(device)
        num_pics += images.size(0)
        
        classification_output, caption_output = combined_model(images) # Forward pass

        if training:
            optimizer.zero_grad()

        # Classification Task
        # loss_classification = topk_binary_cross_entropy(classification_output, classes, topk=TOP_C)
        loss_classification = torch.nn.functional.binary_cross_entropy_with_logits(classification_output, classes)
        total_loss_classification += loss_classification.item()

        # Caption (words) Classification Task
        # loss_caption = topk_binary_cross_entropy(caption_output, words, topk=TOP_W)
        loss_caption = torch.nn.functional.binary_cross_entropy_with_logits(caption_output, words)
        total_loss_caption += loss_caption.item()

        # Total loss
        loss = loss_classification + loss_caption
        total_loss += loss.item()

        # Accumulate accuracy values
        acc_classification, acc_caption = topk_accuracy(classification_output, caption_output, words, classes)
        total_acc_classification += acc_classification
        total_acc_caption += acc_caption

        if idx % 100 == 0:
            with open(outfile, 'a') as f:
                print(f"{strmodel} epoch {epoch:2}, pics {f'{idx*BATCH_SIZE},'.ljust(7)} {'train,' if training else 'val,  '} loss: {total_loss/(idx+1):.3f}, cls: {total_loss_classification/(idx+1):.3f}, cap: {total_loss_caption/(idx+1):.3f}", file=f)

        # debug only
        # if idx == 10:
        #     break

        # Backward pass and optimization
        if training:
            loss.backward()
            optimizer.step()

        # del images, images_classification, words, classes  # maybe helps save GPU memory...
        torch.cuda.empty_cache()

    # Averages
    avg_total_loss = total_loss / len(loader)
    avg_total_loss_classification = total_loss_classification / len(loader)
    avg_total_loss_caption = total_loss_caption / len(loader)
    
    avg_acc_classification = total_acc_classification / num_pics
    avg_acc_caption = total_acc_caption / num_pics
    avg_acc = (avg_acc_classification + avg_acc_caption) / 2

    return avg_total_loss, avg_total_loss_classification, avg_total_loss_caption,\
        avg_acc, avg_acc_classification, avg_acc_caption


def run_sample_val_image(epoch):
    for images, images_classification, words, classes in val_loader:
        images, images_classification, words, classes =\
            images.to(device), images_classification.to(device), words.to(device), classes.to(device)
        break
    
    classification_output, caption_output = combined_model(images) # Forward pass

    transform = transforms.ToPILImage()
    img = transform(images[0])
    img.save(f'{outdir}/{epoch}.jpg')

    topc = topk_indices(classification_output[0], k=TOP_C)
    out_classes = [COCO_CLASS_NAMES[i+1] for i in topc]  # network outputs start at 0, classes start at 1

    topw = topk_indices(caption_output[0], k=TOP_W)
    out_words = [vocab[i] for i in topw] 

    with open(f'{outdir}/{epoch}.txt', 'w') as f:
        f.write('classes\n' + ', '.join(out_classes) + '\n\n') 
        f.write('words\n' + ', '.join(out_words) + '\n\n')


# Training (and validation)
for epoch in range(1, NUM_EPOCHS+1):
    for training in [True,False]: # train (True), validate (False)
        avg_total_loss, avg_total_loss_classification, avg_total_loss_caption,\
        avg_acc, avg_acc_classification, avg_acc_caption =\
            run_epoch(epoch, combined_model, training=training)
        
        trvl = 'Train,' if training else 'Val,  '
        with open(outfile, 'a') as f:
            print(f"{strmodel} Epoch {epoch:2} avg, {trvl} loss: {avg_total_loss:.3f}, cls: {avg_total_loss_classification:.3f}, cap: {avg_total_loss_caption:.3f}", file=f)
            
            print(f"{strmodel} Epoch {epoch:2} avg, {trvl} acc: {avg_acc:.3f}, cls: {avg_acc_classification:.3f}, cap: {avg_acc_caption:.3f}", file=f)

    run_sample_val_image(epoch=epoch)
       

    # Save the trained model each epoch
    torch.save(combined_model.state_dict(), f'{outdir}/model.pth')
