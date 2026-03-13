# test_imagefolder.py
import torchvision.datasets as datasets
import torchvision.transforms as transforms

t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

train = datasets.ImageFolder('/home/developer/workspace/AntonioWork/old_versions/not_ok/data/imagenet_extracted/train', transform=t)
val   = datasets.ImageFolder('/home/developer/workspace/AntonioWork/old_versions/not_ok/data/imagenet_extracted/validation', transform=t)

print(f"Train classes: {len(train.classes)}, samples: {len(train)}")
print(f"Val classes:   {len(val.classes)}, samples: {len(val)}")
print(f"Class indices match: {train.classes == val.classes}")