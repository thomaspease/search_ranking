import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torch

def encode_categories(df):
  df.category = df.category.apply(lambda categories: categories.split(' /')[0])
  cat_list = list(enumerate(df.category.unique()))
  dict = {}
  for index, category in cat_list:
    dict[category] = index
  df.category = df.category.apply(lambda category: dict.get(category))
  return df

class ImageDataset(Dataset):
  def __init__(self, transform_image):
    self.product_frame = encode_categories(pd.read_csv('./data/fb/cleaned_products.csv', lineterminator="\n"))
    self.image_map_df = pd.read_csv('./data/fb/images.csv')
    self.root_dir = './images'
    self.transform_image = transform_image

  def __getitem__(self, index): 
    img_id = self.image_map_df.iloc[index, 1]
    img_name = f'./images/{img_id}.jpg'
    im = Image.open(img_name)
    img = im.convert('RGB')
    
    # Match product ID from the image map csv, to the product ID in the products csv
    img_row = self.product_frame.loc[self.product_frame.id == self.image_map_df.iloc[index, 2]]
    category = img_row.iloc[0, 3]

    if self.transform_image:
      img = self.transform_image(img)

    item = {'image': img, 'category': category}
    
    return item

  def __len__(self):
    return len(self.product_frame)
  
transform = transforms.Compose(
  [transforms.ToTensor(),
  transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0)),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

dataset = ImageDataset(transform_image=transform)
img_datasets = {}
img_datasets['train'], img_datasets['val'], img_datasets['test']= random_split(dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))

dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val', 'test']}

data_loaders = {}
data_loaders['train'] = DataLoader(img_datasets['train'], batch_size=8, shuffle=True)
data_loaders['val'] = DataLoader(img_datasets['val'], batch_size=8, shuffle=True)
data_loaders['test'] = DataLoader(img_datasets['test'], batch_size=8, shuffle=True)