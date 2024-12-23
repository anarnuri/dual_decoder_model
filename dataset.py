import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import glob
import cv2

class VAEDataset(Dataset):
    def __init__(self, num_classes: int, max_mech_size: int) -> None:
        self.imgs_path = os.path.join('/gpfs/scratch/anurizada/bar_linkages/', '')
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        self.num_classes = num_classes
        self.folder_dict = {'RRRR': 0, 'Steph1T1': 1, 'Steph1T2': 2, 'Steph1T3': 3, 'Steph2T1A1': 4, 'Steph2T1A2': 5, 'Steph2T2A1': 6, 'Steph2T2A2': 7, 'Steph3T1A1': 8, 'Steph3T1A2': 9, 'Steph3T2A1': 10, 'Steph3T2A2': 11, 'Type811-0': 12, 'Type811-1': 13, 'Type811-2': 14, 'Type811-3': 15, 'Type811-4': 16, 'Type812-0': 17, 'Type812-1': 18, 'Type812-2': 19, 'Type812-3': 20, 'Type812-4': 21, 'Type812-5': 22, 'Type812-6': 23, 'Type812-7': 24, 'Type813-0': 25, 'Type813-1': 26, 'Type813-2': 27, 'Type814-0': 28, 'Type814-1': 29, 'Type814-10': 30, 'Type814-11': 31, 'Type814-12': 32, 'Type814-13': 33, 'Type814-14': 34, 'Type814-15': 35, 'Type814-16': 36, 'Type814-17': 37, 'Type814-18': 38, 'Type814-19': 39, 'Type814-2': 40, 'Type814-3': 41, 'Type814-4': 42, 'Type814-5': 43, 'Type814-6': 44, 'Type814-7': 45, 'Type814-8': 46, 'Type814-9': 47, 'Type815-0': 48, 'Type815-1': 49, 'Type815-2': 50, 'Type815-3': 51, 'Type815-4': 52, 'Type815-5': 53, 'Type815-6': 54, 'Type815-7': 55, 'Type815-8': 56, 'Type815-9': 57, 'Type816-0': 58, 'Type816-1': 59, 'Type816-10': 60, 'Type816-11': 61, 'Type816-2': 62, 'Type816-3': 63, 'Type816-4': 64, 'Type816-5': 65, 'Type816-6': 66, 'Type816-7': 67, 'Type816-8': 68, 'Type816-9': 69, 'Type817-0': 70, 'Type817-1': 71, 'Type817-10': 72, 'Type817-2': 73, 'Type817-3': 74, 'Type817-4': 75, 'Type817-5': 76, 'Type817-6': 77, 'Type817-7': 78, 'Type817-8': 79, 'Type817-9': 80, 'Type818-0': 81, 'Type818-1': 82, 'Type818-2': 83, 'Type818-3': 84, 'Type819-0': 85, 'Type819-1': 86, 'Type819-2': 87, 'Type821-0': 88, 'Type821-1': 89, 'Type821-2': 90, 'Type821-3': 91, 'Type821-4': 92, 'Type821-5': 93, 'Type821-6': 94, 'Type821-7': 95, 'Type821-8': 96, 'Type821-9': 97, 'Type822-0': 98, 'Type822-1': 99, 'Type822-10': 100, 'Type822-11': 101, 'Type822-12': 102, 'Type822-13': 103, 'Type822-14': 104, 'Type822-15': 105, 'Type822-16': 106, 'Type822-17': 107, 'Type822-18': 108, 'Type822-19': 109, 'Type822-2': 110, 'Type822-3': 111, 'Type822-4': 112, 'Type822-5': 113, 'Type822-6': 114, 'Type822-7': 115, 'Type822-8': 116, 'Type822-9': 117, 'Type823-0': 118, 'Type823-1': 119, 'Type823-10': 120, 'Type823-11': 121, 'Type823-12': 122, 'Type823-13': 123, 'Type823-14': 124, 'Type823-15': 125, 'Type823-2': 126, 'Type823-3': 127, 'Type823-4': 128, 'Type823-5': 129, 'Type823-6': 130, 'Type823-7': 131, 'Type823-8': 132, 'Type823-9': 133, 'Type824-0': 134, 'Type824-1': 135, 'Type824-10': 136, 'Type824-11': 137, 'Type824-12': 138, 'Type824-13': 139, 'Type824-14': 140, 'Type824-15': 141, 'Type824-2': 142, 'Type824-3': 143, 'Type824-4': 144, 'Type824-5': 145, 'Type824-6': 146, 'Type824-7': 147, 'Type824-8': 148, 'Type824-9': 149, 'Type825-0': 150, 'Type825-1': 151, 'Type825-2': 152, 'Type825-3': 153, 'Type825-4': 154, 'Type825-5': 155, 'Type831-0': 156, 'Type831-1': 157, 'Type831-2': 158, 'Type831-3': 159, 'Type832-0': 160, 'Type832-1': 161, 'Type832-2': 162, 'Type832-3': 163, 'Type832-4': 164, 'Watt1T1A1': 165, 'Watt1T1A2': 166, 'Watt1T2A1': 167, 'Watt1T2A2': 168, 'Watt1T3A1': 169, 'Watt1T3A2': 170, 'Watt2T1A1': 171, 'Watt2T1A2': 172, 'Watt2T2A1': 173, 'Watt2T2A2': 174}

        self.max_mech_size = max_mech_size
        self.num_classes = num_classes

        self.transform = transforms.Compose([
                transforms.ToTensor(),  # Convert image to tensor
            ])

        for class_path in file_list:
            class_name = os.path.basename(class_path)

            img_paths = os.listdir(class_path)
            
            for img_path in img_paths:
                description = [x for x in img_path.split('_') if x]
                index = description.index(class_name)
                description = [float(x) for x in description[:index]]
                description = description + [0.5] * (22 - len(description))
                description = torch.tensor(description)
                img_path = os.path.join(class_path, img_path)
                
                # Create a tensor of zeros with size (1, num_categories)
                one_hot_encoding = torch.zeros(1, self.num_classes, dtype=torch.float32)

                # Set the corresponding index to 1
                one_hot_encoding[0, int(self.folder_dict[class_name])] = 1
                one_hot_encoding = one_hot_encoding.squeeze()
                
                self.data.append([img_path, description, one_hot_encoding])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path, mech, mech_type = self.data[idx]
        img = cv2.imread(img_path, 0)
        img_tensor = self.transform(img)        
    
        mech = mech.view(-1, 2) / 10.0
        match_tensor = torch.all(mech == torch.tensor([0.05, 0.05]), dim=1)
        dec_num_padding_tokens = torch.sum(match_tensor).item()

        # Create a mask that identifies rows not equal to [0.0, 0.0]
        mask = ~torch.all(mech == torch.tensor([0.05, 0.05]), dim=1)

        # Use the mask to filter out the rows
        mech = mech[mask]
        
        # Add only <s> token
        decoder_input = torch.cat(
            [
                torch.zeros(1, mech.size()[1], dtype=torch.float32),
                mech,
                torch.ones(dec_num_padding_tokens, mech.size()[1], dtype=torch.float32) * 0.5,
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                mech,
                torch.ones(1, mech.size()[1], dtype=torch.float32),
                torch.ones(dec_num_padding_tokens, mech.size()[1], dtype=torch.float32) * 0.5,
            ],
            dim=0,
        )
        
        mask = torch.cat((torch.ones(1, self.max_mech_size - dec_num_padding_tokens, dtype=torch.bool), torch.zeros(1, dec_num_padding_tokens, dtype=torch.bool)), dim=1)
        
        return {
            "encoder_input": img_tensor,  # (seq_len, d_input)
            "decoder_input": decoder_input,  # (seq_len, d_output)
            "label": label,  # (seq_len, d_output)
            "decoder_mask":  mask & causal_mask(decoder_input.size(0)), # (1, seq_len, seq_len),
            "mech_type": mech_type, 
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
