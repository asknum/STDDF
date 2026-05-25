import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model.STDDF_MiT import STDDF_MiT

class STDDF_FullLayerVis(STDDF_MiT):
    def __init__(self, pretrained=None):
        super(STDDF_FullLayerVis, self).__init__(pretrained=pretrained)

    def forward_all_layers(self, t1_input, t2_input):
        t1_list = self.backbone(t1_input)
        t2_list = self.backbone(t2_input)

        t1_x1, t1_x2, t1_x3, t1_x4 = t1_list[0], t1_list[1], t1_list[2], t1_list[3]
        t2_x1, t2_x2, t2_x3, t2_x4 = t2_list[0], t2_list[1], t2_list[2], t2_list[3]

        f1 = self.dem1(t1_x1, t2_x1)
        f2 = self.dem2(t1_x2, t2_x2)
        f3 = self.dem3(t1_x3, t2_x3)
        f4 = self.dem4(t1_x4, t2_x4)

        fe1 = self.egam1(f1)
        fe2 = self.egam2(f2)
        fe3 = self.egam3(f3)
        fe4 = self.egam4(f4)

        ff4 = self.conv4(fe4)
        
        ff3 = self.bgcaa43(ff4, fe3)
        ff3 = self.conv3(ff3)
        
        ff2 = self.bgcaa32(ff3, fe2)
        ff2 = self.conv2(ff2)
        
        ff1 = self.bgcaa21(ff2, fe1)
        ff1 = self.conv1(ff1)

        ff2_up = self.bgcaa21(ff2, ff1)
        ff3_up = self.bgcaa31(ff3, ff1)
        ff4_up = self.bgcaa41(ff4, ff1) 

        return {
            "T1_Backbone": [t1_x1, t1_x2, t1_x3, t1_x4],
            "T2_Backbone": [t2_x1, t2_x2, t2_x3, t2_x4],
            "DEM (Difference)": [f1, f2, f3, f4],
            "EGAM (Edge)": [fe1, fe2, fe3, fe4],
            "FGFM (Fusion)": [ff1, ff2_up, ff3_up, ff4_up],
            "FGFM_Raw": [ff1, ff2, ff3, ff4]
        }

def convert_to_heatmap(feature_tensor, target_size=(256, 256)):
    heatmap = torch.mean(feature_tensor, dim=1, keepdim=True)
    heatmap = F.interpolate(heatmap, size=target_size, mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().cpu().detach().numpy()
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    import argparse
    from PIL import Image
    from torchvision import transforms
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--t1', type=str, required=True, help='Path to T1 image')
    parser.add_argument('--t2', type=str, required=True, help='Path to T2 image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='feature_visualization_complete.png', help='Output image path')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"✓ Using device: {device}")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    t1_img = Image.open(args.t1).convert('RGB')
    t2_img = Image.open(args.t2).convert('RGB')
    t1 = transform(t1_img).unsqueeze(0).to(device)
    t2 = transform(t2_img).unsqueeze(0).to(device)
    
    model = STDDF_FullLayerVis()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded: {args.t1}")
    print(f"✓ Loaded: {args.t2}")
    print(f"✓ Loaded checkpoint: {args.checkpoint}")
    
    with torch.no_grad():
        features_dict = model.forward_all_layers(t1, t2)
    rows = ["T1_Backbone", "T2_Backbone", "DEM (Difference)", "EGAM (Edge)", "FGFM_Raw", "FGFM (Fusion)"]
    row_names = ["T1 Input", "T2 Input", "3D-DEM", "EGAM", "FGFM Raw", "FGFM Upsampled"]
    
    fig, axes = plt.subplots(6, 4, figsize=(16, 24))
    
    for r_idx, key in enumerate(rows):
        feature_list = features_dict[key]
        for c_idx, feat in enumerate(feature_list):
            heatmap_img = convert_to_heatmap(feat)
            
            ax = axes[r_idx, c_idx]
            ax.imshow(heatmap_img)
            ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(args.output, dpi=150, bbox_inches='tight', pad_inches=0)
    print(f"✓ Saved: {args.output} (6x4 grid, 24 feature maps)")
    plt.close()